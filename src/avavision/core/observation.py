"""What one camera frame shows: capture quality, pack registration and detected objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from .geometry import UNIT_RECT, Homography, Point, Quad, Rect
from .models import CellLocation, CellLocationKind, CompartmentIndex, MedicationID, PackLayout

# --------------------------------------------------------------------------- capture quality


class CaptureIssue(StrEnum):
    BLURRY = "blurry"
    TOO_DARK = "tooDark"
    TOO_BRIGHT = "tooBright"
    GLARE = "glare"
    ANALYSIS_FAILED = "analysisFailed"


class QualityMetrics(BaseModel):
    sharpness: float
    mean_luminance: float
    shadow_clip_fraction: float
    glare_fraction: float


class QualityAssessment(BaseModel):
    issues: list[CaptureIssue] = []
    metrics: QualityMetrics | None = None

    @property
    def is_acceptable(self) -> bool:
        return not self.issues


ACCEPTED = QualityAssessment()

# --------------------------------------------------------------------------- registration


class RegistrationIssue(StrEnum):
    PACK_NOT_FOUND = "packNotFound"
    LOW_DETECTOR_CONFIDENCE = "lowDetectorConfidence"
    PACK_NOT_CONVEX = "packNotConvex"
    PACK_TOO_SMALL = "packTooSmall"
    PACK_TOUCHES_FRAME_EDGE = "packTouchesFrameEdge"
    PERSPECTIVE_TOO_STEEP = "perspectiveTooSteep"
    ASPECT_RATIO_MISMATCH = "aspectRatioMismatch"


class RegistrationPolicy(BaseModel):
    minimum_area_fraction: float = 0.25
    minimum_detector_confidence: float = 0.7
    maximum_corner_angle_deviation: float = 20
    aspect_ratio_tolerance: float = 0.12
    minimum_edge_margin: float = 0.005


class Registration(BaseModel):
    model_config = ConfigDict(frozen=True)

    quad: Quad
    image_to_pack: Homography
    detector_confidence: float


class RegistrationOutcome(BaseModel):
    registration: Registration | None = None
    issues: list[RegistrationIssue] = []


PACK_CORNERS = [Point(x=0, y=0), Point(x=1, y=0), Point(x=1, y=1), Point(x=0, y=1)]


def register(
    quad: Quad | None,
    detector_confidence: float,
    image_width: int,
    image_height: int,
    layout: PackLayout,
    policy: RegistrationPolicy | None = None,
) -> RegistrationOutcome:
    """Validates a detected pack outline (normalized image coordinates) and builds image→pack mapping."""
    policy = policy or RegistrationPolicy()
    if quad is None or image_width <= 0 or image_height <= 0:
        return RegistrationOutcome(issues=[RegistrationIssue.PACK_NOT_FOUND])
    if not quad.is_convex:
        return RegistrationOutcome(issues=[RegistrationIssue.PACK_NOT_CONVEX])
    issues: list[RegistrationIssue] = []
    if detector_confidence < policy.minimum_detector_confidence:
        issues.append(RegistrationIssue.LOW_DETECTOR_CONFIDENCE)
    if quad.area < policy.minimum_area_fraction:
        issues.append(RegistrationIssue.PACK_TOO_SMALL)
    m = policy.minimum_edge_margin
    if any(c.x < m or c.y < m or c.x > 1 - m or c.y > 1 - m for c in quad.corners):
        issues.append(RegistrationIssue.PACK_TOUCHES_FRAME_EDGE)
    pixels = quad.scaled(image_width, image_height)
    if any(abs(a - 90) > policy.maximum_corner_angle_deviation for a in pixels.interior_angles):
        issues.append(RegistrationIssue.PERSPECTIVE_TOO_STEEP)
    if layout.is_calibrated:
        width = (pixels.top_left.distance(pixels.top_right) + pixels.bottom_left.distance(pixels.bottom_right)) / 2
        height = (pixels.top_left.distance(pixels.bottom_left) + pixels.top_right.distance(pixels.bottom_right)) / 2
        if height <= 0 or abs((width / height) / layout.aspect_ratio - 1) > policy.aspect_ratio_tolerance:
            issues.append(RegistrationIssue.ASPECT_RATIO_MISMATCH)
    if issues:
        return RegistrationOutcome(issues=issues)
    transform = Homography.mapping(quad.corners, PACK_CORNERS)
    if transform is None:
        return RegistrationOutcome(issues=[RegistrationIssue.PACK_NOT_CONVEX])
    return RegistrationOutcome(
        registration=Registration(quad=quad, image_to_pack=transform, detector_confidence=detector_confidence)
    )


# --------------------------------------------------------------------------- detections


class MeaningKind(StrEnum):
    PILL = "pill"
    MEDICATION = "medication"
    BROKEN = "broken"
    FOREIGN = "foreign"
    IGNORE = "ignore"


@dataclass(frozen=True)
class LabelMeaning:
    """What a model label means. Encoded as "pill", "broken", "foreign", "ignore" or "medication:<id>"."""

    kind: MeaningKind
    medication_id: MedicationID | None = None

    @classmethod
    def parse(cls, text: str) -> LabelMeaning | None:
        if text.startswith("medication:"):
            medication = text[len("medication:") :]
            return cls(MeaningKind.MEDICATION, medication) if medication else None
        try:
            kind = MeaningKind(text)
        except ValueError:
            return None
        return None if kind == MeaningKind.MEDICATION else cls(kind)

    def __str__(self) -> str:
        return f"medication:{self.medication_id}" if self.kind == MeaningKind.MEDICATION else self.kind.value


PILL = LabelMeaning(MeaningKind.PILL)


class Detection(BaseModel):
    label: str
    confidence: float
    box: Rect  # normalized image coordinates
    identity: IdentityEvidence | None = None


class FrameObservation(BaseModel):
    captured_at: datetime
    quality: QualityAssessment
    registration: RegistrationOutcome
    detections: list[Detection] = []
    image_sha256: str | None = None

    @property
    def is_usable(self) -> bool:
        return self.quality.is_acceptable and self.registration.registration is not None


@dataclass
class PlacedObject:
    detection_index: int
    meaning: LabelMeaning
    confidence: float
    identity: IdentityEvidence | None
    pack_center: Point | None
    location: CellLocation


@dataclass
class FrameAssignment:
    inside: dict[CompartmentIndex, list[PlacedObject]] = field(default_factory=dict)
    ambiguous: dict[CompartmentIndex, list[PlacedObject]] = field(default_factory=dict)
    outside_grid: list[PlacedObject] = field(default_factory=list)


def assign(detections: list[Detection], registration: Registration, layout: PackLayout, meaning) -> FrameAssignment:
    """Maps each detection's box centre to a compartment. ``ignore`` labels are dropped; everything else is
    kept whatever its confidence, so low-confidence objects trigger review instead of disappearing."""
    result = FrameAssignment()
    for index, detection in enumerate(detections):
        resolved: LabelMeaning = meaning(detection.label)
        if resolved.kind == MeaningKind.IGNORE:
            continue
        center = registration.image_to_pack.apply(detection.box.center)
        if center is None:
            result.outside_grid.append(
                PlacedObject(
                    index,
                    resolved,
                    detection.confidence,
                    detection.identity,
                    None,
                    CellLocation(kind=CellLocationKind.OUTSIDE),
                )
            )
            continue
        location = layout.locate(center)
        placed = PlacedObject(index, resolved, detection.confidence, detection.identity, center, location)
        if location.kind == CellLocationKind.INSIDE:
            result.inside.setdefault(location.compartments[0], []).append(placed)
        elif location.kind == CellLocationKind.BORDER:
            for compartment in location.compartments:
                result.ambiguous.setdefault(compartment, []).append(placed)
        elif UNIT_RECT.contains(center):
            result.outside_grid.append(placed)
    return result


def pill_compartments(
    frame: FrameObservation, layout: PackLayout, minimum_confidence: float, meaning
) -> dict[int, CompartmentIndex]:
    """Confident whole doses clearly inside one compartment, keyed by detection index."""
    if frame.registration.registration is None:
        return {}
    assignment = assign(frame.detections, frame.registration.registration, layout, meaning)
    result = {}
    for index, objects in assignment.inside.items():
        for obj in objects:
            if obj.confidence >= minimum_confidence and obj.meaning.kind in (MeaningKind.PILL, MeaningKind.MEDICATION):
                result[obj.detection_index] = index
    return result


from ..brain.identity import IdentityEvidence  # noqa: E402  (resolve forward reference)

Detection.model_rebuild()
FrameObservation.model_rebuild()
