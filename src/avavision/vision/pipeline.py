"""Frame analysis in three steps:

1. ``locate`` — pack position and capture quality (fast; runs on every live frame).
2. ``detect`` — pills in every compartment (on the few frames that are evaluated).
3. ``identify`` — the brain's opinion of every confident pill (batched on the GPU when available).
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime

import cv2
import numpy as np

from ..brain.identity import IdentityClassifier, IdentityEvidence
from ..core.gate import ActiveModel
from ..core.geometry import Rect
from ..core.models import CompartmentIndex, PackLayout
from ..core.observation import CaptureIssue, FrameObservation, RegistrationOutcome, pill_compartments, register
from .embedder import Embedder
from .pack_finder import FinderMode, Orientation, find_pack
from .quality import QualityPolicy, assess
from .segmentation import PocketSegmenter, Rectified, rectify


@dataclass
class IdentifiedPill:
    detection_index: int
    compartment: CompartmentIndex
    crop: np.ndarray  # BGR, perspective-corrected
    vector: np.ndarray
    identity: IdentityEvidence


@dataclass
class AnalyzedFrame:
    observation: FrameObservation
    image: np.ndarray
    rectified: Rectified | None = None
    pills: list[IdentifiedPill] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)


class FrameAnalyzer:
    def __init__(
        self,
        layout: PackLayout,
        mode: FinderMode = FinderMode.MARKERS,
        orientation: Orientation = Orientation.AUTOMATIC,
        segmenter: PocketSegmenter | None = None,
        detector=None,
        quality_policy: QualityPolicy | None = None,
        locate_size: int = 1600,
    ):
        self.layout = layout
        self.mode = mode
        self.orientation = orientation
        self.segmenter = segmenter or PocketSegmenter()
        self.detector = detector  # optional trained ONNX detector; replaces the segmenter when present
        self.quality_policy = quality_policy or QualityPolicy()
        self.locate_size = locate_size

    def locate(self, image: np.ndarray, captured_at: datetime | None = None) -> AnalyzedFrame:
        start = time.perf_counter()
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        scale = min(1.0, self.locate_size / max(h, w))
        small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
        found = find_pack(small, self.mode, self.layout, self.orientation)
        outcome: RegistrationOutcome = register(
            found.quad if found else None, found.confidence if found else 0.0, w, h, self.layout
        )
        region = found.quad.bounding_box if found else None
        quality = assess(small, region, self.quality_policy)
        digest = hashlib.sha256(small.tobytes()).hexdigest()
        observation = FrameObservation(
            captured_at=captured_at or datetime.now(UTC), quality=quality, registration=outcome, image_sha256=digest
        )
        return AnalyzedFrame(observation=observation, image=image, timings_ms={"locate": _ms(start)})

    def detect(self, frame: AnalyzedFrame) -> AnalyzedFrame:
        registration = frame.observation.registration.registration
        if not frame.observation.is_usable or registration is None:
            return frame
        start = time.perf_counter()
        try:
            frame.rectified = rectify(frame.image, registration, self.layout)
            if self.detector is not None:
                detections = self.detector.detect(frame.image)
            else:
                detections = self.segmenter.detect(frame.rectified, self.layout)
            frame.observation = frame.observation.model_copy(update={"detections": detections})
        except Exception:  # a failed detection makes the frame unusable instead of guessing
            quality = frame.observation.quality.model_copy(
                update={"issues": [*frame.observation.quality.issues, CaptureIssue.ANALYSIS_FAILED]}
            )
            frame.observation = frame.observation.model_copy(update={"quality": quality})
        frame.timings_ms["detect"] = _ms(start)
        return frame

    def identify(
        self,
        frame: AnalyzedFrame,
        embedder: Embedder,
        classifier: IdentityClassifier,
        model: ActiveModel,
        minimum_confidence: float,
    ) -> AnalyzedFrame:
        if frame.rectified is None:
            return frame
        start = time.perf_counter()
        located = pill_compartments(frame.observation, self.layout, minimum_confidence, model.meaning)
        indices = sorted(located)
        crops = [pill_crop(frame.rectified, frame.observation.detections[i].box) for i in indices]
        keep = [k for k, crop in enumerate(crops) if crop is not None]
        if keep:
            vectors = embedder.embed([crops[k] for k in keep])
            evidence = classifier.classify_many(vectors)
            detections = list(frame.observation.detections)
            for k, vector, ev in zip(keep, vectors, evidence, strict=True):
                i = indices[k]
                detections[i] = detections[i].model_copy(update={"identity": ev})
                frame.pills.append(IdentifiedPill(i, located[i], crops[k], vector, ev))
            frame.observation = frame.observation.model_copy(update={"detections": detections})
        frame.timings_ms["identify"] = _ms(start)
        return frame


def pill_crop(rectified: Rectified, box: Rect, padding: float = 0.2) -> np.ndarray | None:
    """A square, perspective-corrected crop around one pill."""
    x0, y0, x1, y1 = rectified.image_box_to_canvas(box)
    side = max(x1 - x0, y1 - y0) * (1 + 2 * padding)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    canvas = rectified.canvas
    a, b = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    c, d = int(min(canvas.shape[1], cx + side / 2)), int(min(canvas.shape[0], cy + side / 2))
    return canvas[b:d, a:c].copy() if c - a >= 4 and d - b >= 4 else None


def compartment_crop(
    rectified: Rectified, layout: PackLayout, index: CompartmentIndex, margin: float = 0.15
) -> np.ndarray:
    """One compartment of the perspective-corrected pack, with a margin of its neighbours for context."""
    rect = layout.cell_rect(index)
    canvas = rectified.canvas
    h, w = canvas.shape[:2]
    dx, dy = rect.width * margin, rect.height * margin
    x0, y0 = max(0, int((rect.x - dx) * w)), max(0, int((rect.y - dy) * h))
    x1, y1 = min(w, int((rect.max_x + dx) * w)), min(h, int((rect.max_y + dy) * h))
    return canvas[y0:y1, x0:x1].copy()


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


def propagate_identities(source: AnalyzedFrame, target: AnalyzedFrame, layout: PackLayout, tolerance: float = 0.02):
    """Copies the brain's opinions from ``source`` to the matching pills of ``target`` (same compartment, pack
    position within ``tolerance``), so pills are embedded once per check instead of once per frame. A pill
    without a match keeps no opinion, which makes the frames disagree and sends it to review."""
    src_reg = source.observation.registration.registration
    dst_reg = target.observation.registration.registration
    if src_reg is None or dst_reg is None or source is target:
        return target

    def centre(registration, box):
        return registration.image_to_pack.apply(box.center)

    anchors = []
    for pill in source.pills:
        c = centre(src_reg, source.observation.detections[pill.detection_index].box)
        if c is not None:
            anchors.append((c, pill))
    detections = list(target.observation.detections)
    used: set[int] = set()
    for i, detection in enumerate(detections):
        c = centre(dst_reg, detection.box)
        if c is None:
            continue
        best = min(
            ((c.distance(a), k) for k, (a, _) in enumerate(anchors) if k not in used),
            default=None,
        )
        if best is not None and best[0] <= tolerance:
            used.add(best[1])
            detections[i] = detection.model_copy(update={"identity": anchors[best[1]][1].identity})
    target.observation = target.observation.model_copy(update={"detections": detections})
    return target
