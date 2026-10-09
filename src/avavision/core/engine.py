"""The decision engine: compares the observed pack with its expected profile.

Safety rules: weak, missing or conflicting evidence never produces an accepted result; ``verified`` needs
identity authority (a released model or earned brain trust) for every expected medication and a
calibrated layout. The brain can always escalate, never relax.
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from ..brain.identity import DecisionKind
from .gate import UNAVAILABLE, ActiveModel, CapabilityKind, ModelCapability
from .models import CompartmentExpectation, CompartmentIndex, MedicationID, PackLayout, PackProfile
from .observation import (
    CaptureIssue,
    FrameAssignment,
    FrameObservation,
    LabelMeaning,
    MeaningKind,
    PlacedObject,
    RegistrationIssue,
    assign,
)
from .physical import fits_none

# --------------------------------------------------------------------------- verdicts


class CompartmentStatus(StrEnum):
    VERIFIED = "verified"  # count and identity confirmed; the pharmacist still signs off the pack
    COUNT_MATCHED = "countMatched"  # count matches; identity not checked automatically
    NEEDS_REVIEW = "needsReview"  # weak, conflicting or missing evidence
    MISMATCH = "mismatch"  # confident evidence the content is wrong

    @property
    def severity(self) -> int:
        return list(CompartmentStatus).index(self)


class FindingKind(StrEnum):
    MISSING = "missing"
    EXTRA = "extra"
    WRONG_QUANTITY = "wrongQuantity"
    UNEXPECTED_MEDICATION = "unexpectedMedication"
    BROKEN_DOSE = "brokenDose"
    FOREIGN_OBJECT = "foreignObject"
    LOW_CONFIDENCE_OBJECT = "lowConfidenceObject"
    OBJECT_ON_BORDER = "objectOnBorder"
    UNSTABLE_ACROSS_FRAMES = "unstableAcrossFrames"
    IDENTITY_NOT_VERIFIED = "identityNotVerified"
    SUSPECTED_MEDICATION = "suspectedMedication"
    UNRECOGNISED_PILL = "unrecognisedPill"
    CONFLICTING_IDENTITY = "conflictingIdentity"
    ADVISOR_DISAGREES = "advisorDisagrees"  # an advisory second opinion saw something else; see core.advisory
    PHYSICAL_MISMATCH = "physicalMismatch"  # size or colour fits none of the expected medications; core.physical
    VIEW_OBSCURED = "viewObscured"  # glare or a covering hides part of the compartment; its count proves nothing
    PACK_NOT_SEEN = "packNotSeen"  # the camera found far too few doses in the whole pack; see DecisionPolicy
    NO_EXPECTATION = "noExpectation"
    LAYOUT_UNCALIBRATED = "layoutUncalibrated"
    NOT_EVALUATED = "notEvaluated"


_MISMATCH = {
    FindingKind.MISSING,
    FindingKind.EXTRA,
    FindingKind.WRONG_QUANTITY,
    FindingKind.UNEXPECTED_MEDICATION,
    FindingKind.BROKEN_DOSE,
    FindingKind.FOREIGN_OBJECT,
}


class Finding(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: FindingKind
    medication_id: MedicationID | None = None
    expected: int | None = None
    observed: int | None = None
    count: int | None = None

    @property
    def minimum_status(self) -> CompartmentStatus:
        if self.kind in _MISMATCH:
            return CompartmentStatus.MISMATCH
        if self.kind == FindingKind.IDENTITY_NOT_VERIFIED:
            return CompartmentStatus.COUNT_MATCHED
        return CompartmentStatus.NEEDS_REVIEW


def finding(kind: FindingKind, **kwargs) -> Finding:
    return Finding(kind=kind, **kwargs)


class CompartmentVerdict(BaseModel):
    compartment: CompartmentIndex
    status: CompartmentStatus
    findings: list[Finding]
    expected_count: int | None
    observed_count: int | None
    observed_medications: dict[MedicationID, int] = {}

    @classmethod
    def build(cls, compartment, findings, expected_count, observed_count, observed_medications=None):
        status = max((f.minimum_status for f in findings), key=lambda s: s.severity, default=CompartmentStatus.VERIFIED)
        return cls(
            compartment=compartment,
            status=status,
            findings=findings,
            expected_count=expected_count,
            observed_count=observed_count,
            observed_medications=observed_medications or {},
        )


class PackStatus(StrEnum):
    VERIFIED = "verified"
    COUNT_MATCHED = "countMatched"
    NEEDS_REVIEW = "needsReview"
    MISMATCH = "mismatch"
    RETAKE_REQUIRED = "retakeRequired"


class PackFindingKind(StrEnum):
    INSUFFICIENT_USABLE_FRAMES = "insufficientUsableFrames"
    CAPTURE_ISSUES = "captureIssues"
    REGISTRATION_ISSUES = "registrationIssues"
    OBJECTS_OUTSIDE_COMPARTMENTS = "objectsOutsideCompartments"
    MODEL_UNAVAILABLE = "modelUnavailable"
    PROFILE_LAYOUT_MISMATCH = "profileLayoutMismatch"
    PACK_CARD_MISMATCH = "packCardMismatch"  # the camera saw another pack's header card; core.card
    TOO_FEW_DOSES_SEEN = "tooFewDosesSeen"  # the detector is not seeing this pack; see DecisionPolicy


class PackFinding(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: PackFindingKind
    usable: int | None = None
    required: int | None = None
    count: int | None = None
    capture_issues: tuple[CaptureIssue, ...] = ()
    registration_issues: tuple[RegistrationIssue, ...] = ()
    codes: tuple[str, ...] = ()


class VerificationResult(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    evaluated_at: datetime
    layout_id: str
    profile_id: uuid.UUID
    model_id: str | None
    model_version: str | None
    capability: ModelCapability
    trusted_medications: list[MedicationID] = []
    status: PackStatus
    pack_findings: list[PackFinding]
    compartments: list[CompartmentVerdict]
    #: Verified compartments chosen at random for mandatory pharmacist inspection.
    spot_checks: list[CompartmentIndex] = []
    usable_frame_count: int

    def verdict(self, index: CompartmentIndex) -> CompartmentVerdict | None:
        return next((v for v in self.compartments if v.compartment == index), None)

    @property
    def compartments_requiring_review(self) -> list[CompartmentIndex]:
        spot = set(self.spot_checks)
        return [
            v.compartment for v in self.compartments if v.status != CompartmentStatus.VERIFIED or v.compartment in spot
        ]


# --------------------------------------------------------------------------- engine


class DecisionPolicy(BaseModel):
    required_consistent_frames: int = 3
    #: Detections below this confidence are never counted; they send the compartment to review.
    minimum_detection_confidence: float = 0.6
    #: Fraction of verified compartments still inspected (at least one per pack when any is verified).
    spot_check_rate: float = 0.1
    #: When fewer than this fraction of the doses the whole pack should hold are found, the detector is not
    #: seeing the pack (on the first real photo the classic segmenter found none): a compartment that agrees,
    #: such as an empty one expected empty, proves nothing then.
    minimum_seen_fraction: float = 0.5


class BrainContext(BaseModel):
    trusted_medications: frozenset[MedicationID] = frozenset()
    well_known_medications: frozenset[MedicationID] = frozenset()


@dataclass
class _Tally:
    generic: int = 0
    medications: dict[MedicationID, int] = field(default_factory=dict)
    broken: int = 0
    foreign: int = 0

    @property
    def doses(self) -> int:
        return self.generic + sum(self.medications.values())


class VerificationEngine:
    def __init__(
        self,
        layout: PackLayout,
        model: ActiveModel | None,
        brain: BrainContext | None = None,
        policy: DecisionPolicy | None = None,
    ):
        self.layout = layout
        self.model = model
        self.brain = brain
        self.policy = policy or DecisionPolicy()

    def can_identify(self, medication: MedicationID) -> bool:
        by_model = self.model is not None and self.model.capability.can_identify(medication)
        return by_model or (self.brain is not None and medication in self.brain.trusted_medications)

    @property
    def has_identity_authority(self) -> bool:
        if self.model is not None and self.model.capability.kind == CapabilityKind.IDENTITY:
            return True
        return self.brain is not None and bool(self.brain.trusted_medications)

    def evaluate(
        self,
        profile: PackProfile,
        frames: list[FrameObservation],
        evaluated_at: datetime | None = None,
        result_id: uuid.UUID | None = None,
    ) -> VerificationResult:
        evaluated_at = evaluated_at or datetime.now(UTC)
        result_id = result_id or uuid.uuid4()
        required = max(1, self.policy.required_consistent_frames)
        usable = [f for f in frames if f.is_usable]

        def unevaluated(status: PackStatus, findings: list[PackFinding]) -> VerificationResult:
            verdicts = [
                CompartmentVerdict.build(
                    i,
                    [finding(FindingKind.NOT_EVALUATED)],
                    (e.total_quantity if (e := profile.expectation(i)) else None),
                    None,
                )
                for i in self.layout.all_compartments
            ]
            return self._result(result_id, profile, status, findings, verdicts, [], len(usable), evaluated_at)

        if self.model is None or not self.model.capability.can_count:
            return unevaluated(PackStatus.NEEDS_REVIEW, [PackFinding(kind=PackFindingKind.MODEL_UNAVAILABLE)])
        if profile.layout_id != self.layout.id:
            return unevaluated(PackStatus.NEEDS_REVIEW, [PackFinding(kind=PackFindingKind.PROFILE_LAYOUT_MISMATCH)])

        window = [f for f in usable[-required:] if f.registration.registration is not None]
        if len(window) != required:
            findings = [
                PackFinding(kind=PackFindingKind.INSUFFICIENT_USABLE_FRAMES, usable=len(usable), required=required)
            ]
            rejected = [f for f in frames if not f.is_usable]
            capture = {i for f in rejected for i in f.quality.issues}
            registration = {i for f in rejected for i in f.registration.issues}
            if capture:
                findings.append(
                    PackFinding(
                        kind=PackFindingKind.CAPTURE_ISSUES,
                        capture_issues=tuple(i for i in CaptureIssue if i in capture),
                    )
                )
            if registration:
                findings.append(
                    PackFinding(
                        kind=PackFindingKind.REGISTRATION_ISSUES,
                        registration_issues=tuple(i for i in RegistrationIssue if i in registration),
                    )
                )
            return unevaluated(PackStatus.RETAKE_REQUIRED, findings)

        assignments = [
            assign(f.detections, f.registration.registration, self.layout, self.model.meaning) for f in window
        ]
        obscured = {i for f in window for i in f.obscured}
        verdicts = [
            self._verdict(i, profile.expectation(i), assignments, i in obscured) for i in self.layout.all_compartments
        ]

        pack_findings = []
        expected_total = sum(v.expected_count or 0 for v in verdicts)
        seen = sum(v.observed_count or 0 for v in verdicts)
        if expected_total and seen < self.policy.minimum_seen_fraction * expected_total:
            pack_findings.append(
                PackFinding(kind=PackFindingKind.TOO_FEW_DOSES_SEEN, count=seen, required=expected_total)
            )
            accepted = (CompartmentStatus.VERIFIED, CompartmentStatus.COUNT_MATCHED)
            verdicts = [
                CompartmentVerdict.build(
                    v.compartment,
                    [*v.findings, finding(FindingKind.PACK_NOT_SEEN)],
                    v.expected_count,
                    v.observed_count,
                    v.observed_medications,
                )
                if v.status in accepted
                else v
                for v in verdicts
            ]
        outside = max(
            (
                sum(
                    1
                    for o in a.outside_grid
                    if o.confidence >= self.policy.minimum_detection_confidence
                    and o.meaning.kind != MeaningKind.OBSCURED
                )
                for a in assignments
            ),
            default=0,
        )
        if outside:
            pack_findings.append(PackFinding(kind=PackFindingKind.OBJECTS_OUTSIDE_COMPARTMENTS, count=outside))
        worst = max((v.status for v in verdicts), key=lambda s: s.severity, default=CompartmentStatus.VERIFIED)
        status = PackStatus(worst.value)
        if pack_findings and status in (PackStatus.VERIFIED, PackStatus.COUNT_MATCHED):
            status = PackStatus.NEEDS_REVIEW
        return self._result(
            result_id,
            profile,
            status,
            pack_findings,
            verdicts,
            self._spot_checks(verdicts, result_id),
            len(usable),
            evaluated_at,
        )

    def _spot_checks(self, verdicts: list[CompartmentVerdict], seed: uuid.UUID) -> list[CompartmentIndex]:
        verified = [v.compartment for v in verdicts if v.status == CompartmentStatus.VERIFIED]
        if not verified or self.policy.spot_check_rate <= 0:
            return []
        rng = random.Random(seed.int)
        chosen = [c for c in verified if rng.random() < self.policy.spot_check_rate]
        return sorted(chosen or [rng.choice(verified)])

    def _resolved(self, obj: PlacedObject) -> tuple[LabelMeaning, bool]:
        """Final meaning once trusted brain identification is taken into account; flags model/brain conflict."""
        named = obj.identity.decision.named if obj.identity else None
        if named is not None and not self.can_identify(named):
            named = None
        if obj.meaning.kind == MeaningKind.PILL:
            return (LabelMeaning(MeaningKind.MEDICATION, named) if named else obj.meaning), False
        if obj.meaning.kind == MeaningKind.MEDICATION and named and named != obj.meaning.medication_id:
            return LabelMeaning(MeaningKind.PILL), True
        return obj.meaning, False

    def _verdict(
        self,
        index: CompartmentIndex,
        expectation: CompartmentExpectation | None,
        assignments: list[FrameAssignment],
        obscured: bool = False,
    ) -> CompartmentVerdict:
        """``obscured``: a frame showed glare or a covering over this compartment (``FrameObservation.obscured``)."""
        expected = {k: v for k, v in (expectation.quantities if expectation else {}).items() if v > 0}
        findings: list[Finding] = []
        tallies: list[_Tally] = []
        low = border = conflicts = unrecognised = physical = 0
        suspected: set[MedicationID] = set()
        notices_strangers = (
            bool(expected) and self.brain is not None and all(m in self.brain.well_known_medications for m in expected)
        )
        for a in assignments:
            tally = _Tally()
            low_here = conflicts_here = unrecognised_here = physical_here = 0
            for obj in a.inside.get(index, []):
                if obj.meaning.kind == MeaningKind.OBSCURED:  # whatever its confidence
                    obscured = True
                    continue
                if obj.confidence < self.policy.minimum_detection_confidence:
                    low_here += 1
                    continue
                meaning, conflict = self._resolved(obj)
                conflicts_here += conflict
                if meaning.kind in (MeaningKind.PILL, MeaningKind.MEDICATION) and fits_none(
                    obj.physical, set(expected)
                ):
                    physical_here += 1
                if meaning.kind == MeaningKind.PILL:
                    tally.generic += 1
                    decision = obj.identity.decision if obj.identity else None
                    if decision and decision.named and decision.named not in expected:
                        suspected.add(decision.named)
                    elif decision and decision.kind == DecisionKind.UNRECOGNISED and notices_strangers:
                        unrecognised_here += 1
                elif meaning.kind == MeaningKind.MEDICATION:
                    tally.medications[meaning.medication_id] = tally.medications.get(meaning.medication_id, 0) + 1
                elif meaning.kind == MeaningKind.BROKEN:
                    tally.broken += 1
                elif meaning.kind == MeaningKind.FOREIGN:
                    tally.foreign += 1
            tallies.append(tally)
            low = max(low, low_here)
            border = max(border, len(a.ambiguous.get(index, [])))
            conflicts = max(conflicts, conflicts_here)
            unrecognised = max(unrecognised, unrecognised_here)
            physical = max(physical, physical_here)

        if low:
            findings.append(finding(FindingKind.LOW_CONFIDENCE_OBJECT, count=low))
        if border:
            findings.append(finding(FindingKind.OBJECT_ON_BORDER, count=border))
        if not self.layout.is_calibrated:
            findings.append(finding(FindingKind.LAYOUT_UNCALIBRATED))
        if conflicts:
            findings.append(finding(FindingKind.CONFLICTING_IDENTITY, count=conflicts))
        findings += [finding(FindingKind.SUSPECTED_MEDICATION, medication_id=m) for m in sorted(suspected)]
        if unrecognised:
            findings.append(finding(FindingKind.UNRECOGNISED_PILL, count=unrecognised))
        if physical:
            findings.append(finding(FindingKind.PHYSICAL_MISMATCH, count=physical))
        if obscured:
            findings.append(finding(FindingKind.VIEW_OBSCURED))

        consensus = tallies[0] if tallies else None
        if consensus is None or any(t != consensus for t in tallies):
            findings.append(finding(FindingKind.UNSTABLE_ACROSS_FRAMES))
            if expectation is None:
                findings.append(finding(FindingKind.NO_EXPECTATION))
            return CompartmentVerdict.build(index, findings, expectation.total_quantity if expectation else None, None)

        if consensus.foreign:
            findings.append(finding(FindingKind.FOREIGN_OBJECT, count=consensus.foreign))
        if consensus.broken:
            findings.append(finding(FindingKind.BROKEN_DOSE, count=consensus.broken))
        if expectation is None:
            findings.append(finding(FindingKind.NO_EXPECTATION))
            return CompartmentVerdict.build(index, findings, None, consensus.doses, consensus.medications)

        # Low-confidence and border objects might be doses here: a count difference is certain only if
        # they could not explain it. Otherwise the review findings above already apply. Behind glare or a
        # covering anything could hide (or glare could look like a tablet), so no count is certain there.
        expected_count = expectation.total_quantity
        if not obscured and consensus.doses + low + border < expected_count:
            findings.append(finding(FindingKind.MISSING, expected=expected_count, observed=consensus.doses))
        elif not obscured and consensus.doses > expected_count:
            findings.append(finding(FindingKind.EXTRA, expected=expected_count, observed=consensus.doses))
        for medication, count in sorted(consensus.medications.items()):
            if medication not in expected:
                findings.append(finding(FindingKind.UNEXPECTED_MEDICATION, medication_id=medication, observed=count))

        if self.has_identity_authority and consensus.generic == 0 and all(self.can_identify(m) for m in expected):
            for medication, quantity in sorted(expected.items()):
                observed = consensus.medications.get(medication, 0)
                if observed != quantity and not obscured:
                    findings.append(
                        finding(
                            FindingKind.WRONG_QUANTITY, medication_id=medication, expected=quantity, observed=observed
                        )
                    )
        else:
            findings.append(finding(FindingKind.IDENTITY_NOT_VERIFIED))
        return CompartmentVerdict.build(index, findings, expected_count, consensus.doses, consensus.medications)

    def _result(self, result_id, profile, status, findings, verdicts, spot_checks, usable, evaluated_at):
        return VerificationResult(
            id=result_id,
            evaluated_at=evaluated_at,
            layout_id=self.layout.id,
            profile_id=profile.id,
            model_id=self.model.manifest.model_id if self.model else None,
            model_version=self.model.manifest.version if self.model else None,
            capability=self.model.capability if self.model else UNAVAILABLE,
            trusted_medications=sorted(self.brain.trusted_medications) if self.brain else [],
            status=status,
            pack_findings=findings,
            compartments=verdicts,
            spot_checks=spot_checks,
            usable_frame_count=usable,
        )
