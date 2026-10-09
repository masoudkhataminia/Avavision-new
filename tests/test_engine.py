from __future__ import annotations

import uuid

from avavision.brain.identity import UNRECOGNISED, IdentityDecision
from avavision.core.engine import (
    BrainContext,
    CompartmentStatus,
    FindingKind,
    PackFindingKind,
    PackStatus,
    VerificationEngine,
    finding,
)
from avavision.core.geometry import Point, Rect
from avavision.core.models import ExpectedItem, cell
from avavision.core.observation import (
    ACCEPTED,
    CaptureIssue,
    FrameObservation,
    QualityAssessment,
    RegistrationIssue,
    RegistrationOutcome,
    pill_compartments,
)
from conftest import (
    ASPIRIN,
    ATORVASTATIN,
    FIXED,
    LAYOUT,
    METFORMIN,
    ONE,
    TARGET,
    detection,
    detections,
    engine,
    frame,
    frames,
    full_pack,
    identity_model,
    profile,
    with_identity,
)


def evaluate(dets, prof=None, eng=None):
    return (eng or engine()).evaluate(prof or profile(ONE), frames(dets), evaluated_at=FIXED)


def kinds(verdict):
    return [f.kind for f in verdict.findings]


def test_count_only_never_verifies():
    result = evaluate(full_pack())
    assert result.status == PackStatus.COUNT_MATCHED
    assert all(
        v.status == CompartmentStatus.COUNT_MATCHED and kinds(v) == [FindingKind.IDENTITY_NOT_VERIFIED]
        for v in result.compartments
    )
    assert len(result.compartments_requiring_review) == 6


def test_missing_and_extra_are_mismatches():
    others = [d for i in LAYOUT.all_compartments if i != TARGET for d in detections("pill", 1, i)]
    missing = evaluate(others).verdict(TARGET)
    assert missing.status == CompartmentStatus.MISMATCH
    assert finding(FindingKind.MISSING, expected=1, observed=0) in missing.findings
    shifted = [
        d.model_copy(update={"box": d.box.model_copy(update={"x": d.box.x + 0.03})})
        for d in detections("pill", 1, TARGET)
    ]
    extra = evaluate(full_pack() + shifted).verdict(TARGET)
    assert finding(FindingKind.EXTRA, expected=1, observed=2) in extra.findings


def test_frames_that_disagree_require_review():
    fs = [frame(full_pack()), frame(full_pack() + detections("pill", 2, TARGET)[1:]), frame(full_pack())]
    result = engine().evaluate(profile(ONE), fs)
    v = result.verdict(TARGET)
    assert (
        v.status == CompartmentStatus.NEEDS_REVIEW
        and FindingKind.UNSTABLE_ACROSS_FRAMES in kinds(v)
        and v.observed_count is None
    )


def test_low_confidence_object_is_never_counted_silently():
    v = evaluate(full_pack() + detections("pill", 2, TARGET, confidence=0.4)[1:]).verdict(TARGET)
    assert (
        v.status == CompartmentStatus.NEEDS_REVIEW and finding(FindingKind.LOW_CONFIDENCE_OBJECT, count=1) in v.findings
    )
    assert v.observed_count == 1


def test_uncertain_objects_make_short_count_uncertain_not_missing():
    others = [d for i in LAYOUT.all_compartments if i != TARGET for d in detections("pill", 1, i)]
    v = evaluate(others + detections("pill", 1, TARGET, confidence=0.4)).verdict(TARGET)
    assert v.status == CompartmentStatus.NEEDS_REVIEW and FindingKind.MISSING not in kinds(v)
    three = profile(ONE, {TARGET: [ExpectedItem(medication_id=METFORMIN, quantity=3)]})
    v = evaluate(others + detections("pill", 1, TARGET, confidence=0.4), three).verdict(TARGET)
    assert finding(FindingKind.MISSING, expected=3, observed=0) in v.findings and v.status == CompartmentStatus.MISMATCH


def test_object_on_border_flags_both_compartments():
    left = cell(0, 0)
    r = LAYOUT.cell_rect(left)
    result = evaluate([*full_pack(), detection("pill", Point(x=r.max_x - 0.01, y=r.center.y))])
    for index in (left, TARGET):
        v = result.verdict(index)
        assert (
            v.status == CompartmentStatus.NEEDS_REVIEW and finding(FindingKind.OBJECT_ON_BORDER, count=1) in v.findings
        )


def test_foreign_and_broken_are_mismatches_and_ignored_labels_do_not_count():
    other = cell(1, 2)
    result = evaluate(full_pack() + detections("foreign", 1, TARGET) + detections("broken", 1, other))
    assert result.verdict(TARGET).status == CompartmentStatus.MISMATCH
    assert finding(FindingKind.BROKEN_DOSE, count=1) in result.verdict(other).findings
    assert (
        evaluate(full_pack() + detections("background", 1, TARGET)).verdict(TARGET).status
        == CompartmentStatus.COUNT_MATCHED
    )


def test_unknown_labels_count_as_pills():
    shifted = [
        d.model_copy(update={"box": d.box.model_copy(update={"x": d.box.x + 0.03})})
        for d in detections("mystery", 1, TARGET)
    ]
    assert evaluate(full_pack() + shifted).verdict(TARGET).status == CompartmentStatus.MISMATCH


def test_too_few_usable_frames_requires_retake():
    blurry = QualityAssessment(issues=[CaptureIssue.BLURRY, CaptureIssue.GLARE])
    lost = FrameObservation(
        captured_at=FIXED, quality=ACCEPTED, registration=RegistrationOutcome(issues=[RegistrationIssue.PACK_TOO_SMALL])
    )
    result = engine().evaluate(profile(ONE), [frame(full_pack()), frame(full_pack(), blurry), lost, frame(full_pack())])
    assert result.status == PackStatus.RETAKE_REQUIRED
    assert [f.kind for f in result.pack_findings] == [
        PackFindingKind.INSUFFICIENT_USABLE_FRAMES,
        PackFindingKind.CAPTURE_ISSUES,
        PackFindingKind.REGISTRATION_ISSUES,
    ]
    assert result.pack_findings[1].capture_issues == (CaptureIssue.BLURRY, CaptureIssue.GLARE)
    assert all(kinds(v) == [FindingKind.NOT_EVALUATED] for v in result.compartments)


def test_without_model_or_with_wrong_layout_nothing_is_evaluated():
    assert (
        evaluate(full_pack(), eng=VerificationEngine(LAYOUT, None)).pack_findings[0].kind
        == PackFindingKind.MODEL_UNAVAILABLE
    )
    p = profile(ONE)
    p.layout_id = "other"
    assert evaluate(full_pack(), p).pack_findings[0].kind == PackFindingKind.PROFILE_LAYOUT_MISMATCH


def test_uncalibrated_layout_cannot_accept():
    layout = LAYOUT.model_copy(update={"is_calibrated": False})
    result = evaluate(full_pack(), eng=engine(layout=layout))
    assert result.status == PackStatus.NEEDS_REVIEW
    assert evaluate(full_pack()[1:], eng=engine(layout=layout)).status == PackStatus.MISMATCH


def test_objects_outside_grid_send_pack_to_review():
    layout = LAYOUT.model_copy(update={"grid_region": Rect(x=0.1, y=0.1, width=0.8, height=0.8)})
    inside = [detection("pill", layout.cell_rect(i).center) for i in layout.all_compartments]
    result = engine(layout=layout).evaluate(profile(ONE), frames([*inside, detection("pill", Point(x=0.03, y=0.5))]))
    assert (
        result.pack_findings[0].kind == PackFindingKind.OBJECTS_OUTSIDE_COMPARTMENTS
        and result.status == PackStatus.NEEDS_REVIEW
    )


def test_expected_empty_compartments_must_be_empty():
    p = profile([], {TARGET: ONE})
    assert evaluate(detections("pill", 1, TARGET), p).status == PackStatus.COUNT_MATCHED
    assert evaluate(full_pack(), p).verdict(cell(0, 0)).findings[0] == finding(
        FindingKind.EXTRA, expected=0, observed=1
    )


# --------------------------------------------------------------------------- identity


IDENTITY_PROFILE = profile(
    [],
    {TARGET: [ExpectedItem(medication_id=METFORMIN, quantity=1), ExpectedItem(medication_id=ATORVASTATIN, quantity=1)]},
)


def two_in_target(a: str, b: str):
    r = LAYOUT.cell_rect(TARGET)
    return [
        detection(a, Point(x=r.center.x - 0.03, y=r.center.y)),
        detection(b, Point(x=r.center.x + 0.03, y=r.center.y)),
    ]


def test_released_identity_model_verifies_with_spot_checks():
    eng = engine(identity_model({METFORMIN, ATORVASTATIN}))
    result = evaluate(two_in_target("metformin", "atorvastatin"), IDENTITY_PROFILE, eng)
    v = result.verdict(TARGET)
    assert (
        v.status == CompartmentStatus.VERIFIED
        and v.findings == []
        and v.observed_medications == {METFORMIN: 1, ATORVASTATIN: 1}
    )
    assert (
        result.status == PackStatus.VERIFIED
        and result.spot_checks
        and result.compartments_requiring_review == result.spot_checks
    )
    eng.policy.spot_check_rate = 0
    assert (
        evaluate(two_in_target("metformin", "atorvastatin"), IDENTITY_PROFILE, eng).compartments_requiring_review == []
    )


def test_spot_checks_are_deterministic_per_result():
    eng = engine(identity_model({METFORMIN, ATORVASTATIN}))
    rid = uuid.uuid4()
    fs = frames(two_in_target("metformin", "atorvastatin"))
    assert (
        eng.evaluate(IDENTITY_PROFILE, fs, result_id=rid).spot_checks
        == eng.evaluate(IDENTITY_PROFILE, fs, result_id=rid).spot_checks
    )


def test_wrong_medication_with_right_count_is_mismatch():
    v = evaluate(
        two_in_target("metformin", "metformin"), IDENTITY_PROFILE, engine(identity_model({METFORMIN, ATORVASTATIN}))
    ).verdict(TARGET)
    assert v.status == CompartmentStatus.MISMATCH
    assert v.findings == [
        finding(FindingKind.WRONG_QUANTITY, medication_id=ATORVASTATIN, expected=1, observed=0),
        finding(FindingKind.WRONG_QUANTITY, medication_id=METFORMIN, expected=1, observed=2),
    ]


def test_medication_outside_capability_is_generic_pill():
    v = evaluate(
        two_in_target("metformin", "atorvastatin"), IDENTITY_PROFILE, engine(identity_model({METFORMIN}))
    ).verdict(TARGET)
    assert v.status == CompartmentStatus.COUNT_MATCHED and v.observed_medications == {METFORMIN: 1}


# --------------------------------------------------------------------------- brain


SINGLE = profile([], {TARGET: ONE})


def brain_engine(trusted=frozenset(), well_known=frozenset(), model="count"):
    return engine(
        model, brain=BrainContext(trusted_medications=frozenset(trusted), well_known_medications=frozenset(well_known))
    )


def pill_in_target(decision):
    return [with_identity(d, decision) for d in detections("pill", 1, TARGET)]


def test_trusted_brain_identity_verifies_with_count_only_model():
    result = evaluate(pill_in_target(IdentityDecision.identified(METFORMIN)), SINGLE, brain_engine({METFORMIN}))
    assert result.verdict(TARGET).status == CompartmentStatus.VERIFIED and result.trusted_medications == [METFORMIN]


def test_untrusted_brain_identity_never_verifies_but_escalates():
    v = evaluate(pill_in_target(IdentityDecision.identified(METFORMIN)), SINGLE, brain_engine()).verdict(TARGET)
    assert v.status == CompartmentStatus.COUNT_MATCHED and v.observed_medications == {}
    v = evaluate(pill_in_target(IdentityDecision.identified(ASPIRIN)), SINGLE, brain_engine()).verdict(TARGET)
    assert (
        v.status == CompartmentStatus.NEEDS_REVIEW
        and finding(FindingKind.SUSPECTED_MEDICATION, medication_id=ASPIRIN) in v.findings
    )


def test_trusted_brain_catches_wrong_medication():
    v = evaluate(
        pill_in_target(IdentityDecision.identified(ASPIRIN)), SINGLE, brain_engine({METFORMIN, ASPIRIN})
    ).verdict(TARGET)
    assert v.status == CompartmentStatus.MISMATCH
    assert finding(FindingKind.UNEXPECTED_MEDICATION, medication_id=ASPIRIN, observed=1) in v.findings


def test_unrecognised_pill_flagged_only_when_expected_medications_well_known():
    v = evaluate(pill_in_target(UNRECOGNISED), SINGLE, brain_engine(well_known={METFORMIN})).verdict(TARGET)
    assert finding(FindingKind.UNRECOGNISED_PILL, count=1) in v.findings
    assert (
        evaluate(pill_in_target(UNRECOGNISED), SINGLE, brain_engine()).verdict(TARGET).status
        == CompartmentStatus.COUNT_MATCHED
    )


def test_model_and_brain_disagreement_goes_to_review():
    eng = engine(identity_model({METFORMIN, ASPIRIN}), brain=BrainContext(trusted_medications=frozenset({ASPIRIN})))
    dets = [with_identity(d, IdentityDecision.identified(ASPIRIN)) for d in detections("metformin", 1, TARGET)]
    v = evaluate(dets, SINGLE, eng).verdict(TARGET)
    assert finding(FindingKind.CONFLICTING_IDENTITY, count=1) in v.findings and v.status != CompartmentStatus.VERIFIED


def test_partial_trust_leaves_compartment_unverified():
    dets = [
        with_identity(d, IdentityDecision.identified(m))
        for d, m in zip(two_in_target("pill", "pill"), [METFORMIN, ATORVASTATIN], strict=True)
    ]
    v = evaluate(dets, IDENTITY_PROFILE, brain_engine({METFORMIN})).verdict(TARGET)
    assert v.status == CompartmentStatus.COUNT_MATCHED and v.observed_medications == {METFORMIN: 1}


def test_pill_compartments_lists_confident_doses_clearly_inside():
    r = LAYOUT.cell_rect(TARGET)
    dets = [
        detection("pill", r.center),
        detection("pill", Point(x=r.center.x, y=r.center.y + 0.01), 0.3),
        detection("foreign", r.center),
        detection("pill", Point(x=r.max_x - 0.005, y=r.center.y)),
    ]
    assert pill_compartments(frame(dets), LAYOUT, 0.6, engine().model.meaning) == {0: TARGET}
