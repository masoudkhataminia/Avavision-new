from __future__ import annotations

import math

import numpy as np
import pytest

from avavision.core.audit import GENESIS_HASH, CheckRecord, make_entry, verify_chain
from avavision.core.engine import CompartmentStatus
from avavision.core.evaluation import EvaluationReport, EvaluationSample, MedicationMetrics, evaluate
from avavision.core.gate import (
    COUNT_ONLY,
    UNAVAILABLE,
    BlockerKind,
    CapabilityKind,
    ModelCapability,
    ModelStage,
    builtin_pocket_segmenter,
    evaluate_gate,
)
from avavision.core.geometry import Homography, Point, Quad, Rect
from avavision.core.models import (
    WEEKLY_7X4,
    CellLocationKind,
    ExpectedItem,
    LayoutError,
    PackProfile,
    ProfileIssueKind,
    cell,
)
from avavision.core.observation import PILL, LabelMeaning, MeaningKind, RegistrationIssue, register
from avavision.core.session import CheckSession, Phase, SessionError
from avavision.core.signoff import (
    CompartmentReview,
    ReviewOutcome,
    SignOff,
    SignOffDecision,
    SignOffError,
    SignOffErrorKind,
)
from conftest import (
    ASPIRIN,
    ATORVASTATIN,
    CATALOG,
    FIXED,
    IDENTITY_LABELS,
    LAYOUT,
    METFORMIN,
    ONE,
    QUAD,
    engine,
    frames,
    full_pack,
    manifest,
    profile,
)

SQUARE = [Point(x=0, y=0), Point(x=1, y=0), Point(x=1, y=1), Point(x=0, y=1)]


def quad(x0, y0, x1, y1) -> Quad:
    return Quad.from_corners([Point(x=x0, y=y0), Point(x=x1, y=y0), Point(x=x1, y=y1), Point(x=x0, y=y1)])


# --------------------------------------------------------------------------- geometry and layout


def test_homography_maps_and_inverts():
    skewed = [Point(x=0.12, y=0.2), Point(x=0.85, y=0.1), Point(x=0.95, y=0.9), Point(x=0.05, y=0.8)]
    h = Homography.mapping(skewed, SQUARE)
    for s, t in zip(skewed, SQUARE, strict=True):
        m = h.apply(s)
        assert m.x == pytest.approx(t.x, abs=1e-9) and m.y == pytest.approx(t.y, abs=1e-9)
    p = Point(x=0.37, y=0.61)
    back = h.inverse.apply(h.apply(p))
    assert back.x == pytest.approx(p.x) and back.y == pytest.approx(p.y)
    many = h.apply_many(np.array([[s.x, s.y] for s in skewed]))
    assert np.allclose(many, [[0, 0], [1, 0], [1, 1], [0, 1]], atol=1e-9)
    collinear = [Point(x=0, y=0), Point(x=0.5, y=0.5), Point(x=1, y=1), Point(x=0.2, y=0.2)]
    assert Homography.mapping(collinear, SQUARE) is None


def test_quad_properties_and_rotation():
    q = quad(0, 0, 2, 1)
    assert q.area == pytest.approx(2) and q.is_convex and all(a == pytest.approx(90) for a in q.interior_angles)
    bowtie = Quad.from_corners([Point(x=0, y=0), Point(x=1, y=1), Point(x=1, y=0), Point(x=0, y=1)])
    assert not bowtie.is_convex
    turned = QUAD.rotated(1)
    assert turned.top_left == QUAD.bottom_left and turned.top_right == QUAD.top_left
    assert QUAD.rotated(4) == QUAD and QUAD.rotated(-1) == QUAD.rotated(3)


def test_layout_locate_and_validation():
    assert (
        WEEKLY_7X4.validation_errors() == [] and len(WEEKLY_7X4.all_compartments) == 28 and not WEEKLY_7X4.is_calibrated
    )
    assert WEEKLY_7X4.label(cell(3, 6)) == "Day 7 · Bedtime"
    assert LAYOUT.locate(LAYOUT.cell_rect(cell(1, 1)).center).compartments == (cell(1, 1),)
    border = LAYOUT.locate(Point(x=1 / 3 + 0.01, y=0.25))
    assert border.kind == CellLocationKind.BORDER and border.compartments == (cell(0, 0), cell(0, 1))
    corner = LAYOUT.locate(Point(x=2 / 3 - 0.01, y=0.51))
    assert corner.compartments == (cell(0, 1), cell(0, 2), cell(1, 1), cell(1, 2))
    assert LAYOUT.locate(Point(x=0.01, y=0.25)).compartments == (cell(0, 0),)
    assert LAYOUT.locate(Point(x=1.2, y=0.5)).kind == CellLocationKind.OUTSIDE
    bad = LAYOUT.model_copy(
        update={
            "rows": 0,
            "row_labels": ["x"],
            "grid_region": Rect(x=0.5, y=0, width=0.7, height=1),
            "border_band": 0.5,
        }
    )
    assert bad.validation_errors() == [
        LayoutError.INVALID_GRID_SIZE,
        LayoutError.LABEL_COUNT_MISMATCH,
        LayoutError.GRID_REGION_OUTSIDE_PACK,
        LayoutError.INVALID_BORDER_BAND,
    ]


def test_profile_issues():
    p = profile(ONE)
    assert p.issues(LAYOUT, CATALOG) == []
    p.reference = " "
    p.compartments = [c for c in p.compartments if c.compartment != cell(1, 2)]
    p.set_items(cell(5, 5), [])
    p.compartments.append(
        p.expectation(cell(0, 0)).model_copy(update={"items": [ExpectedItem(medication_id="unknown", quantity=0)]})
    )
    assert [i.kind for i in p.issues(LAYOUT, CATALOG)] == [
        ProfileIssueKind.EMPTY_REFERENCE,
        ProfileIssueKind.COMPARTMENT_OUTSIDE_LAYOUT,
        ProfileIssueKind.DUPLICATE_COMPARTMENT,
        ProfileIssueKind.INVALID_QUANTITY,
        ProfileIssueKind.UNKNOWN_MEDICATION,
        ProfileIssueKind.UNSPECIFIED_COMPARTMENT,
    ]
    empty = PackProfile.empty("P-1", WEEKLY_7X4)
    assert empty.issues(WEEKLY_7X4, CATALOG) == [] and empty.expectation(cell(2, 3)).total_quantity == 0


# --------------------------------------------------------------------------- registration


def reg(q, confidence=0.95, layout=LAYOUT):
    return register(q, confidence, 1500, 1000, layout)


def test_registration_accepts_and_rejects():
    centre = reg(QUAD).registration.image_to_pack.apply(Point(x=0.5, y=0.5))
    assert centre.x == pytest.approx(0.5) and centre.y == pytest.approx(0.5)
    assert reg(None).issues == [RegistrationIssue.PACK_NOT_FOUND]
    assert reg(QUAD, 0.2).issues == [RegistrationIssue.LOW_DETECTOR_CONFIDENCE]
    assert reg(quad(0.4, 0.4, 0.6, 0.6)).issues == [RegistrationIssue.PACK_TOO_SMALL]
    assert reg(quad(0.0, 0.1, 0.8, 0.9)).issues == [RegistrationIssue.PACK_TOUCHES_FRAME_EDGE]
    steep = Quad.from_corners([Point(x=0.3, y=0.1), Point(x=0.7, y=0.1), Point(x=0.95, y=0.9), Point(x=0.05, y=0.9)])
    assert RegistrationIssue.PERSPECTIVE_TOO_STEEP in reg(steep).issues
    portrait = quad(0.3, 0.05, 0.7, 0.95)
    assert reg(portrait).issues == [RegistrationIssue.ASPECT_RATIO_MISMATCH]
    assert reg(portrait, layout=LAYOUT.model_copy(update={"is_calibrated": False})).registration is not None


# --------------------------------------------------------------------------- gate and evaluation


def report(**over):
    base = dict(
        dataset_id="holdout-v1",
        total_samples=1200,
        error_samples=700,
        count_accuracy=0.99,
        identity_precision=0.99,
        identity_recall=0.99,
        false_acceptance_rate=0.001,
        count_false_acceptance_rate=0.001,
        review_rate=0.1,
        per_medication={
            METFORMIN: MedicationMetrics(samples=80, precision=0.99, recall=0.98),
            ATORVASTATIN: MedicationMetrics(samples=20, precision=1, recall=1),
            ASPIRIN: MedicationMetrics(samples=90, precision=0.9, recall=0.99),
        },
    )
    return EvaluationReport(**(base | over))


def test_gate_integrity_and_stages():
    m = manifest()
    assert evaluate_gate(m, None).blockers[0].kind == BlockerKind.MODEL_FILE_MISSING
    assert evaluate_gate(m, "different").capability == UNAVAILABLE
    assert evaluate_gate(m, "ABC123").capability == COUNT_ONLY
    assert (
        evaluate_gate(m.model_copy(update={"stage": ModelStage.REVOKED}), "abc123").blockers[0].kind
        == BlockerKind.MODEL_REVOKED
    )
    assert evaluate_gate(m.model_copy(update={"schema_version": 99}), "abc123").capability == UNAVAILABLE
    assert (
        evaluate_gate(manifest(labels={"foreign": "foreign"}), "abc123").blockers[0].kind == BlockerKind.NO_PILL_LABELS
    )
    assert [b.kind for b in evaluate_gate(m, "abc123").blockers] == [
        BlockerKind.STAGE_NOT_RELEASED,
        BlockerKind.NO_MEDICATION_LABELS,
        BlockerKind.EVALUATION_MISSING,
    ]


def test_released_model_identifies_only_medications_that_pass():
    decision = evaluate_gate(manifest(ModelStage.RELEASED, IDENTITY_LABELS, report()), "abc123")
    assert decision.capability == ModelCapability(kind=CapabilityKind.IDENTITY, medications=frozenset({METFORMIN}))
    assert [(b.kind, b.medication_id) for b in decision.blockers] == [
        (BlockerKind.MEDICATION_BELOW_THRESHOLD, ASPIRIN),
        (BlockerKind.MEDICATION_UNDERSAMPLED, ATORVASTATIN),
    ]
    weak = report(
        total_samples=900,
        error_samples=100,
        count_accuracy=0.9,
        identity_precision=0.9,
        identity_recall=0.8,
        false_acceptance_rate=0.02,
    )
    decision = evaluate_gate(manifest(ModelStage.RELEASED, IDENTITY_LABELS, weak), "abc123")
    assert decision.capability == COUNT_ONLY and len(decision.blockers) == 6


def test_capability_resolution_and_label_meaning():
    capability = ModelCapability(kind=CapabilityKind.IDENTITY, medications=frozenset({METFORMIN}))
    assert capability.resolve(LabelMeaning(MeaningKind.MEDICATION, ASPIRIN)) == PILL
    assert LabelMeaning.parse("medication:x").medication_id == "x"
    assert LabelMeaning.parse("medication:") is None and LabelMeaning.parse("medication") is None
    assert str(LabelMeaning.parse("medication:metformin-500")) == "medication:metformin-500"
    assert manifest().meaning("never-seen") == PILL
    assert builtin_pocket_segmenter().capability == COUNT_ONLY


def test_evaluator_metrics():
    m, a = METFORMIN, ASPIRIN
    samples = [
        EvaluationSample(
            sample_id="1",
            expected={m: 1},
            truth={m: 1},
            predicted_status="verified",
            predicted_count=1,
            predicted_medications={m: 1},
        ),
        EvaluationSample(
            sample_id="2",
            expected={m: 1},
            truth={a: 1},
            predicted_status="verified",
            predicted_count=1,
            predicted_medications={m: 1},
        ),
        EvaluationSample(
            sample_id="3",
            expected={m: 2},
            truth={m: 1},
            predicted_status="mismatch",
            predicted_count=1,
            predicted_medications={m: 1},
        ),
        EvaluationSample(
            sample_id="4",
            expected={m: 1},
            truth={m: 1},
            truth_has_broken_or_foreign=True,
            predicted_status="needsReview",
            predicted_count=None,
        ),
    ]
    r = evaluate(samples, "unit")
    assert (r.total_samples, r.error_samples, r.count_accuracy) == (4, 3, 1)
    assert (
        r.false_acceptance_rate == pytest.approx(1 / 3) and r.count_false_acceptance_rate == 0 and r.review_rate == 0.25
    )
    assert r.identity_precision == pytest.approx(2 / 3) and r.identity_recall == pytest.approx(0.5)
    assert evaluate([], "empty").false_acceptance_rate == 1


# --------------------------------------------------------------------------- sign-off, session, audit


def count_matched_result():
    return engine().evaluate(profile(ONE), frames(full_pack()), evaluated_at=FIXED)


def sign_off(decision, reviews, pharmacist="MK", acknowledged=False):
    return SignOff(
        pharmacist=pharmacist,
        decision=decision,
        reviews=reviews,
        acknowledged_pack_findings=acknowledged,
        signed_at=FIXED,
    )


def all_reviewed(outcome=ReviewOutcome.CONFIRMED_CORRECT):
    return [CompartmentReview(compartment=i, outcome=outcome) for i in LAYOUT.all_compartments]


def expect_error(kind, call):
    with pytest.raises(SignOffError) as error:
        call()
    assert error.value.kind == kind
    return error.value


def test_sign_off_rules():
    from avavision.core.signoff import validate_sign_off

    result = count_matched_result()
    e = expect_error(
        SignOffErrorKind.UNREVIEWED_COMPARTMENTS,
        lambda: validate_sign_off(sign_off(SignOffDecision.RELEASED, all_reviewed()[:-1]), result),
    )
    assert e.compartments == [cell(1, 2)]
    validate_sign_off(sign_off(SignOffDecision.RELEASED, all_reviewed()), result)
    reviews = all_reviewed()
    reviews[0] = reviews[0].model_copy(update={"outcome": ReviewOutcome.UNRESOLVED})
    expect_error(
        SignOffErrorKind.UNRESOLVED_COMPARTMENTS,
        lambda: validate_sign_off(sign_off(SignOffDecision.RELEASED, reviews), result),
    )
    validate_sign_off(sign_off(SignOffDecision.WITHHELD, reviews), result)
    expect_error(
        SignOffErrorKind.MISSING_PHARMACIST,
        lambda: validate_sign_off(sign_off(SignOffDecision.WITHHELD, [], " "), result),
    )
    stray = [CompartmentReview(compartment=cell(9, 9), outcome=ReviewOutcome.CONFIRMED_CORRECT)]
    expect_error(
        SignOffErrorKind.REVIEW_FOR_UNKNOWN_COMPARTMENT,
        lambda: validate_sign_off(sign_off(SignOffDecision.WITHHELD, stray), result),
    )
    from avavision.core.engine import PackFinding, PackFindingKind

    result.pack_findings = [PackFinding(kind=PackFindingKind.OBJECTS_OUTSIDE_COMPARTMENTS, count=1)]
    expect_error(
        SignOffErrorKind.PACK_FINDINGS_NOT_ACKNOWLEDGED,
        lambda: validate_sign_off(sign_off(SignOffDecision.RELEASED, all_reviewed()), result),
    )
    validate_sign_off(sign_off(SignOffDecision.RELEASED, all_reviewed(), acknowledged=True), result)


def test_session_lifecycle_and_audit_chain():
    session = CheckSession(LAYOUT, profile(ONE), CATALOG)
    for f in frames(full_pack()):
        session.record(f.model_copy(update={"image_sha256": "img"}))
    result = session.analyze(engine())
    assert session.phase == Phase.ANALYZED and all(
        v.status == CompartmentStatus.COUNT_MATCHED for v in result.compartments
    )
    with pytest.raises(SessionError):
        session.record(frames([])[0])
    with pytest.raises(SignOffError):
        session.complete(sign_off(SignOffDecision.RELEASED, []), "0.2", "station", None)
    assert session.phase == Phase.ANALYZED
    record = session.complete(sign_off(SignOffDecision.RELEASED, all_reviewed()), "0.2", "station", None)
    assert session.phase == Phase.COMPLETED and record.frame_image_sha256s == ["img"] * 3

    entries, previous = [], GENESIS_HASH
    for pharmacist in ("A", "B", "C"):
        r = record.model_copy(update={"sign_off": record.sign_off.model_copy(update={"pharmacist": pharmacist})})
        entry = make_entry(r, len(entries), previous)
        entries.append(entry)
        previous = entry.hash
    assert verify_chain(entries) is None
    assert CheckRecord.model_validate_json(entries[1].payload).sign_off.pharmacist == "B"
    tampered = entries[1].model_copy(update={"payload": entries[1].payload.replace('"B"', '"Z"')})
    assert verify_chain([entries[0], tampered, entries[2]]).defect == "hashMismatch"
    assert verify_chain([entries[0], entries[2]]).defect == "sequenceGap"


def test_invalid_profile_cannot_start_session_and_frames_are_bounded():
    p = profile(ONE)
    p.compartments.pop()
    with pytest.raises(SessionError):
        CheckSession(LAYOUT, p, CATALOG)
    session = CheckSession(LAYOUT, profile(ONE), CATALOG)
    for f in frames([], count=20):
        session.record(f)
    assert len(session.frames) == CheckSession.MAXIMUM_RETAINED_FRAMES
    session.analyze(engine())
    session.retake()
    assert session.phase == Phase.CAPTURING and not session.frames


def test_json_round_trip_of_result():
    result = count_matched_result()
    from avavision.core.engine import VerificationResult

    assert VerificationResult.model_validate_json(result.model_dump_json()) == result
    assert not math.isnan(result.usable_frame_count)
