from __future__ import annotations

import numpy as np
import pytest

from avavision.brain.brain import Brain
from avavision.brain.calibration import CalibrationOutcome, IdentityCalibrator, wilson_lower_bound
from avavision.brain.identity import (
    AMBIGUOUS,
    INSUFFICIENT,
    UNRECOGNISED,
    DecisionKind,
    IdentityClassifier,
    IdentityDecision,
    IdentityEvidence,
    IdentityPolicy,
)
from avavision.brain.knowledge import AddOutcome, Exemplar, ExemplarSource, KnowledgeBase, KnowledgePolicy
from avavision.brain.learning import LabellingError, LearningPlan, Observation, PillSighting, plan_learning
from avavision.brain.trust import TrustLedger, TrustPolicy, TrustState
from avavision.core.models import ExpectedItem, cell
from avavision.core.signoff import CompartmentReview, ReviewOutcome, SignOff, SignOffDecision
from conftest import ASPIRIN, LAYOUT, METFORMIN, SyntheticPills, profile

MEDS = ["med-a", "med-b", "med-c", "med-d"]


def knowledge(s: SyntheticPills, meds, groups=3, per_group=8) -> KnowledgeBase:
    kb = KnowledgeBase(s.embedder)
    for m in meds:
        for g in range(groups):
            for _ in range(per_group):
                kb.add(
                    Exemplar(
                        medication_id=m,
                        vector=s.pill(m),
                        embedder_id=s.embedder,
                        source=ExemplarSource.TEACHING,
                        group_id=f"{m}-{g}",
                    )
                )
    return kb


def test_knowledge_duplicates_capacity_and_forgetting(synthetic):
    kb = KnowledgeBase(
        synthetic.embedder, KnowledgePolicy(maximum_exemplars_per_medication=5, duplicate_similarity=0.999)
    )
    first = synthetic.pill(METFORMIN)

    def ex(v, emb=synthetic.embedder):
        return Exemplar(
            medication_id=METFORMIN, vector=v, embedder_id=emb, source=ExemplarSource.TEACHING, group_id="g"
        )

    assert kb.add(ex(first))[0] == AddOutcome.ADDED
    assert kb.add(ex(first))[0] == AddOutcome.DUPLICATE
    assert kb.add(ex(first, "other"))[0] == AddOutcome.INCOMPATIBLE
    for _ in range(10):
        kb.add(ex(synthetic.pill(METFORMIN)))
    assert kb.count(METFORMIN) == 5
    kb.forget(METFORMIN)
    assert kb.medications == set()


def test_uncalibrated_brain_never_names(synthetic):
    kb = knowledge(synthetic, MEDS)
    evidence = IdentityClassifier(kb.index(), IdentityPolicy()).classify(synthetic.pill("med-a"))
    assert (
        evidence.decision == INSUFFICIENT
        and evidence.candidates[0].medication_id == "med-a"
        and len(evidence.candidates) == 3
    )


def test_calibration_learns_thresholds_that_identify_and_reject_unknown(synthetic):
    kb = knowledge(synthetic, MEDS)
    report, policy = IdentityCalibrator().calibrate(kb, IdentityPolicy())
    assert policy is not None, report
    assert (
        policy.version == 1
        and report.precision_lower_bound >= 0.9
        and report.false_identifications == 0
        and report.coverage > 0.9
    )
    classifier = IdentityClassifier(kb.index(), policy)
    named = 0
    for m in MEDS:
        for _ in range(25):
            decision = classifier.classify(synthetic.pill(m)).decision
            if decision.named:
                assert decision.named == m
                named += 1
    assert named > 90
    assert all(classifier.classify(synthetic.unknown()).decision.named is None for _ in range(50))


def test_indistinguishable_medications_are_never_named(synthetic):
    synthetic.prototypes["twin-b"] = synthetic.prototype("twin-a")
    kb = knowledge(synthetic, ["twin-a", "twin-b"], groups=4)
    report, policy = IdentityCalibrator().calibrate(kb, IdentityPolicy())
    if policy is None:
        assert report.outcome == CalibrationOutcome.TARGET_UNREACHABLE
    else:
        classifier = IdentityClassifier(kb.index(), policy)
        assert sum(classifier.classify(synthetic.pill("twin-a")).decision.named is not None for _ in range(40)) < 4


def test_calibration_needs_enough_knowledge(synthetic):
    report, policy = IdentityCalibrator().calibrate(knowledge(synthetic, ["only"]), IdentityPolicy())
    assert policy is None and report.outcome == CalibrationOutcome.INSUFFICIENT_DATA and report.medications == 1


def test_poorly_known_medication_is_not_named_or_overruled_and_margin(synthetic):
    kb = knowledge(synthetic, ["well-known"])
    kb.add(
        Exemplar(
            medication_id="new",
            vector=synthetic.pill("new"),
            embedder_id=synthetic.embedder,
            source=ExemplarSource.TEACHING,
            group_id="x",
        )
    )
    classifier = IdentityClassifier(kb.index(), IdentityPolicy(accept_similarity=0.5))
    assert classifier.classify(synthetic.pill("new")).decision == INSUFFICIENT
    assert classifier.classify(synthetic.pill("well-known")).decision == IdentityDecision.identified("well-known")
    ambiguous = IdentityClassifier(
        knowledge(synthetic, ["a", "b"]).index(), IdentityPolicy(accept_similarity=-1, minimum_margin=3)
    )
    assert ambiguous.classify(synthetic.pill("a")).decision == AMBIGUOUS


def test_wilson_lower_bound():
    assert wilson_lower_bound(0, 0) == 0
    assert wilson_lower_bound(81, 100) == pytest.approx(0.7222, abs=1e-3)
    assert wilson_lower_bound(100, 100) == pytest.approx(0.9630, abs=1e-3)


def test_trust_is_earned_and_lost_immediately():
    policy = TrustPolicy(minimum_predictions=20, minimum_precision_lower_bound=0.8, minimum_streak=10)
    ledger = TrustLedger(embedder_id="e")
    for _ in range(19):
        ledger.observe(IdentityDecision.identified("m"), "m")
    status = ledger.status("m", policy)
    assert status.state == TrustState.LEARNING and status.progress == pytest.approx(0.95)
    ledger.observe(IdentityDecision.identified("m"), "m")
    assert ledger.trusted(policy) == {"m"}
    ledger.observe(IdentityDecision.identified("m"), "other")
    assert ledger.status("m", policy).state == TrustState.SUSPENDED and ledger.record("other").confusions == 1
    for _ in range(10):
        ledger.observe(IdentityDecision.identified("m"), "m")
    assert ledger.status("m", policy).state == TrustState.TRUSTED
    ledger.reset_streaks()
    assert ledger.status("m", policy).state != TrustState.TRUSTED
    for decision in (AMBIGUOUS, UNRECOGNISED, INSUFFICIENT):
        ledger.observe(decision, "n")
    assert ledger.record("n").abstentions == 3 and ledger.record("n").named_count == 0


def sightings(s, index, count, medication, identified=True):
    decision = IdentityDecision.identified(medication) if identified else UNRECOGNISED
    return [
        PillSighting(
            compartment=index,
            vector=s.pill(medication),
            embedder_id=s.embedder,
            identity=IdentityEvidence(embedder_id=s.embedder, policy_version=1, decision=decision),
            crop_file="crop.jpg",
        )
        for _ in range(count)
    ]


def released(reviews):
    return SignOff(pharmacist="MK", decision=SignOffDecision.RELEASED, reviews=reviews, acknowledged_pack_findings=True)


def test_learns_only_from_inspected_confirmed_compartments(synthetic):
    single, corrected, miscounted, mixed = cell(0, 0), cell(0, 1), cell(0, 2), cell(1, 0)
    p = profile(
        [ExpectedItem(medication_id=METFORMIN, quantity=2)],
        {mixed: [ExpectedItem(medication_id=METFORMIN, quantity=1), ExpectedItem(medication_id=ASPIRIN, quantity=1)]},
    )
    seen = (
        sightings(synthetic, single, 2, METFORMIN)
        + sightings(synthetic, corrected, 2, METFORMIN)
        + sightings(synthetic, miscounted, 1, METFORMIN)
        + sightings(synthetic, mixed, 1, METFORMIN, identified=False)
        + sightings(synthetic, mixed, 1, ASPIRIN)
        + sightings(synthetic, cell(1, 1), 2, METFORMIN)
    )
    reviews = [
        CompartmentReview(compartment=c, outcome=o)
        for c, o in [
            (single, ReviewOutcome.CONFIRMED_CORRECT),
            (corrected, ReviewOutcome.CORRECTED),
            (miscounted, ReviewOutcome.CONFIRMED_CORRECT),
            (mixed, ReviewOutcome.CONFIRMED_CORRECT),
        ]
    ]
    plan = plan_learning("check-1", released(reviews), p, LAYOUT, seen)
    assert len(plan.exemplars) == 2 and all(
        e.medication_id == METFORMIN and e.group_id == "check-1" for e in plan.exemplars
    )
    assert len(plan.observations) == 2 and len(plan.tasks) == 1
    task = plan.tasks[0]
    assert task.compartment_label == "D1 · PM" and task.expected == {METFORMIN: 1, ASPIRIN: 1} and task.priority == 0.5
    ids = [s.id for s in task.sightings]
    with pytest.raises(LabellingError):
        task.resolve({ids[0]: METFORMIN})
    with pytest.raises(LabellingError):
        task.resolve({ids[0]: METFORMIN, ids[1]: METFORMIN})
    resolved = task.resolve({ids[0]: METFORMIN, ids[1]: ASPIRIN})
    assert [e.medication_id for e in resolved.exemplars] == [METFORMIN, ASPIRIN]
    assert {e.source for e in resolved.exemplars} == {ExemplarSource.LABELLED}


def test_brain_teaches_calibrates_and_earns_trust(synthetic):
    brain = Brain(
        synthetic.embedder, TrustPolicy(minimum_predictions=30, minimum_precision_lower_bound=0.85, minimum_streak=20)
    )
    assert brain.well_known_medications == set()
    meds = ["med-a", "med-b", "med-c"]
    for m in meds:
        for g in range(3):
            brain.teach(m, [(synthetic.pill(m), None) for _ in range(8)], group_id=f"{m}-{g}")
    assert brain.identity_policy.is_calibrated, brain.last_calibration
    assert brain.well_known_medications == set(meds) and brain.trusted_medications == set()
    for _ in range(40):
        decision = brain.classifier().classify(synthetic.pill("med-a")).decision
        brain.learn(LearningPlan(observations=[Observation(decision, "med-a")]))
    assert brain.trust_status("med-a").state == TrustState.TRUSTED and brain.summary.trusted_medications == ["med-a"]

    p = profile([ExpectedItem(medication_id="med-b", quantity=1), ExpectedItem(medication_id="med-c", quantity=1)])
    seen = sightings(synthetic, cell(0, 0), 1, "med-b") + sightings(synthetic, cell(0, 0), 1, "med-c")
    brain.learn(
        plan_learning(
            "c",
            released([CompartmentReview(compartment=cell(0, 0), outcome=ReviewOutcome.CONFIRMED_CORRECT)]),
            p,
            LAYOUT,
            seen,
        )
    )
    task = brain.labelling_queue[0]
    before = len(brain.knowledge.exemplars)
    brain.resolve_task(task.id, {task.sightings[0].id: "med-b", task.sightings[1].id: "med-c"})
    assert brain.labelling_queue == [] and len(brain.knowledge.exemplars) == before + 2
    with pytest.raises(KeyError):
        brain.resolve_task(task.id, {})
    brain.forget("med-a")
    assert "med-a" not in brain.knowledge.medications and brain.trusted_medications == set()


def test_migration_keeps_reembedded_pills_and_restarts_trust(synthetic):
    brain = Brain(synthetic.embedder)
    brain.teach("med-a", [(synthetic.pill("med-a"), "a.jpg") for _ in range(3)])
    vectors = {e.id: e.vector * 2 / np.linalg.norm(e.vector * 2) for e in brain.knowledge.exemplars[:2]}
    migrated = brain.migrated("v2", vectors)
    assert migrated.embedder_id == "v2" and len(migrated.knowledge.exemplars) == 2
    assert migrated.identity_policy.version == 0 and {e.crop_file for e in migrated.knowledge.exemplars} == {"a.jpg"}
    assert DecisionKind.IDENTIFIED.value == "identified"
