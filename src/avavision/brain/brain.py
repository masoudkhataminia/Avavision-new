"""AvaVision's learning brain.

Safety contract (asymmetric trust):
- From day one the brain may only *raise* concerns (a pill that looks like an unexpected medication, or
  like nothing it knows) — this can never make a result less safe.
- It may *grant* acceptance for a medication only once its track record, built exclusively from pills a
  pharmacist inspected, passes the trust policy; one wrong identification suspends that trust.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
from pydantic import BaseModel

from ..core.physical import PhysicalFeatures, PhysicalPolicy, PhysicalRange
from .calibration import CalibrationReport, IdentityCalibrator
from .identity import EmbedderID, IdentityClassifier, IdentityPolicy, MedicationID
from .knowledge import AddOutcome, Exemplar, ExemplarSource, KnowledgeBase
from .learning import LabellingTask, LearningPlan
from .physical import PhysicalMemory, PhysicalSample
from .trust import TrustLedger, TrustPolicy, TrustStatus


class BrainSummary(BaseModel):
    """Brain state captured in each audit record."""

    embedder_id: str
    policy_version: int
    exemplar_count: int
    known_medications: int
    trusted_medications: list[MedicationID]


@dataclass
class LearningSummary:
    exemplars_added: int = 0
    duplicates_skipped: int = 0
    observations_recorded: int = 0
    tasks_queued: int = 0
    measurements_added: int = 0
    recalibration: CalibrationReport | None = None


class Brain:
    def __init__(self, embedder_id: EmbedderID, trust_policy: TrustPolicy | None = None):
        self.knowledge = KnowledgeBase(embedder_id)
        self.identity_policy = IdentityPolicy()
        self.trust_policy = trust_policy or TrustPolicy()
        self.ledger = TrustLedger(embedder_id=embedder_id)
        self.labelling_queue: list[LabellingTask] = []
        self.physical = PhysicalMemory()
        self.physical_policy = PhysicalPolicy()
        self.last_calibration: CalibrationReport | None = None
        self.exemplars_at_calibration = 0

    @property
    def embedder_id(self) -> EmbedderID:
        return self.knowledge.embedder_id

    def classifier(self) -> IdentityClassifier:
        return IdentityClassifier(self.knowledge.index(), self.identity_policy)

    @property
    def trusted_medications(self) -> set[MedicationID]:
        return self.ledger.trusted(self.trust_policy)

    @property
    def well_known_medications(self) -> set[MedicationID]:
        """Medications known well enough to notice a pill that is none of them."""
        if not self.identity_policy.is_calibrated:
            return set()
        p = self.identity_policy
        return {
            m
            for m in self.knowledge.medications
            if self.knowledge.count(m) >= p.minimum_exemplars and self.knowledge.group_count(m) >= p.minimum_groups
        }

    def trust_status(self, medication: MedicationID) -> TrustStatus:
        return self.ledger.status(medication, self.trust_policy)

    # ------------------------------------------------------------------ learning

    def learn(self, plan: LearningPlan, at: datetime | None = None) -> LearningSummary:
        at = at or datetime.now(UTC)
        summary = LearningSummary()
        for exemplar in plan.exemplars:
            outcome, _ = self.knowledge.add(exemplar)
            if outcome in (AddOutcome.ADDED, AddOutcome.ADDED_REPLACING):
                summary.exemplars_added += 1
            else:
                summary.duplicates_skipped += 1
        for observation in plan.observations:
            self.ledger.observe(observation.decision, observation.truth, at)
            summary.observations_recorded += 1
        for sample in plan.measurements:
            self.physical.add(sample)
        summary.measurements_added = len(plan.measurements)
        self.labelling_queue.extend(plan.tasks)
        self.labelling_queue.sort(key=lambda t: (-t.priority, -t.created_at.timestamp()))
        summary.tasks_queued = len(plan.tasks)
        if self.needs_recalibration:
            summary.recalibration = self.recalibrate(at=at)
        return summary

    def teach(
        self,
        medication: MedicationID,
        vectors: list[tuple[np.ndarray, str | None]],
        group_id: str | None = None,
        at: datetime | None = None,
        measurements: list[PhysicalFeatures] | None = None,
    ) -> LearningSummary:
        """Teaches a medication from photos of pills the pharmacist knows to be that medication."""
        at = at or datetime.now(UTC)
        group = group_id or f"teach-{at.timestamp()}"
        plan = LearningPlan(
            measurements=[PhysicalSample(medication, f, group, created_at=at) for f in measurements or []],
            exemplars=[
                Exemplar(
                    medication_id=medication,
                    vector=v,
                    embedder_id=self.embedder_id,
                    source=ExemplarSource.TEACHING,
                    group_id=group,
                    created_at=at,
                    crop_file=crop,
                )
                for v, crop in vectors
            ],
        )
        return self.learn(plan, at)

    def resolve_task(
        self, task_id: str, labels: dict[str, MedicationID], at: datetime | None = None
    ) -> LearningSummary:
        task = next((t for t in self.labelling_queue if t.id == task_id), None)
        if task is None:
            raise KeyError(task_id)
        plan = task.resolve(labels, at)
        self.labelling_queue = [t for t in self.labelling_queue if t.id != task_id]
        return self.learn(plan, at)

    def discard_task(self, task_id: str) -> None:
        self.labelling_queue = [t for t in self.labelling_queue if t.id != task_id]

    def forget(self, medication: MedicationID) -> None:
        self.knowledge.forget(medication)
        self.ledger.forget(medication)
        self.physical.forget(medication)

    def physical_ranges(self) -> dict[MedicationID, PhysicalRange]:
        """Size and colour ranges of every medication confirmed often enough (see ``core.physical``)."""
        return self.physical.ranges(self.physical_policy)

    # ------------------------------------------------------------------ calibration

    @property
    def needs_recalibration(self) -> bool:
        """Memory grew by a quarter, and by at least ten pills, since the last calibration."""
        count = len(self.knowledge.exemplars)
        return count >= self.exemplars_at_calibration + max(10, self.exemplars_at_calibration // 4)

    def recalibrate(
        self, calibrator: IdentityCalibrator | None = None, at: datetime | None = None
    ) -> CalibrationReport:
        """Re-learns thresholds. If they change, every medication must rebuild its streak."""
        report, policy = (calibrator or IdentityCalibrator()).calibrate(self.knowledge, self.identity_policy, at)
        self.last_calibration = report
        self.exemplars_at_calibration = len(self.knowledge.exemplars)
        if policy is not None and (
            policy.accept_similarity != self.identity_policy.accept_similarity
            or policy.minimum_margin != self.identity_policy.minimum_margin
        ):
            self.identity_policy = policy
            self.ledger.reset_streaks()
        return report

    def migrated(self, embedder_id: EmbedderID, vectors: dict[str, np.ndarray]) -> Brain:
        """Moves to a new embedder: exemplars are re-embedded from stored crops (missing ones dropped);
        thresholds and track records start again because the space changed."""
        nxt = Brain(embedder_id, self.trust_policy)
        plan = LearningPlan()
        for e in self.knowledge.exemplars:
            if e.id in vectors:
                plan.exemplars.append(
                    Exemplar(
                        medication_id=e.medication_id,
                        vector=vectors[e.id],
                        embedder_id=embedder_id,
                        source=e.source,
                        group_id=e.group_id,
                        id=e.id,
                        created_at=e.created_at,
                        crop_file=e.crop_file,
                    )
                )
        nxt.learn(plan)
        nxt.physical = self.physical  # size and colour do not depend on the eye
        nxt.physical_policy = self.physical_policy
        return nxt

    @property
    def summary(self) -> BrainSummary:
        return BrainSummary(
            embedder_id=self.embedder_id,
            policy_version=self.identity_policy.version,
            exemplar_count=len(self.knowledge.exemplars),
            known_medications=len(self.knowledge.medications),
            trusted_medications=sorted(self.trusted_medications),
        )
