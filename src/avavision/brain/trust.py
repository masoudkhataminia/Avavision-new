"""Earned, revocable trust per medication, built only from pills a pharmacist inspected."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel

from .calibration import wilson_lower_bound
from .identity import DecisionKind, IdentityDecision, MedicationID


class TrustPolicy(BaseModel):
    minimum_predictions: int = 100
    minimum_precision_lower_bound: float = 0.97
    #: Consecutive correct identifications since the last error or threshold change.
    minimum_streak: int = 50


class TrackRecord(BaseModel):
    correct: int = 0
    false_identifications: int = 0
    confusions: int = 0
    abstentions: int = 0
    streak: int = 0
    last_false_identification_at: datetime | None = None

    @property
    def named_count(self) -> int:
        return self.correct + self.false_identifications

    @property
    def precision_lower_bound(self) -> float:
        return wilson_lower_bound(self.correct, self.named_count)


class TrustState(StrEnum):
    LEARNING = "learning"
    TRUSTED = "trusted"
    SUSPENDED = "suspended"


class TrustStatus(BaseModel):
    state: TrustState
    progress: float = 0.0


class TrustLedger(BaseModel):
    embedder_id: str
    records: dict[MedicationID, TrackRecord] = {}

    def record(self, medication: MedicationID) -> TrackRecord:
        return self.records.get(medication, TrackRecord())

    def _mutable(self, medication: MedicationID) -> TrackRecord:
        return self.records.setdefault(medication, TrackRecord())

    def observe(self, decision: IdentityDecision, truth: MedicationID, at: datetime | None = None) -> None:
        if decision.kind == DecisionKind.IDENTIFIED and decision.medication_id == truth:
            r = self._mutable(truth)
            r.correct += 1
            r.streak += 1
        elif decision.kind == DecisionKind.IDENTIFIED:
            r = self._mutable(decision.medication_id)
            r.false_identifications += 1
            r.streak = 0
            r.last_false_identification_at = at or datetime.now(UTC)
            self._mutable(truth).confusions += 1
        else:
            self._mutable(truth).abstentions += 1

    def reset_streaks(self) -> None:
        for r in self.records.values():
            r.streak = 0

    def forget(self, medication: MedicationID) -> None:
        self.records.pop(medication, None)

    def status(self, medication: MedicationID, policy: TrustPolicy) -> TrustStatus:
        r = self.record(medication)
        if (
            r.named_count >= policy.minimum_predictions
            and r.precision_lower_bound >= policy.minimum_precision_lower_bound
            and r.streak >= policy.minimum_streak
        ):
            return TrustStatus(state=TrustState.TRUSTED, progress=1.0)
        if r.false_identifications > 0 and r.streak < policy.minimum_streak:
            return TrustStatus(state=TrustState.SUSPENDED)
        parts = [
            r.named_count / max(1, policy.minimum_predictions),
            r.precision_lower_bound / max(1e-4, policy.minimum_precision_lower_bound),
            r.streak / max(1, policy.minimum_streak),
        ]
        return TrustStatus(state=TrustState.LEARNING, progress=min(1.0, min(parts)))

    def trusted(self, policy: TrustPolicy) -> set[MedicationID]:
        return {m for m in self.records if self.status(m, policy).state == TrustState.TRUSTED}
