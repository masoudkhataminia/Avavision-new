"""Open-set identification: which known medication does this pill look like, if any?"""

from __future__ import annotations

from enum import StrEnum

import numpy as np
from pydantic import BaseModel, ConfigDict

MedicationID = str
EmbedderID = str


class IdentityPolicy(BaseModel):
    """Turns similarities into decisions. Thresholds depend on the embedder and are learned by calibration;
    an uncalibrated policy never names a medication."""

    neighbours: int = 5
    minimum_exemplars: int = 5
    minimum_groups: int = 2
    accept_similarity: float | None = None
    minimum_margin: float = 0.0
    version: int = 0

    @property
    def is_calibrated(self) -> bool:
        return self.accept_similarity is not None


class IdentityCandidate(BaseModel):
    model_config = ConfigDict(frozen=True)

    medication_id: MedicationID
    score: float
    support: int


class DecisionKind(StrEnum):
    IDENTIFIED = "identified"  # a claim, not yet a trusted fact
    AMBIGUOUS = "ambiguous"  # two medications too close to call
    UNRECOGNISED = "unrecognised"  # unlike everything the brain knows well
    INSUFFICIENT_KNOWLEDGE = "insufficientKnowledge"


class IdentityDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: DecisionKind
    medication_id: MedicationID | None = None

    @classmethod
    def identified(cls, medication: MedicationID) -> IdentityDecision:
        return cls(kind=DecisionKind.IDENTIFIED, medication_id=medication)

    @property
    def named(self) -> MedicationID | None:
        return self.medication_id if self.kind == DecisionKind.IDENTIFIED else None


AMBIGUOUS = IdentityDecision(kind=DecisionKind.AMBIGUOUS)
UNRECOGNISED = IdentityDecision(kind=DecisionKind.UNRECOGNISED)
INSUFFICIENT = IdentityDecision(kind=DecisionKind.INSUFFICIENT_KNOWLEDGE)


class IdentityEvidence(BaseModel):
    """The brain's opinion of one pill, kept with the detection for audit."""

    embedder_id: EmbedderID
    policy_version: int
    decision: IdentityDecision
    candidates: list[IdentityCandidate] = []


def normalize(vector: np.ndarray) -> np.ndarray | None:
    v = np.asarray(vector, dtype=np.float32).ravel()
    if v.size == 0 or not np.all(np.isfinite(v)):
        return None
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 0 else None


class KnowledgeIndex:
    """Contiguous matrices per medication for fast nearest-neighbour search (vectorised with numpy)."""

    def __init__(self, embedder_id: EmbedderID, entries: dict[MedicationID, tuple[np.ndarray, list[str]]]):
        self.embedder_id = embedder_id
        self.entries = entries  # medication → (matrix n×d, group ids)
        self.dimension = next((m.shape[1] for m, _ in entries.values()), 0)

    def group_count(self, medication: MedicationID, excluding_group: str | None = None) -> int:
        if medication not in self.entries:
            return 0
        return len({g for g in self.entries[medication][1] if g != excluding_group})

    def candidates(
        self,
        query: np.ndarray,
        neighbours: int,
        excluding_group: str | None = None,
        excluding_medication: MedicationID | None = None,
    ) -> list[IdentityCandidate]:
        if query.shape[-1] != self.dimension or self.dimension == 0:
            return []
        k = max(1, neighbours)
        result = []
        for medication, (matrix, groups) in self.entries.items():
            if medication == excluding_medication:
                continue
            similarities = matrix @ query
            if excluding_group is not None:
                similarities = similarities[np.array([g != excluding_group for g in groups], dtype=bool)]
            if similarities.size == 0:
                continue
            top = np.sort(similarities)[-k:]
            result.append(
                IdentityCandidate(medication_id=medication, score=float(top.mean()), support=int(similarities.size))
            )
        result.sort(key=lambda c: (-c.score, c.medication_id))
        return result


class IdentityClassifier:
    def __init__(self, index: KnowledgeIndex, policy: IdentityPolicy):
        self.index = index
        self.policy = policy

    def classify(self, query: np.ndarray) -> IdentityEvidence:
        return self.evidence(query)

    def classify_many(self, queries: np.ndarray) -> list[IdentityEvidence]:
        return [self.evidence(q) for q in queries]

    def evidence(
        self, query: np.ndarray, excluding_group: str | None = None, excluding_medication: MedicationID | None = None
    ) -> IdentityEvidence:
        candidates = self.index.candidates(
            query, self.policy.neighbours, excluding_group=excluding_group, excluding_medication=excluding_medication
        )
        return IdentityEvidence(
            embedder_id=self.index.embedder_id,
            policy_version=self.policy.version,
            decision=self._decide(candidates, excluding_group),
            candidates=candidates[:3],
        )

    def _decide(self, candidates: list[IdentityCandidate], excluding_group: str | None) -> IdentityDecision:
        threshold = self.policy.accept_similarity
        if threshold is None or not candidates:
            return INSUFFICIENT
        known = [
            c
            for c in candidates
            if c.support >= self.policy.minimum_exemplars
            and self.index.group_count(c.medication_id, excluding_group) >= self.policy.minimum_groups
        ]
        if not known:
            return INSUFFICIENT
        best = candidates[0]
        # A poorly known medication scoring highest must not be overruled by a well-known one.
        if best.medication_id != known[0].medication_id:
            return INSUFFICIENT if best.score >= threshold else UNRECOGNISED
        if best.score < threshold:
            return UNRECOGNISED
        if len(candidates) > 1 and best.score - candidates[1].score < self.policy.minimum_margin:
            return AMBIGUOUS
        return IdentityDecision.identified(best.medication_id)
