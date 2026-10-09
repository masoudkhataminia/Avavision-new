"""The brain's memory: pharmacist-confirmed pill appearances."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

import numpy as np

from .identity import EmbedderID, KnowledgeIndex, MedicationID


class ExemplarSource(StrEnum):
    TEACHING = "teaching"  # pharmacist photographed known pills
    CONFIRMED_CHECK = "confirmedCheck"  # compartment confirmed correct during a check
    LABELLED = "labelled"  # pharmacist labelled the pill in the queue


@dataclass
class Exemplar:
    medication_id: MedicationID
    vector: np.ndarray  # L2-normalized float32
    embedder_id: EmbedderID
    source: ExemplarSource
    #: Pills from the same photo or check share a group; calibration never compares within a group.
    group_id: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    crop_file: str | None = None


class AddOutcome(StrEnum):
    ADDED = "added"
    ADDED_REPLACING = "addedReplacing"
    DUPLICATE = "duplicate"
    INCOMPATIBLE = "incompatible"


@dataclass
class KnowledgePolicy:
    maximum_exemplars_per_medication: int = 400
    duplicate_similarity: float = 0.995


class KnowledgeBase:
    """Memory for one embedder, with capacity per medication and redundancy-based forgetting."""

    def __init__(self, embedder_id: EmbedderID, policy: KnowledgePolicy | None = None):
        self.embedder_id = embedder_id
        self.policy = policy or KnowledgePolicy()
        self._by_medication: dict[MedicationID, list[Exemplar]] = {}
        self._index: KnowledgeIndex | None = None

    @property
    def exemplars(self) -> list[Exemplar]:
        return [e for items in self._by_medication.values() for e in items]

    @property
    def medications(self) -> set[MedicationID]:
        return {m for m, items in self._by_medication.items() if items}

    def exemplars_for(self, medication: MedicationID) -> list[Exemplar]:
        return list(self._by_medication.get(medication, []))

    def count(self, medication: MedicationID) -> int:
        return len(self._by_medication.get(medication, []))

    def group_count(self, medication: MedicationID) -> int:
        return len({e.group_id for e in self._by_medication.get(medication, [])})

    def add(self, exemplar: Exemplar) -> tuple[AddOutcome, str | None]:
        """Returns the outcome and, when capacity forced it, the id of the forgotten exemplar."""
        dimension = next((e.vector.shape[0] for e in self.exemplars[:1]), exemplar.vector.shape[0])
        if exemplar.embedder_id != self.embedder_id or exemplar.vector.shape[0] != dimension:
            return AddOutcome.INCOMPATIBLE, None
        same = self._by_medication.setdefault(exemplar.medication_id, [])
        if same:
            similarities = np.stack([e.vector for e in same]) @ exemplar.vector
            if float(similarities.max()) >= self.policy.duplicate_similarity:
                return AddOutcome.DUPLICATE, None
        same.append(exemplar)
        self._index = None
        if len(same) <= max(1, self.policy.maximum_exemplars_per_medication):
            return AddOutcome.ADDED, None
        forgotten = self._most_redundant(same)
        removed = same.pop(forgotten)
        return AddOutcome.ADDED_REPLACING, removed.id

    def remove(self, exemplar_id: str) -> None:
        for medication, items in self._by_medication.items():
            self._by_medication[medication] = [e for e in items if e.id != exemplar_id]
        self._index = None

    def forget(self, medication: MedicationID) -> None:
        self._by_medication.pop(medication, None)
        self._index = None

    def index(self) -> KnowledgeIndex:
        if self._index is None:
            self._index = KnowledgeIndex(
                self.embedder_id,
                {
                    m: (np.stack([e.vector for e in items]).astype(np.float32), [e.group_id for e in items])
                    for m, items in sorted(self._by_medication.items())
                    if items
                },
            )
        return self._index

    @staticmethod
    def _most_redundant(items: list[Exemplar]) -> int:
        """The exemplar whose nearest same-medication neighbour is closest; ties forget the older one."""
        matrix = np.stack([e.vector for e in items])
        similarity = matrix @ matrix.T
        np.fill_diagonal(similarity, -np.inf)
        nearest = similarity.max(axis=1)
        best = int(np.argmax(nearest))
        ties = np.flatnonzero(nearest == nearest[best])
        return int(min(ties, key=lambda i: items[i].created_at))
