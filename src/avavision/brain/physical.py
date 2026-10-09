"""The brain's physical memory: measured size and colour of pharmacist-confirmed pills, per medication.

Kept apart from the appearance memory: it does not depend on the embedder (it survives a change of eye) and is
not thinned by the embedder's duplicate filter, because near-identical looking pills still add size statistics.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

from ..core.physical import PhysicalFeatures, PhysicalPolicy, PhysicalRange, learn_ranges
from .identity import MedicationID


@dataclass
class PhysicalSample:
    medication_id: MedicationID
    features: PhysicalFeatures
    #: Pills from the same pack share a group; ranges need several packs.
    group_id: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class PhysicalMemory:
    def __init__(self, capacity_per_medication: int = 300):
        self.capacity = capacity_per_medication
        self._samples: dict[MedicationID, list[PhysicalSample]] = {}
        self._ranges: dict[MedicationID, PhysicalRange] | None = None

    @property
    def samples(self) -> list[PhysicalSample]:
        return [s for rows in self._samples.values() for s in rows]

    def count(self, medication: MedicationID) -> int:
        return len(self._samples.get(medication, []))

    def add(self, sample: PhysicalSample) -> None:
        """Keeps the newest ``capacity`` pills of each medication: packaging and suppliers change over time."""
        rows = self._samples.setdefault(sample.medication_id, [])
        rows.append(sample)
        rows.sort(key=lambda s: s.created_at)
        del rows[: max(0, len(rows) - self.capacity)]
        self._ranges = None

    def forget(self, medication: MedicationID) -> None:
        self._samples.pop(medication, None)
        self._ranges = None

    def ranges(self, policy: PhysicalPolicy) -> dict[MedicationID, PhysicalRange]:
        if self._ranges is None:
            self._ranges = learn_ranges(
                {m: [(s.features, s.group_id) for s in rows] for m, rows in self._samples.items()}, policy
            )
        return self._ranges
