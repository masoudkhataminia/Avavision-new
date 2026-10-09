"""What the brain learns from each pharmacist decision."""

from __future__ import annotations

import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np

from ..core.models import CompartmentIndex, PackLayout, PackProfile
from ..core.physical import PhysicalFeatures
from ..core.signoff import ReviewOutcome, SignOff
from .identity import EmbedderID, IdentityDecision, IdentityEvidence, MedicationID
from .knowledge import Exemplar, ExemplarSource
from .physical import PhysicalSample


@dataclass
class PillSighting:
    """A pill seen during a check, with what the brain thought of it."""

    compartment: CompartmentIndex
    vector: np.ndarray | None
    embedder_id: EmbedderID | None
    identity: IdentityEvidence | None
    crop_file: str | None = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    features: PhysicalFeatures | None = None


@dataclass
class Observation:
    decision: IdentityDecision
    truth: MedicationID


@dataclass
class LearningPlan:
    exemplars: list[Exemplar] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    tasks: list[LabellingTask] = field(default_factory=list)
    measurements: list[PhysicalSample] = field(default_factory=list)

    def learn(self, sighting: PillSighting, medication: MedicationID, source: ExemplarSource, group: str, at: datetime):
        if sighting.vector is not None and sighting.embedder_id is not None:
            self.exemplars.append(
                Exemplar(
                    medication_id=medication,
                    vector=sighting.vector,
                    embedder_id=sighting.embedder_id,
                    source=source,
                    group_id=group,
                    created_at=at,
                    crop_file=sighting.crop_file,
                )
            )
        if sighting.identity is not None:
            self.observations.append(Observation(sighting.identity.decision, medication))
        if sighting.features is not None:
            self.measurements.append(PhysicalSample(medication, sighting.features, group, created_at=at))


class LabellingError(ValueError):
    pass


@dataclass
class LabellingTask:
    """Pills of a confirmed compartment holding several medications; the pharmacist says which is which."""

    check_id: str
    compartment_label: str
    expected: dict[MedicationID, int]
    sightings: list[PillSighting]
    priority: float
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def resolve(self, labels: dict[str, MedicationID], at: datetime | None = None) -> LearningPlan:
        missing = [s.id for s in self.sightings if s.id not in labels]
        if missing:
            raise LabellingError(f"unlabelled pills: {missing}")
        if Counter(labels[s.id] for s in self.sightings) != Counter({k: v for k, v in self.expected.items() if v > 0}):
            raise LabellingError("labels do not add up to the expected quantities")
        plan = LearningPlan()
        for s in self.sightings:
            plan.learn(s, labels[s.id], ExemplarSource.LABELLED, self.check_id, at or datetime.now(UTC))
        return plan


def plan_learning(
    check_id: str,
    sign_off: SignOff,
    profile: PackProfile,
    layout: PackLayout,
    sightings: list[PillSighting],
    at: datetime | None = None,
) -> LearningPlan:
    """Learns only from compartments a pharmacist personally inspected and confirmed correct.

    Automatically accepted compartments are never fed back (the brain must not confirm itself); corrected
    or unresolved compartments are skipped because their true content is unknown. ``sightings`` are the
    pills of one frame (the evidence frame) clearly inside a compartment.
    """
    at = at or datetime.now(UTC)
    plan = LearningPlan()
    by_compartment: dict[CompartmentIndex, list[PillSighting]] = {}
    for s in sightings:
        by_compartment.setdefault(s.compartment, []).append(s)
    for review in sign_off.reviews:
        if review.outcome != ReviewOutcome.CONFIRMED_CORRECT:
            continue
        expectation = profile.expectation(review.compartment)
        if expectation is None:
            continue
        quantities = {k: v for k, v in expectation.quantities.items() if v > 0}
        seen = by_compartment.get(review.compartment, [])
        if not seen or len(seen) != expectation.total_quantity:
            continue
        if len(quantities) == 1:
            medication = next(iter(quantities))
            for s in seen:
                plan.learn(s, medication, ExemplarSource.CONFIRMED_CHECK, check_id, at)
        elif all(s.vector is not None for s in seen):
            unsure = sum(1 for s in seen if s.identity is None or s.identity.decision.named is None)
            plan.tasks.append(
                LabellingTask(
                    check_id=check_id,
                    compartment_label=layout.label(review.compartment),
                    expected=quantities,
                    sightings=seen,
                    priority=unsure / len(seen),
                    created_at=at,
                )
            )
    return plan
