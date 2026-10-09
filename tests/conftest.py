"""Shared builders for engine-level tests (mirrors the original Swift fixtures)."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from avavision.brain.identity import IdentityDecision, IdentityEvidence
from avavision.core.engine import BrainContext, VerificationEngine
from avavision.core.gate import (
    ActiveModel,
    CapabilityKind,
    GateDecision,
    ModelCapability,
    ModelManifest,
    ModelStage,
    evaluate_gate,
)
from avavision.core.geometry import Point, Quad, Rect
from avavision.core.models import Catalog, ExpectedItem, Medication, PackLayout, PackProfile, cell
from avavision.core.observation import ACCEPTED, Detection, FrameObservation, RegistrationOutcome, register

METFORMIN = "metformin-500"
ATORVASTATIN = "atorvastatin-20"
ASPIRIN = "aspirin-100"
FIXED = datetime(2027, 1, 15, 8, 0, tzinfo=UTC)

CATALOG = Catalog(
    medications=[
        Medication(id=METFORMIN, name="Metformin", strength="500 mg"),
        Medication(id=ATORVASTATIN, name="Atorvastatin", strength="20 mg"),
        Medication(id=ASPIRIN, name="Aspirin", strength="100 mg"),
    ]
)

LAYOUT = PackLayout(
    id="test-2x3",
    display_name="Test 2×3",
    rows=2,
    columns=3,
    row_labels=["AM", "PM"],
    column_labels=["D1", "D2", "D3"],
    width_mm=150,
    height_mm=100,
    grid_region=Rect(x=0, y=0, width=1, height=1),
    border_band=0.1,
    is_calibrated=True,
)

QUAD = Quad(
    top_left=Point(x=0.1, y=0.1),
    top_right=Point(x=0.9, y=0.1),
    bottom_right=Point(x=0.9, y=0.9),
    bottom_left=Point(x=0.1, y=0.9),
)

REGISTRATION = register(QUAD, 0.95, 1500, 1000, LAYOUT).registration
assert REGISTRATION is not None
PACK_TO_IMAGE = REGISTRATION.image_to_pack.inverse

TARGET = cell(0, 1)
ONE = [ExpectedItem(medication_id=METFORMIN, quantity=1)]

GENERIC_LABELS = {"pill": "pill", "broken": "broken", "foreign": "foreign", "background": "ignore"}
IDENTITY_LABELS = GENERIC_LABELS | {
    "metformin": f"medication:{METFORMIN}",
    "atorvastatin": f"medication:{ATORVASTATIN}",
    "aspirin": f"medication:{ASPIRIN}",
}


def detection(label: str, at: Point, confidence: float = 0.9) -> Detection:
    c = PACK_TO_IMAGE.apply(at)
    return Detection(label=label, confidence=confidence, box=Rect(x=c.x - 0.01, y=c.y - 0.01, width=0.02, height=0.02))


def detections(label: str, count: int, index, confidence: float = 0.9, layout: PackLayout = LAYOUT) -> list[Detection]:
    r = layout.cell_rect(index)
    return [
        detection(label, Point(x=r.center.x + (i - (count - 1) / 2) * r.width * 0.12, y=r.center.y), confidence)
        for i in range(count)
    ]


def with_identity(d: Detection, decision: IdentityDecision) -> Detection:
    return d.model_copy(update={"identity": IdentityEvidence(embedder_id="e", policy_version=1, decision=decision)})


def frame(dets: list[Detection], quality=ACCEPTED) -> FrameObservation:
    return FrameObservation(
        captured_at=FIXED, quality=quality, registration=RegistrationOutcome(registration=REGISTRATION), detections=dets
    )


def frames(dets: list[Detection], count: int = 3) -> list[FrameObservation]:
    return [frame(dets) for _ in range(count)]


def profile(default=None, overrides=None) -> PackProfile:
    p = PackProfile(reference="PACK-001", layout_id=LAYOUT.id, compartments=[], created_at=FIXED)
    for index in LAYOUT.all_compartments:
        p.set_items(index, (overrides or {}).get(index, default or []))
    return p


def manifest(stage=ModelStage.DEVELOPMENT, labels=None, evaluation=None) -> ModelManifest:
    return ModelManifest(
        model_id="test-detector",
        version="1.0.0",
        stage=stage,
        model_sha256="abc123",
        labels=labels or GENERIC_LABELS,
        evaluation=evaluation,
    )


def count_only_model() -> ActiveModel:
    m = manifest()
    return ActiveModel(manifest=m, decision=evaluate_gate(m, "abc123"))


def identity_model(medications: set[str]) -> ActiveModel:
    return ActiveModel(
        manifest=manifest(ModelStage.RELEASED, IDENTITY_LABELS),
        decision=GateDecision(
            capability=ModelCapability(kind=CapabilityKind.IDENTITY, medications=frozenset(medications))
        ),
    )


def engine(model="count", layout: PackLayout = LAYOUT, brain: BrainContext | None = None) -> VerificationEngine:
    active = count_only_model() if model == "count" else model
    return VerificationEngine(layout, active, brain)


def full_pack(label: str = "pill") -> list[Detection]:
    return [d for i in LAYOUT.all_compartments for d in detections(label, 1, i)]


class SyntheticPills:
    """Each medication has a prototype; pills are noisy copies of it."""

    embedder = "test-embedder"
    dimension = 32

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.prototypes: dict[str, np.ndarray] = {}

    def prototype(self, medication: str) -> np.ndarray:
        if medication not in self.prototypes:
            self.prototypes[medication] = self.rng.normal(size=self.dimension)
        return self.prototypes[medication]

    def pill(self, medication: str, noise: float = 0.25) -> np.ndarray:
        base = self.prototype(medication)
        base = base / np.linalg.norm(base)
        v = base + noise * self.rng.normal(size=self.dimension) / np.sqrt(self.dimension)
        return (v / np.linalg.norm(v)).astype(np.float32)

    def unknown(self) -> np.ndarray:
        v = self.rng.normal(size=self.dimension)
        return (v / np.linalg.norm(v)).astype(np.float32)


@pytest.fixture
def synthetic() -> SyntheticPills:
    return SyntheticPills()
