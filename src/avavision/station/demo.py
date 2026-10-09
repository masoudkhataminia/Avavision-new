"""The demo camera: a simulated station that shows the pack being checked, with optional packing errors.

It lets the whole station (UI, checks, sign-off, learning, audit) be tried without a camera.
"""

from __future__ import annotations

import random
import threading
import time
from enum import StrEnum

import numpy as np

from ..core.models import Appearance, Catalog, ExpectedItem, Medication, PackProfile, cell
from ..vision.camera import FrameSource
from ..vision.synthetic import PALETTE, Scene, Station

DEMO_MEDICATIONS = [
    Medication(
        id="metformin-500", name="Metformin", strength="500 mg", appearance=Appearance(colour="white", shape="oval")
    ),
    Medication(
        id="atorvastatin-20",
        name="Atorvastatin",
        strength="20 mg",
        appearance=Appearance(colour="white", shape="round"),
    ),
    Medication(
        id="aspirin-100", name="Aspirin", strength="100 mg", appearance=Appearance(colour="orange", shape="round")
    ),
    Medication(
        id="perindopril-5", name="Perindopril", strength="5 mg", appearance=Appearance(colour="green", shape="oval")
    ),
    Medication(
        id="amlodipine-5", name="Amlodipine", strength="5 mg", appearance=Appearance(colour="white", shape="capsule")
    ),
    Medication(id="warfarin-1", name="Warfarin", strength="1 mg", appearance=Appearance(colour="brown", shape="round")),
]


class DemoFault(StrEnum):
    NONE = "none"
    MISSING = "missing"  # one dose left out
    EXTRA = "extra"  # one dose too many
    SWAPPED = "swapped"  # one dose replaced by another medication
    FOREIGN = "foreign"  # something that is not a tablet
    EMPTY_TRAY = "emptyTray"


def demo_catalog() -> Catalog:
    return Catalog(medications=list(DEMO_MEDICATIONS))


def demo_profile(station: Station) -> PackProfile:
    layout = station.layout()
    profile = PackProfile.empty("DEMO-001", layout)
    plan = {
        0: [("metformin-500", 1), ("aspirin-100", 1)],
        1: [("perindopril-5", 1)],
        2: [("metformin-500", 1), ("atorvastatin-20", 1)],
        3: [("amlodipine-5", 1)],
    }
    for row, items in plan.items():
        for column in range(layout.columns):
            profile.set_items(cell(row, column), [ExpectedItem(medication_id=m, quantity=q) for m, q in items])
    return profile


def scene_for(profile: PackProfile | None, fault: DemoFault, seed: int = 0) -> Scene:
    scene = Scene()
    if profile is None or fault == DemoFault.EMPTY_TRAY:
        return scene
    for expectation in profile.compartments:
        scene.contents[expectation.compartment] = [
            item.medication_id
            for item in expectation.items
            for _ in range(item.quantity)
            if item.medication_id in PALETTE
        ]
    filled = sorted(i for i, c in scene.contents.items() if c)
    if not filled or fault == DemoFault.NONE:
        return scene
    rng = random.Random(seed)
    target = rng.choice(filled)
    contents = scene.contents[target]
    if fault == DemoFault.MISSING:
        contents.pop(rng.randrange(len(contents)))
    elif fault == DemoFault.EXTRA:
        contents.append(contents[0])
    elif fault == DemoFault.SWAPPED:
        k = rng.randrange(len(contents))
        contents[k] = next(m for m in PALETTE if m not in contents)
    elif fault == DemoFault.FOREIGN:
        scene.foreign.append(target)
    return scene


class DemoCamera(FrameSource):
    """Cycles through a few renders of the current scene (each with its own sensor noise)."""

    def __init__(self, station: Station | None = None, interval: float = 1 / 15, variants: int = 4):
        self.station = station or Station()
        self.interval = interval
        self.variants = variants
        self._lock = threading.Lock()
        self._frames: list[np.ndarray] = []
        self._position = 0
        self.fault = DemoFault.NONE
        self.fault_seed = 0
        self.show(None, DemoFault.EMPTY_TRAY)

    def show(self, profile: PackProfile | None, fault: DemoFault = DemoFault.NONE, seed: int = 0) -> None:
        scene = scene_for(profile, fault, seed)
        frames = [self.station.render(scene, seed=1000 + k) for k in range(self.variants)]
        with self._lock:
            self._frames = frames
            self.fault = fault
            self.fault_seed = seed

    def read(self) -> np.ndarray | None:
        time.sleep(self.interval)
        with self._lock:
            frame = self._frames[self._position % len(self._frames)]
            self._position += 1
        return frame.copy()
