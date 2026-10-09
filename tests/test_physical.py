from __future__ import annotations

import statistics

import pytest

from avavision.brain.brain import Brain
from avavision.brain.physical import PhysicalSample
from avavision.core.engine import CompartmentStatus, FindingKind
from avavision.core.gate import builtin_pocket_segmenter
from avavision.core.physical import (
    PhysicalEvidence,
    PhysicalFeatures,
    PhysicalPolicy,
    fits_none,
    learn_ranges,
    physical_evidence,
)
from avavision.storage.database import Database
from avavision.vision.embedder import ClassicEmbedder
from avavision.vision.pipeline import FrameAnalyzer
from avavision.vision.synthetic import Station, full_scene
from conftest import ASPIRIN, METFORMIN, ONE, TARGET, detections, engine, frames, profile

WHITE_OVAL = PhysicalFeatures(length_mm=11.5, width_mm=6.8, lightness=93, a=0, b=2)
ORANGE_ROUND = PhysicalFeatures(length_mm=5.7, width_mm=5.7, lightness=75, a=12, b=60)


def jitter(f: PhysicalFeatures, k: int) -> PhysicalFeatures:
    d = (k % 5 - 2) * 0.05
    return f.model_copy(update={"length_mm": f.length_mm + d, "width_mm": f.width_mm - d, "a": f.a + d * 10})


def samples(medication, f, packs=2, per_pack=6):
    return [(jitter(f, k), f"{medication}-{k % packs}") for k in range(packs * per_pack)]


def test_ranges_need_enough_pills_from_enough_packs():
    assert learn_ranges({METFORMIN: samples(METFORMIN, WHITE_OVAL, packs=1)}) == {}
    assert learn_ranges({METFORMIN: samples(METFORMIN, WHITE_OVAL, per_pack=3)}) == {}
    ranges = learn_ranges({METFORMIN: samples(METFORMIN, WHITE_OVAL), ASPIRIN: samples(ASPIRIN, ORANGE_ROUND)})
    assert ranges[METFORMIN].admits(WHITE_OVAL) and not ranges[METFORMIN].admits(ORANGE_ROUND)
    # Same colour, a different size: a white round tablet is not metformin.
    white_round = WHITE_OVAL.model_copy(update={"length_mm": 6.6, "width_mm": 6.6})
    assert not ranges[METFORMIN].admits(white_round)
    # Small variations in size and light stay inside.
    assert ranges[METFORMIN].admits(WHITE_OVAL.model_copy(update={"length_mm": 12.0, "lightness": 85}))


def test_fits_none_only_when_every_expected_medication_is_known():
    ranges = learn_ranges({METFORMIN: samples(METFORMIN, WHITE_OVAL)})
    orange = physical_evidence(ORANGE_ROUND, ranges)
    assert fits_none(orange, {METFORMIN})
    assert not fits_none(orange, {METFORMIN, ASPIRIN})  # aspirin has no ranges yet: no claim
    assert not fits_none(physical_evidence(WHITE_OVAL, ranges), {METFORMIN})
    assert not fits_none(None, {METFORMIN}) and not fits_none(orange, set())


def test_engine_escalates_a_pill_of_the_wrong_size_or_colour_and_never_accepts_on_it():
    ranges = learn_ranges({METFORMIN: samples(METFORMIN, WHITE_OVAL)})

    def with_physical(features):
        return [
            d.model_copy(update={"physical": physical_evidence(features, ranges)})
            for d in detections("pill", 1, TARGET)
        ]

    result = engine().evaluate(profile(overrides={TARGET: ONE}), frames(with_physical(ORANGE_ROUND)))
    verdict = result.verdict(TARGET)
    assert verdict.status == CompartmentStatus.NEEDS_REVIEW
    assert [f.kind for f in verdict.findings][:1] == [FindingKind.PHYSICAL_MISMATCH]

    fine = engine().evaluate(profile(overrides={TARGET: ONE}), frames(with_physical(WHITE_OVAL))).verdict(TARGET)
    assert fine.status == CompartmentStatus.COUNT_MATCHED  # consistent size and colour accept nothing by themselves
    assert FindingKind.PHYSICAL_MISMATCH not in [f.kind for f in fine.findings]


def test_measured_sizes_on_the_simulated_station():
    station = Station()
    layout = station.layout()
    analyzer = FrameAnalyzer(layout)
    brain = Brain(ClassicEmbedder.id)
    sizes = {}
    for medication in ("metformin-500", "aspirin-100", "atorvastatin-20"):
        frame = analyzer.detect(analyzer.locate(station.render(full_scene(layout, [medication]), seed=4)))
        frame = analyzer.identify(frame, ClassicEmbedder(), brain.classifier(), builtin_pocket_segmenter(), 0.6)
        features = [p.features for p in frame.pills if p.features]
        assert len(features) == len(frame.pills) == 28
        sizes[medication] = (
            statistics.median(f.length_mm for f in features),
            statistics.median(f.width_mm for f in features),
            statistics.median(f.b for f in features),
        )
    # Rendered sizes (pixel axes + outline at 0.2 mm per pixel): 11.2 × 6.4, 5.6 round, 6.4 round.
    assert sizes["metformin-500"][:2] == pytest.approx((11.2, 6.4), abs=0.8)
    assert sizes["aspirin-100"][:2] == pytest.approx((5.6, 5.6), abs=0.8)
    assert sizes["atorvastatin-20"][:2] == pytest.approx((6.4, 6.4), abs=0.8)
    assert sizes["aspirin-100"][2] > 40 > abs(sizes["metformin-500"][2])  # orange vs white


def test_physical_memory_learns_survives_storage_and_a_new_eye(tmp_path):
    brain = Brain("eye-1")
    brain.physical_policy = PhysicalPolicy(minimum_samples=4)
    brain.teach(METFORMIN, [], group_id="a", measurements=[WHITE_OVAL] * 3)
    brain.teach(METFORMIN, [], group_id="b", measurements=[jitter(WHITE_OVAL, 1)] * 3)
    assert METFORMIN in brain.physical_ranges()

    db = Database(tmp_path)
    db.save_brain(brain)
    loaded = db.load_brain("eye-1")
    assert loaded.physical.count(METFORMIN) == 6
    assert loaded.migrated("eye-2", {}).physical.count(METFORMIN) == 6
    loaded.forget(METFORMIN)
    db.save_brain(loaded)
    assert db.load_brain("eye-1").physical.count(METFORMIN) == 0

    memory = Brain("eye-1").physical
    memory.capacity = 3
    for _ in range(5):
        memory.add(PhysicalSample(METFORMIN, WHITE_OVAL, "g"))
    assert memory.count(METFORMIN) == 3
    assert isinstance(physical_evidence(WHITE_OVAL, {}), PhysicalEvidence)
