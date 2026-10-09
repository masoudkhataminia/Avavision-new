from __future__ import annotations

import time
from collections import Counter

import cv2
import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from avavision.brain.brain import Brain
from avavision.core.engine import CompartmentStatus, FindingKind, PackStatus, VerificationEngine
from avavision.core.gate import builtin_pocket_segmenter
from avavision.core.models import ExpectedItem, PackProfile, cell
from avavision.core.observation import CaptureIssue, pill_compartments
from avavision.vision.camera import LiveFeed, StillSource
from avavision.vision.detector import OnnxDetector, decode
from avavision.vision.embedder import ClassicEmbedder, square_fit
from avavision.vision.pack_finder import FinderMode, find_by_outline, marker_sheet
from avavision.vision.pipeline import FrameAnalyzer, propagate_identities
from avavision.vision.quality import assess
from avavision.vision.synthetic import Scene, Station, full_scene

STATION = Station()
LAYOUT = STATION.layout()
MODEL = builtin_pocket_segmenter()
TWO = ["metformin-500", "aspirin-100"]


def analyse(image, analyzer=None):
    analyzer = analyzer or FrameAnalyzer(LAYOUT)
    return analyzer.detect(analyzer.locate(image))


def counts(frame, minimum=0.6):
    return Counter(pill_compartments(frame.observation, LAYOUT, minimum, MODEL.meaning).values())


def test_markers_register_the_pack_and_quality_passes():
    frame = FrameAnalyzer(LAYOUT).locate(STATION.render(full_scene(LAYOUT, TWO)))
    assert frame.observation.is_usable, (frame.observation.quality, frame.observation.registration)
    quad = frame.observation.registration.registration.quad
    x0, y0, x1, y1 = STATION.frame
    assert quad.top_left.x == pytest.approx(x0 / STATION.width, abs=0.003)
    assert quad.bottom_right.y == pytest.approx(y1 / STATION.height, abs=0.003)


def test_missing_markers_mean_no_registration():
    image = STATION.render(full_scene(LAYOUT, TWO))
    image[:200, :400] = 40  # cover the top-left marker
    assert FrameAnalyzer(LAYOUT).locate(image).observation.registration.registration is None


def test_every_compartment_is_counted_exactly():
    scene = full_scene(LAYOUT, TWO)
    scene.contents[cell(2, 5)] = ["perindopril-5"]
    scene.contents[cell(0, 0)] = []
    c = counts(analyse(STATION.render(scene, seed=3)))
    for index in LAYOUT.all_compartments:
        assert c.get(index, 0) == len(scene.contents[index]), index


def test_touching_pills_are_never_counted_confidently():
    scene = Scene(contents={cell(1, 1): ["atorvastatin-20", "atorvastatin-20"]})
    station = Station()
    image = station.render(scene)
    # Push the two pills together.
    layout = station.layout()
    x0, y0, x1, y1 = station.frame
    r = layout.cell_rect(cell(1, 1))
    cx, cy = int(x0 + r.center.x * (x1 - x0)), int(y0 + r.center.y * (y1 - y0))
    image = station.render(Scene(), seed=1)
    for dx in (-13, 13):
        cv2.circle(image, (cx + dx, cy), 14, (250, 250, 250), cv2.FILLED)
        cv2.circle(image, (cx + dx, cy), 14, (190, 190, 190), 2)
    frame = analyse(image)
    pocket = [
        d
        for i, d in enumerate(frame.observation.detections)
        if pill_compartments(frame.observation, LAYOUT, 0, MODEL.meaning).get(i) == cell(1, 1)
    ]
    assert len(pocket) >= 2 and all(d.confidence < 0.6 for d in pocket)


def test_engine_on_real_pixels_count_matches_and_finds_missing_dose():
    scene = full_scene(LAYOUT, TWO)
    profile = PackProfile.empty("P", LAYOUT)
    for index in LAYOUT.all_compartments:
        profile.set_items(index, [ExpectedItem(medication_id=m, quantity=1) for m in TWO])
    engine = VerificationEngine(LAYOUT, MODEL)
    frames = [analyse(STATION.render(scene, seed=s)).observation for s in range(3)]
    result = engine.evaluate(profile, frames)
    assert result.status == PackStatus.COUNT_MATCHED, [
        (str(v.compartment), v.findings) for v in result.compartments if v.status != CompartmentStatus.COUNT_MATCHED
    ]

    scene.contents[cell(3, 6)] = ["metformin-500"]
    frames = [analyse(STATION.render(scene, seed=s)).observation for s in range(3)]
    verdict = engine.evaluate(profile, frames).verdict(cell(3, 6))
    assert verdict.status == CompartmentStatus.MISMATCH and verdict.findings[0].kind == FindingKind.MISSING


def test_blur_and_darkness_are_rejected():
    image = STATION.render(full_scene(LAYOUT, TWO))
    assert CaptureIssue.BLURRY in assess(cv2.GaussianBlur(image, (0, 0), 6)).issues
    assert CaptureIssue.TOO_DARK in assess((image * 0.15).astype(np.uint8)).issues
    assert assess(image).issues == []


def test_outline_finder_on_plain_card():
    image = np.full((900, 1200, 3), 30, np.uint8)
    cv2.rectangle(image, (180, 160), (1020, 740), (230, 230, 230), cv2.FILLED)
    found = find_by_outline(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    assert found is not None and found.quad.top_left.x == pytest.approx(0.15, abs=0.01)
    assert found.quad.bottom_right.y == pytest.approx(740 / 900, abs=0.01)
    assert find_by_outline(np.full((900, 1200), 30, np.uint8)) is None
    assert marker_sheet().shape == (1040, 1040)
    assert FinderMode("markers") == FinderMode.MARKERS


def test_brain_learns_colours_with_classic_embedder_and_propagates():
    embedder = ClassicEmbedder()
    brain = Brain(embedder.id)
    # Rendered pills are pixel-identical, which the duplicate filter would (rightly) collapse.
    brain.knowledge.policy.duplicate_similarity = 1.01
    analyzer = FrameAnalyzer(LAYOUT)
    for seed in range(3):
        for medication in ("aspirin-100", "perindopril-5", "warfarin-1"):
            frame = analyzer.identify(
                analyse(STATION.render(full_scene(LAYOUT, [medication]), seed=10 + seed)),
                embedder,
                brain.classifier(),
                MODEL,
                0.6,
            )
            brain.teach(medication, [(p.vector, None) for p in frame.pills], group_id=f"{medication}-{seed}")
    assert brain.identity_policy.is_calibrated, brain.last_calibration
    scene = full_scene(LAYOUT, ["aspirin-100", "perindopril-5"])
    first = analyzer.identify(analyse(STATION.render(scene, seed=50)), embedder, brain.classifier(), MODEL, 0.6)
    named = [p.identity.decision.named for p in first.pills if p.identity.decision.named]
    assert len(named) > 30 and all(n in ("aspirin-100", "perindopril-5") for n in named)

    second = propagate_identities(first, analyse(STATION.render(scene, seed=51)), LAYOUT)
    copied = [d for d in second.observation.detections if d.identity is not None]
    assert len(copied) == len(first.pills)

    # With the opinions shared across frames, a mixed pack is evaluated consistently.
    profile = PackProfile.empty("P", LAYOUT)
    for index in LAYOUT.all_compartments:
        profile.set_items(index, [ExpectedItem(medication_id=m, quantity=1) for m in ("aspirin-100", "perindopril-5")])
    third = propagate_identities(first, analyse(STATION.render(scene, seed=52)), LAYOUT)
    result = VerificationEngine(LAYOUT, MODEL).evaluate(
        profile, [first.observation, second.observation, third.observation]
    )
    assert result.status == PackStatus.COUNT_MATCHED


def test_detector_decoding_and_onnx_session(tmp_path):
    boxes = np.array([[0.5, 0.5, 0.1, 0.2], [0.2, 0.2, 0.1, 0.1]], dtype=np.float32)
    logits = np.array([[4.0, -4.0], [-6.0, -6.0]], dtype=np.float32)
    decoded = decode(boxes, logits, ["pill", "broken"], 0.25)
    assert len(decoded) == 1 and decoded[0].label == "pill"
    assert decoded[0].box.x == pytest.approx(0.45) and decoded[0].box.height == pytest.approx(0.2)

    graph = helper.make_graph(
        [
            helper.make_node(
                "Constant",
                [],
                ["dets"],
                value=helper.make_tensor("d", TensorProto.FLOAT, [1, 2, 4], boxes.ravel().tolist()),
            ),
            helper.make_node(
                "Constant",
                [],
                ["labels"],
                value=helper.make_tensor("l", TensorProto.FLOAT, [1, 2, 2], logits.ravel().tolist()),
            ),
        ],
        "fake-rfdetr",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 64, 64])],
        [
            helper.make_tensor_value_info("dets", TensorProto.FLOAT, [1, 2, 4]),
            helper.make_tensor_value_info("labels", TensorProto.FLOAT, [1, 2, 2]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.save(model, tmp_path / "fake.onnx")
    detections = OnnxDetector(tmp_path / "fake.onnx", ["pill", "broken"], input_size=64).detect(
        np.zeros((100, 100, 3), np.uint8)
    )
    assert [d.label for d in detections] == ["pill"]


def test_live_feed_keeps_newest_frame_and_square_fit():
    images = [np.full((10, 10, 3), v, np.uint8) for v in (1, 2, 3)]
    feed = LiveFeed(StillSource(images, interval=0.001))
    feed.start()
    deadline = time.time() + 2
    while feed.frames_read < 5 and time.time() < deadline:
        time.sleep(0.01)
    feed.stop()
    assert feed.frames_read >= 5 and feed.latest() is not None
    assert square_fit(np.zeros((20, 40, 3), np.uint8), 32).shape == (32, 32, 3)
