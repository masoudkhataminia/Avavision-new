from __future__ import annotations

import pytest

from avavision.core.engine import CompartmentStatus, FindingKind, PackStatus
from avavision.core.models import WEEKLY_7X4_PORTRAIT, ExpectedItem, PackProfile, cell
from avavision.core.peers import PeerPolicy, apply_peer_check, peer_groups, unlike_peers
from avavision.station.demo import DemoFault, demo_profile, scene_for
from avavision.vision.peers import card_hue, compartment_signatures
from avavision.vision.pipeline import FrameAnalyzer
from avavision.vision.synthetic import Station
from conftest import ONE, TARGET, engine, frames, full_pack, profile

MORNING = [
    ExpectedItem(medication_id="metformin-500", quantity=1),
    ExpectedItem(medication_id="aspirin-100", quantity=1),
]


def weekly_profile() -> PackProfile:
    p = PackProfile.empty("P", WEEKLY_7X4_PORTRAIT)
    for day in range(7):
        weekly = [ExpectedItem(medication_id="alendronate-70", quantity=1)] if day == 0 else []
        p.set_items(cell(day, 0), MORNING + weekly)
        p.set_items(cell(day, 1), [ExpectedItem(medication_id="perindopril-5", quantity=1)] if day < 2 else [])
    return p


def test_compartments_meant_to_hold_the_same_tablets_form_groups():
    groups = peer_groups(weekly_profile(), WEEKLY_7X4_PORTRAIT)
    # Monday's morning holds the weekly tablet too, so it is not compared with the other mornings; two noon
    # doses are too few to compare and empty compartments are never grouped.
    assert groups == [[cell(day, 0) for day in range(1, 7)]]


def test_one_compartment_unlike_its_group_is_found_and_noise_is_not():
    group = [cell(day, 0) for day in range(7)]
    base = [0.0, 0.2, 0.05, 0.0]
    signatures = {i: [v + 0.004 * (k % 3) for v in base] for k, i in enumerate(group)}
    assert unlike_peers(signatures, [group]) == {}
    signatures[cell(4, 0)] = [0.0, 0.1, 0.05, 0.0]  # half of the coloured tablet area is missing
    assert set(unlike_peers(signatures, [group])) == {cell(4, 0)}
    # Without a signature (obscured) a compartment takes no part; too few left means no comparison.
    few = {i: signatures[i] for i in group[:2]}
    assert unlike_peers(few, [group]) == {}
    assert unlike_peers(signatures, [group], PeerPolicy(minimum_distance=1.0)) == {}


def test_peer_check_only_escalates():
    result = engine().evaluate(profile(ONE), frames(full_pack()))
    assert result.status == PackStatus.COUNT_MATCHED
    checked = apply_peer_check(result, {TARGET: 0.2})
    verdict = checked.verdict(TARGET)
    assert verdict.status == CompartmentStatus.NEEDS_REVIEW and verdict.findings[-1].kind == FindingKind.UNLIKE_PEERS
    assert checked.status == PackStatus.NEEDS_REVIEW
    assert apply_peer_check(checked, {TARGET: 0.2}) == checked and apply_peer_check(result, {}) == result
    pack = full_pack()
    missing = engine().evaluate(profile(ONE), frames(pack[:1] + pack[2:]))
    mismatched = next(v for v in missing.compartments if v.status == CompartmentStatus.MISMATCH)
    escalated = apply_peer_check(missing, {mismatched.compartment: 0.3}).verdict(mismatched.compartment)
    assert escalated.status == CompartmentStatus.MISMATCH  # never relaxed


@pytest.fixture(scope="module")
def station_frames():
    station = Station()
    layout = station.layout()
    p = demo_profile(station)
    analyzer = FrameAnalyzer(layout)

    def canvas(scene):
        frame = analyzer.detect(analyzer.locate(station.render(scene, seed=4)))
        return frame.rectified.canvas

    return layout, p, canvas


@pytest.mark.parametrize("change", ["none", "swapped", "missing"])
def test_a_coloured_tablet_swapped_or_missing_on_one_day_stands_out(station_frames, change):
    layout, p, canvas = station_frames
    scene = scene_for(p, DemoFault.NONE)
    target = cell(0, 3)  # Day 4 morning: metformin and the orange aspirin
    if change == "swapped":
        scene.contents[target] = ["metformin-500", "perindopril-5"]
    elif change == "missing":
        scene.contents[target] = ["metformin-500"]
    image = canvas(scene)
    assert card_hue(image, layout) is None  # the simulated card is grey
    unlike = unlike_peers(compartment_signatures(image, layout), peer_groups(p, layout))
    assert set(unlike) == (set() if change == "none" else {target})
