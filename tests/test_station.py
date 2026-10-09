from __future__ import annotations

import time
from types import SimpleNamespace

import cv2
import numpy as np
import onnx
import pytest
from fastapi.testclient import TestClient
from onnx import TensorProto, helper, numpy_helper

from avavision.core.gate import CapabilityKind, ModelManifest, ModelStage
from avavision.expert.charts import ChartDose, ChartLine, DoseTime, MedicationChart, Weekday
from avavision.expert.claude import MODEL, ClaudeExpert, CompartmentReading, Match
from avavision.station.api import create_app
from avavision.station.demo import DemoCamera
from avavision.station.service import Station, StationSettings, load_detector
from avavision.storage.database import Database
from avavision.vision.runtime import file_sha256


@pytest.fixture
def client(tmp_path):
    station = Station.open(tmp_path, demo=True)
    with TestClient(create_app(station)) as client:
        client.station = station
        yield client


def wait_until_live(client, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        live = client.get("/api/live").json()
        if live and live["usable"]:
            return live
        time.sleep(0.05)
    raise AssertionError("the demo camera never produced a usable frame")


def reviews(view, outcome="confirmedCorrect"):
    return [
        {"compartment": {"row": c["row"], "column": c["column"]}, "outcome": outcome}
        for c in view["result"]["compartments"]
    ]


def run_check(client, profile_id, fault=None):
    started = client.post("/api/check/start", json={"profile_id": profile_id})
    assert started.status_code == 200, started.text
    if fault:
        client.post("/api/demo/fault", json={"fault": fault})
        time.sleep(0.3)
    wait_until_live(client)
    captured = client.post("/api/check/capture")
    assert captured.status_code == 200, captured.text
    return captured.json()


def test_full_check_sign_off_audit_and_learning(client):
    status = client.get("/api/status").json()
    assert status["demo"] and status["layout"]["calibrated"] and status["audit"]["entries"] == 0
    profiles = client.get("/api/profiles").json()
    assert [p["reference"] for p in profiles] == ["DEMO-001"] and profiles[0]["issues"] == 0

    view = run_check(client, profiles[0]["id"])
    result = view["result"]
    assert result["status"] == "countMatched", [c for c in result["compartments"] if c["status"] != "countMatched"]
    assert all(c["observed_count"] == c["expected_count"] for c in result["compartments"])
    assert view["timings"]["total_ms"] > 0 and view["timings"]["pills_identified"] == 42
    assert sum(d["counted"] for d in view["detections"]) == 42
    assert client.get("/api/check/evidence.jpg").headers["content-type"] == "image/jpeg"

    refused = client.post("/api/check/sign-off", json={"pharmacist": "MK", "decision": "released", "reviews": []})
    assert refused.status_code == 422 and refused.json()["kind"] == "unreviewedCompartments"

    signed = client.post(
        "/api/check/sign-off", json={"pharmacist": "MK", "decision": "released", "reviews": reviews(view)}
    ).json()
    assert signed["phase"] == "completed" and signed["audit_sequence"] == 0
    # Single-medication rows teach the brain directly; mixed rows become labelling tasks.
    assert signed["learning"]["exemplars_added"] + signed["learning"]["duplicates_skipped"] == 14
    assert signed["learning"]["tasks_queued"] == 14

    brain = client.get("/api/brain").json()
    assert len(brain["tasks"]) == 14 and brain["summary"]["exemplar_count"] >= 2
    task = brain["tasks"][0]
    crop = client.get(f"/api/crops/{task['pills'][0]['crop']}")
    assert crop.status_code == 200
    expected = [m for m, q in task["expected"].items() for _ in range(q)]
    wrong = client.post(f"/api/brain/tasks/{task['id']}/resolve", json={"labels": {}})
    assert wrong.status_code == 422
    labels = {p["id"]: m for p, m in zip(task["pills"], expected, strict=True)}
    assert client.post(f"/api/brain/tasks/{task['id']}/resolve", json={"labels": labels}).status_code == 200

    audit = client.get("/api/audit").json()
    assert audit["total"] == 1 and audit["entries"][0]["pharmacist"] == "MK"
    assert client.get("/api/audit/0").json()["record"]["sign_off"]["decision"] == "released"
    assert client.post("/api/audit/verify").json()["intact"]


def test_demo_fault_is_caught_and_pack_can_be_withheld(client):
    profile_id = client.get("/api/profiles").json()[0]["id"]
    view = run_check(client, profile_id, fault="missing")
    statuses = [c["status"] for c in view["result"]["compartments"]]
    assert view["result"]["status"] == "mismatch" and statuses.count("mismatch") == 1
    flagged = next(c for c in view["result"]["compartments"] if c["status"] == "mismatch")
    assert flagged["findings"][0].startswith("fewer doses than expected")
    # A check cannot be restarted while one awaits sign-off; withholding is always allowed.
    assert client.post("/api/check/start", json={"profile_id": profile_id}).status_code == 409
    withheld = client.post("/api/check/sign-off", json={"pharmacist": "MK", "decision": "withheld", "reviews": []})
    assert withheld.json()["phase"] == "completed"


def test_teaching_and_settings(client):
    taught = client.post("/api/brain/teach", json={"medication_id": "aspirin-100"})
    assert taught.status_code == 200, taught.text
    assert taught.json()["pills"] == 56
    assert client.post("/api/brain/teach", json={"medication_id": "nope"}).status_code == 409

    settings = client.put("/api/settings", json={"language": "en", "keep_evidence_images": False}).json()
    assert settings["language"] == "en" and not settings["keep_evidence_images"]
    assert client.get("/api/settings").json()["language"] == "en"
    assert client.get("/api/markers.png").headers["content-type"] == "image/png"
    assert client.get("/api/camera/rectified.jpg").status_code == 200
    assert client.get("/").status_code == 200


class FakeMessages:
    def __init__(self, replies):
        self.replies = list(replies)

    def parse(self, **request):
        parsed = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return SimpleNamespace(
            stop_reason="end_turn",
            stop_details=None,
            parsed_output=parsed,
            model=MODEL,
            usage=SimpleNamespace(input_tokens=10, output_tokens=10, iterations=None),
        )


def fake_expert(*replies) -> ClaudeExpert:
    return ClaudeExpert(client=SimpleNamespace(beta=SimpleNamespace(messages=FakeMessages(replies))))


def test_chart_import_and_second_opinions(client):
    daily = list(Weekday)
    chart = MedicationChart(
        lines=[
            ChartLine(
                name="Aspirin",
                strength="100 mg",
                doses=[ChartDose(time=DoseTime.BREAKFAST, quantity=1)],
                days=daily,
                in_pack=True,
                unclear=[],
            )
        ],
        warnings=[],
    )
    client.station.expert = fake_expert(chart)
    imported = client.post("/api/profiles/import?reference=MRS-X&filename=chart.pdf", content=b"%PDF fake").json()
    draft = imported["draft"]
    assert draft["matches"] == {"0": "aspirin-100"} and imported["call"]["served_model"] == MODEL
    assert sum(len(c["items"]) for c in draft["profile"]["compartments"]) == 7
    assert client.post("/api/profiles/import?reference=X&filename=chart.docx", content=b"x").status_code == 415

    profile_id = client.get("/api/profiles").json()[0]["id"]
    run_check(client, profile_id)
    disagree = CompartmentReading(pill_count=3, matches_expected=Match.NO, differences=["extra tablet"], note="")
    agree = CompartmentReading(pill_count=2, matches_expected=Match.YES, differences=[], note="")
    client.station.expert = fake_expert(disagree, agree)
    reviewed = client.post("/api/check/review", json={}).json()
    assert len(reviewed["advisories"]) == 28 and reviewed["result"]["status"] == "needsReview"
    escalated = [c for c in reviewed["result"]["compartments"] if c["status"] == "needsReview"]
    assert len(escalated) >= 1 and all(c["advisory"]["verdict"] == "disagrees" for c in escalated)


class TintEmbedder:
    """A different vector space: mean colour and its spread."""

    id = "tint-v2"

    def embed(self, crops):
        rows = [np.concatenate([c.reshape(-1, 3).mean(0), c.reshape(-1, 3).std(0)]) for c in crops]
        matrix = np.asarray(rows, dtype=np.float32)
        return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def test_brain_moves_to_a_new_embedder(tmp_path):
    station = Station.open(tmp_path, demo=True)
    with TestClient(create_app(station)) as client:
        assert client.post("/api/brain/teach", json={"medication_id": "aspirin-100"}).status_code == 200
        view = run_check(client, client.get("/api/profiles").json()[0]["id"])
        client.post("/api/check/sign-off", json={"pharmacist": "MK", "decision": "released", "reviews": reviews(view)})
        assert len(station.brain.labelling_queue) == 14

    moved = Station(Database(tmp_path), StationSettings(layout_id="station-7x4"), DemoCamera(), embedder=TintEmbedder())
    assert moved.brain.embedder_id == "tint-v2" and moved.brain.knowledge.exemplars
    assert all(e.vector.shape == (6,) for e in moved.brain.knowledge.exemplars)
    assert len(moved.brain.labelling_queue) == 14
    assert all(s.identity is None and s.vector.shape == (6,) for t in moved.brain.labelling_queue for s in t.sightings)
    moved.db.close()

    reloaded = Database(tmp_path).load_brain("tint-v2")
    assert {e.embedder_id for e in reloaded.knowledge.exemplars} == {"tint-v2"}
    assert all(e.vector.shape == (6,) for e in reloaded.knowledge.exemplars)


def fake_detector(path):
    boxes = np.array([[0.5, 0.5, 0.1, 0.1]], dtype=np.float32)
    logits = np.array([[4.0, -4.0]], dtype=np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("Constant", [], ["dets"], value=numpy_helper.from_array(boxes[None], "d")),
            helper.make_node("Constant", [], ["labels"], value=numpy_helper.from_array(logits[None], "l")),
        ],
        "fake-rfdetr",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 64, 64])],
        [
            helper.make_tensor_value_info("dets", TensorProto.FLOAT, [1, 1, 4]),
            helper.make_tensor_value_info("labels", TensorProto.FLOAT, [1, 1, 2]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.save(model, path)


def test_installed_detector_passes_the_gate_before_use(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    fake_detector(models / "detector.onnx")
    manifest = ModelManifest(
        model_id="avavision-rfdetr",
        version="1",
        stage=ModelStage.DEVELOPMENT,
        model_sha256=file_sha256(models / "detector.onnx"),
        labels={"pill": "pill", "broken": "broken"},
    )
    (models / "detector.json").write_text(manifest.model_dump_json())
    model, detector, note = load_detector(models)
    assert model.manifest.model_id == "avavision-rfdetr" and model.capability.kind == CapabilityKind.COUNT_ONLY
    assert detector.input_size == 64 and detector.classes == ["pill", "broken"] and note is None
    assert [d.label for d in detector.detect(np.zeros((80, 80, 3), np.uint8))] == ["pill"]

    tampered = manifest.model_copy(update={"model_sha256": "0" * 64})
    (models / "detector.json").write_text(tampered.model_dump_json())
    model, detector, note = load_detector(models)
    assert model.manifest.model_id == "avavision-pocket-segmenter" and detector is None and "hash" in note.lower()


def test_command_line(tmp_path, capsys):
    from avavision.__main__ import main

    samples = tmp_path / "samples.json"
    samples.write_text(
        '[{"sample_id": "a", "expected": {"m": 1}, "truth": {"m": 1}, "predicted_count": 1,'
        ' "predicted_status": "countMatched"}]'
    )
    main(["--data", str(tmp_path / "station"), "evaluate", str(samples), "holdout-1"])
    assert '"dataset_id": "holdout-1"' in capsys.readouterr().out
    main(["--data", str(tmp_path / "station"), "verify-audit"])
    assert "intact (0 entries)" in capsys.readouterr().out
    main(["--data", str(tmp_path / "station"), "markers", "--out", str(tmp_path / "m.png")])
    assert (tmp_path / "m.png").stat().st_size > 1000


def test_other_web_sites_cannot_drive_the_station(tmp_path):
    station = Station.open(tmp_path, demo=True)
    app = create_app(station, start=False, allowed_hosts=["127.0.0.1", "localhost", "testserver"])
    client = TestClient(app)
    refused = client.put("/api/settings", json={"language": "fa"}, headers={"Origin": "https://evil.example"})
    assert refused.status_code == 403
    assert (
        client.put("/api/settings", json={"language": "fa"}, headers={"Origin": "http://testserver"}).status_code == 200
    )
    assert client.get("/api/status", headers={"Host": "evil.example"}).status_code == 400
    station.db.close()


def test_size_catches_a_swapped_white_tablet_that_counting_alone_misses(client):
    import uuid

    layout_id = client.get("/api/status").json()["layout"]["id"]
    compartments = [
        {"compartment": {"row": r, "column": c}, "items": [{"medication_id": "metformin-500", "quantity": 1}]}
        for r in range(4)
        for c in range(7)
    ]
    profile = {"id": str(uuid.uuid4()), "reference": "MET-ONLY", "layout_id": layout_id, "compartments": compartments}
    assert client.put(f"/api/profiles/{profile['id']}", json=profile).status_code == 200

    # Before the brain knows metformin's size, a white tablet in its place only matches the count.
    view = run_check(client, profile["id"], fault="swapped")
    assert view["result"]["status"] == "countMatched"
    client.post("/api/check/sign-off", json={"pharmacist": "MK", "decision": "withheld", "reviews": []})

    for _ in range(2):  # two packs taught: size and colour ranges are established
        assert client.post("/api/brain/teach", json={"medication_id": "metformin-500"}).status_code == 200
    view = run_check(client, profile["id"], fault="swapped")
    flagged = [c for c in view["result"]["compartments"] if c["status"] != "countMatched"]
    assert len(flagged) == 1 and flagged[0]["status"] == "needsReview"
    assert any("size or colour" in f for f in flagged[0]["findings"])


def test_observed_pills_shown_next_to_confirmed_references(client):
    assert client.get("/api/medications/aspirin-100/reference.jpg").status_code == 404
    assert client.post("/api/brain/teach", json={"medication_id": "aspirin-100"}).status_code == 200
    assert client.get("/api/medications/aspirin-100/reference.jpg").status_code == 200  # most typical taught pill

    view = run_check(client, client.get("/api/profiles").json()[0]["id"])
    morning = next(c for c in view["result"]["compartments"] if c["row"] == 0 and c["column"] == 0)
    assert len(morning["pills"]) == 2 and all(p["size"].endswith("mm") or "mm" in p["size"] for p in morning["pills"])
    assert {e["id"]: e["reference"] for e in morning["expected"]} == {"metformin-500": False, "aspirin-100": True}
    assert client.get(f"/api/check/pills/{morning['pills'][0]['index']}.jpg").status_code == 200

    photo = cv2.imencode(".jpg", np.full((60, 60, 3), 200, np.uint8))[1].tobytes()
    assert client.put("/api/medications/metformin-500/reference", content=photo).status_code == 200
    assert client.get("/api/medications/metformin-500/reference.jpg").status_code == 200
    assert client.put("/api/medications/nope/reference", content=photo).status_code == 409
    assert client.put("/api/medications/metformin-500/reference", content=b"").status_code == 200
    assert client.get("/api/medications/metformin-500/reference.jpg").status_code == 404


def test_header_card_opens_its_profile_and_another_packs_card_stops_release(client):
    from avavision.vision.codes import read_codes
    from avavision.vision.synthetic import Scene
    from avavision.vision.synthetic import Station as SimulatedStation

    assert read_codes(SimulatedStation().render(Scene(card="WP-123"))) == ["WP-123"]
    assert read_codes(SimulatedStation().render(Scene())) == []

    profile = next(p for p in client.get("/api/profiles").json() if p["reference"] == "DEMO-001")
    assert profile["barcode"] == "DEMO-001"
    view = run_check(client, profile["id"])
    assert view["card_codes"] == ["DEMO-001"] and view["result"]["status"] == "countMatched"
    deadline = time.time() + 5
    while "DEMO-001" not in (client.get("/api/live").json() or {}).get("codes", []) and time.time() < deadline:
        time.sleep(0.1)
    assert "DEMO-001" in client.get("/api/live").json()["codes"]
    client.post("/api/check/sign-off", json={"pharmacist": "MK", "decision": "withheld", "reviews": []})

    view = run_check(client, profile["id"], fault="wrongCard")
    assert view["result"]["status"] == "needsReview"
    assert any("not this profile's card" in f for f in view["result"]["pack_findings"])
    release = {"pharmacist": "MK", "decision": "released", "reviews": reviews(view)}
    refused = client.post("/api/check/sign-off", json=release)
    assert refused.json()["kind"] == "packFindingsNotAcknowledged"
    signed = client.post("/api/check/sign-off", json=release | {"acknowledged_pack_findings": True}).json()
    record = client.get(f"/api/audit/{signed['audit_sequence']}").json()["record"]
    assert record["card_codes"] == ["OTHER-PACK-777"]


def test_checking_records_export_for_qcpp(client):
    view = run_check(client, client.get("/api/profiles").json()[0]["id"])
    client.post("/api/check/sign-off", json={"pharmacist": "MK", "decision": "released", "reviews": reviews(view)})
    rows = client.get("/api/audit/export.csv").text.strip().splitlines()
    assert rows[0].startswith("entry,signed_at,pack_reference") and len(rows) == 2
    assert ",DEMO-001,DEMO-001,station-1,countMatched,released,MK,28,0," in rows[1]
    assert len(client.get("/api/audit/export.csv?since=2999-01-01").text.strip().splitlines()) == 1
