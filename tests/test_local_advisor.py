from __future__ import annotations

import base64
import json
import threading
import time

import cv2
import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from avavision.core.advisory import AdvisoryVerdict
from avavision.core.models import cell
from avavision.expert.claude import CompartmentReading, ExpertIncomplete, ExpertUnavailable
from avavision.expert.local import DEFAULT_MODEL, LocalAdvisorSettings, LocalVisionAdvisor, check_local_url
from avavision.station.api import create_app
from avavision.station.demo import demo_catalog, demo_profile
from avavision.station.service import Station
from avavision.vision.synthetic import Station as SyntheticStation
from test_station import run_check


def reading(count: int | None, match: str = "yes", differences: list[str] | None = None) -> str:
    return CompartmentReading(
        pill_count=count, matches_expected=match, differences=differences or [], note=""
    ).model_dump_json()


class FakeOllama:
    """Answers /api/tags and /api/chat like Ollama does, from a list of replies (the last one repeats)."""

    def __init__(self, *replies: str, models=(DEFAULT_MODEL,), delay: float = 0.0):
        self.replies = list(replies) or [reading(2)]
        self.models = list(models)
        self.delay = delay
        self.requests: list[dict] = []
        self.lock = threading.Lock()
        self.vram = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": m} for m in self.models]})
        if request.url.path == "/api/ps":
            loaded = [{"name": m, "size": 6_000_000_000, "size_vram": self.vram} for m in self.models[:1]]
            return httpx.Response(200, json={"models": loaded if self.requests else []})
        body = json.loads(request.content)
        if body["model"] not in self.models:
            return httpx.Response(404, json={"error": f"model '{body['model']}' not found"})
        time.sleep(self.delay)
        with self.lock:
            self.requests.append(body)
            content = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return httpx.Response(
            200,
            json={
                "model": body["model"],
                "message": {"role": "assistant", "content": content},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 600,
                "eval_count": 40,
            },
        )


def advisor(fake: FakeOllama, **settings) -> LocalVisionAdvisor:
    return LocalVisionAdvisor(LocalAdvisorSettings(enabled=True, **settings), transport=httpx.MockTransport(fake))


@pytest.fixture(scope="module")
def pack():
    synthetic = SyntheticStation()
    layout = synthetic.layout()
    return layout, demo_profile(synthetic), demo_catalog()


def test_the_local_model_must_stay_on_this_computer_or_the_pharmacy_network():
    for url in ("http://127.0.0.1:11434", "http://localhost:11434/", "http://192.168.1.20:11434", "http://10.0.0.5"):
        assert check_local_url(url) == url.rstrip("/")
    for url in ("http://8.8.8.8:11434", "https://ollama.example.com", "ftp://127.0.0.1", "127.0.0.1:11434"):
        with pytest.raises(ValueError):
            check_local_url(url)
    with pytest.raises(ValueError):
        LocalAdvisorSettings(url="http://1.1.1.1:11434")


def test_a_local_opinion_uses_the_same_question_and_can_only_disagree_or_agree(pack):
    layout, profile, catalog = pack
    index = cell(0, 0)
    expected = profile.expectation(index).total_quantity
    fake = FakeOllama(reading(expected), reading(expected + 1, "yes"), reading(None, "unsure"))
    local = advisor(fake)
    image = np.full((240, 200, 3), 200, np.uint8)

    agrees = local.review_compartment(image, index, layout, profile, catalog)
    assert agrees.value.verdict == AdvisoryVerdict.AGREES and agrees.value.source == f"{DEFAULT_MODEL} (local)"
    assert agrees.call.input_tokens == 600 and not agrees.call.fallback_used
    sent = fake.requests[0]
    assert sent["model"] == DEFAULT_MODEL and sent["stream"] is False and sent["options"]["temperature"] == 0
    assert sent["format"] == CompartmentReading.model_json_schema()
    assert "Expected:" in sent["messages"][1]["content"] and layout.label(index) in sent["messages"][1]["content"]
    assert base64.b64decode(sent["messages"][1]["images"][0])[:2] == b"\xff\xd8"  # a JPEG
    # A "yes" whose count contradicts the profile is a disagreement; "unsure" changes nothing.
    assert local.review_compartment(image, index, layout, profile, catalog).value.verdict == AdvisoryVerdict.DISAGREES
    assert local.review_compartment(image, index, layout, profile, catalog).value.verdict == AdvisoryVerdict.UNSURE


def test_local_model_problems_are_reported_not_hidden(pack):
    layout, profile, catalog = pack
    image = np.full((100, 100, 3), 200, np.uint8)
    missing = advisor(FakeOllama(models=["llava:7b"]))
    assert missing.status().problem == f"the model is not installed: run  ollama pull {DEFAULT_MODEL}"
    with pytest.raises(ExpertUnavailable, match="ollama pull"):
        missing.review_compartment(image, cell(0, 0), layout, profile, catalog)
    with pytest.raises(ExpertIncomplete):
        advisor(FakeOllama("not json")).review_compartment(image, cell(0, 0), layout, profile, catalog)

    def refuse(request):
        raise httpx.ConnectError("connection refused", request=request)

    def slow(request):
        raise httpx.ReadTimeout("timed out", request=request)

    with pytest.raises(ExpertUnavailable, match="graphics card"):
        LocalVisionAdvisor(LocalAdvisorSettings(enabled=True), transport=httpx.MockTransport(slow)).review_compartment(
            image, cell(0, 0), layout, profile, catalog
        )
    down = LocalVisionAdvisor(LocalAdvisorSettings(enabled=True), transport=httpx.MockTransport(refuse))
    assert not down.status().running
    with pytest.raises(ExpertUnavailable, match="Ollama"):
        down.review_compartment(image, cell(0, 0), layout, profile, catalog)
    assert advisor(FakeOllama()).status().model_installed


@pytest.fixture
def client(tmp_path):
    station = Station.open(tmp_path, demo=True)
    with TestClient(create_app(station)) as client:
        client.station = station
        yield client


def test_local_second_opinions_escalate_on_the_station(client):
    profile_id = client.get("/api/profiles").json()[0]["id"]
    run_check(client, profile_id)
    off = client.post("/api/check/review", json={"advisor": "local"})
    assert off.status_code >= 400 and "turned off" in off.text

    # One disagreement, then agreement without a count (a count would also have to match each compartment).
    fake = FakeOllama(reading(9, "no", ["an extra tablet"]), reading(None))
    client.station.local = advisor(fake)
    reviewed = client.post("/api/check/review", json={"advisor": "local"}).json()
    assert reviewed["reviewing"] is None and reviewed["review_errors"] == []
    assert len(reviewed["advisories"]) == len(fake.requests) == 28
    assert reviewed["result"]["status"] == "needsReview"
    escalated = [c for c in reviewed["result"]["compartments"] if c["advisory"]["verdict"] == "disagrees"]
    assert len(escalated) == 1 and escalated[0]["status"] == "needsReview"
    assert escalated[0]["advisory"]["source"] == f"{DEFAULT_MODEL} (local)"
    status = client.get("/api/advisor/local").json()
    assert status["enabled"] and status["model_installed"] and status["graphics_share"] == 0  # processor only
    fake.vram = 6_000_000_000
    assert client.get("/api/advisor/local").json()["graphics_share"] == 1


def test_a_local_model_that_is_not_running_stops_the_review_at_once(client):
    profile_id = client.get("/api/profiles").json()[0]["id"]
    run_check(client, profile_id)
    client.station.local = advisor(FakeOllama(models=[]))
    refused = client.post("/api/check/review", json={"advisor": "local"})
    assert refused.status_code >= 400 and "ollama pull" in refused.text
    view = client.get("/api/check").json()
    assert view["reviewing"] is None and view["advisories"] == []


def test_automatic_local_review_runs_after_capture_and_ends_with_a_retake(client):
    profile_id = client.get("/api/profiles").json()[0]["id"]
    client.put("/api/settings", json={"local_advisor": {"enabled": True, "automatic": True}})
    fake = FakeOllama(reading(9, "no"), reading(None), delay=0.02)
    client.station.local = advisor(fake, automatic=True)
    view = run_check(client, profile_id)
    assert view["reviewing"]["total"] == 28
    deadline = time.time() + 20
    while client.get("/api/check").json()["reviewing"] and time.time() < deadline:
        time.sleep(0.05)
    done = client.get("/api/check").json()
    assert len(done["advisories"]) == 28 and done["result"]["status"] == "needsReview"

    # A retake ends a running review; its late answers never reach the new photos.
    client.post("/api/check/retake")
    client.station.local = advisor(FakeOllama(reading(9, "no"), delay=0.2), automatic=True)
    captured = client.post("/api/check/capture").json()
    assert captured["reviewing"] is not None
    client.post("/api/check/retake")
    assert client.get("/api/check").json()["reviewing"] is None
    time.sleep(0.5)
    retaken = client.get("/api/check").json()
    assert retaken["advisories"] == [] and retaken["result"] is None


def test_a_running_review_can_be_stopped(client):
    profile_id = client.get("/api/profiles").json()[0]["id"]
    client.station.settings.local_advisor = LocalAdvisorSettings(enabled=True, automatic=True)
    client.station.local = advisor(FakeOllama(reading(None), delay=0.1), automatic=True)
    run_check(client, profile_id)
    stopped = client.post("/api/check/review/stop").json()
    assert stopped["reviewing"]["stop"] is True
    deadline = time.time() + 5
    while client.get("/api/check").json()["reviewing"] and time.time() < deadline:
        time.sleep(0.05)
    assert len(client.get("/api/check").json()["advisories"]) < 28


def test_settings_refuse_a_model_on_the_internet(client):
    refused = client.put("/api/settings", json={"local_advisor": {"enabled": True, "url": "http://8.8.8.8:11434"}})
    assert refused.status_code >= 400 and "pharmacy network" in refused.text
    saved = client.put("/api/settings", json={"local_advisor": {"enabled": True, "url": "http://192.168.1.20:11434"}})
    assert saved.status_code == 200 and client.station.local is not None
    assert client.get("/api/status").json()["local_advisor"] == {
        "enabled": True,
        "model": DEFAULT_MODEL,
        "automatic": False,
    }


class OneDisagreement:
    """A reviewer that disagrees with one compartment and agrees with the rest."""

    def __init__(self, index):
        self.index = index

    def review_compartment(self, image, index, layout, profile, catalog):
        from avavision.core.advisory import AdvisoryOpinion
        from avavision.expert.claude import ExpertAnswer, ExpertCall

        verdict = AdvisoryVerdict.DISAGREES if index == self.index else AdvisoryVerdict.AGREES
        call = ExpertCall(
            requested_model="x", served_model="x", fallback_used=False, input_tokens=0, output_tokens=0, seconds=0
        )
        return ExpertAnswer(value=AdvisoryOpinion(compartment=index, verdict=verdict, source="x"), call=call)


def test_advisors_are_scored_on_compartments_the_pharmacist_corrected(client):
    from avavision.core.audit import CheckRecord
    from avavision.expert.compare import compare_advisors, report, samples_from_records

    profile_id = client.get("/api/profiles").json()[0]["id"]
    view = run_check(client, profile_id, fault="extra")
    wrong = next(c for c in view["result"]["compartments"] if c["status"] == "mismatch")
    reviews = [
        {
            "compartment": {"row": c["row"], "column": c["column"]},
            "outcome": "corrected" if c is wrong else "confirmedCorrect",
        }
        for c in view["result"]["compartments"]
    ]
    signed = client.post("/api/check/sign-off", json={"pharmacist": "MK", "decision": "released", "reviews": reviews})
    assert signed.status_code == 200, signed.text

    db = client.station.db
    records = [CheckRecord.model_validate_json(e.payload) for e in db.audit_entries()]
    samples = samples_from_records(records, lambda name: cv2.imread(str(db.evidence / name)))
    assert len(samples) == 28 and [s.label for s in samples if not s.correct] == [wrong["label"]]
    assert all(s.image.size for s in samples)
    target = next(s.index for s in samples if not s.correct)
    scores, rows = compare_advisors(
        {"catches it": OneDisagreement(target), "misses it": OneDisagreement(cell(9, 9))}, samples, db.catalog()
    )
    catches, misses = scores
    assert (catches.caught, catches.wrong, catches.false_alarms, catches.correct) == (1, 1, 0, 27)
    assert (misses.caught, misses.false_alarms) == (0, 0) and len(rows) == 56
    assert "1 of 1" in report(scores) and "0 of 27" in report(scores)
