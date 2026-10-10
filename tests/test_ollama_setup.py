from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from avavision.expert.local import DEFAULT_MODEL
from avavision.station.api import create_app
from avavision.station.ollama_setup import WINGET, OllamaSetup, SetupStage, System
from avavision.station.service import Station


class FakeServer:
    """Ollama that answers once it has been started, and streams a two-layer download."""

    def __init__(self, running: bool = True, pull_error: str | None = None):
        self.running = running
        self.pull_error = pull_error
        self.pulled: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if not self.running:
            raise httpx.ConnectError("refused", request=request)
        path = request.url.path
        if path == "/api/version":
            return httpx.Response(200, json={"version": "0.40.2"})
        if path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": m} for m in self.pulled]})
        if path == "/api/ps":
            return httpx.Response(200, json={"models": []})
        if path == "/api/pull":
            model = json.loads(request.content)["model"]
            events = [{"status": "pulling manifest"}]
            if self.pull_error:
                events.append({"error": self.pull_error})
            else:
                for completed in (0, 3_000, 6_000):
                    events.append({"status": "pulling a", "digest": "a", "total": 6_000, "completed": completed})
                events += [
                    {"status": "pulling b", "digest": "b", "total": 1_000, "completed": 1_000},
                    {"status": "verifying sha256 digest"},
                    {"status": "success"},
                ]
                self.pulled.append(model)
            return httpx.Response(200, content="\n".join(json.dumps(e) for e in events).encode())
        return httpx.Response(404)


class FakeComputer(System):
    """A Windows computer without Ollama whose winget installs it."""

    def __init__(self, server: FakeServer, tmp: Path, platform: str = "win32", winget: bool = True):
        self.exe = tmp / "Programs" / "Ollama" / "ollama.exe"
        self.commands: list[list[str]] = []
        server_ref = server
        computer = self

        def which(name):
            if name == "winget":
                return "winget.exe" if winget else None
            return None

        def run(command, **_):
            computer.commands.append(command)
            computer.exe.parent.mkdir(parents=True, exist_ok=True)
            computer.exe.write_bytes(b"")
            return type("Done", (), {"stdout": "Successfully installed", "stderr": ""})()

        def popen(command, **_):
            computer.commands.append(command)
            server_ref.running = True

        super().__init__(
            platform=platform,
            which=which,
            run=run,
            popen=popen,
            sleep=lambda _: None,
            environ={"LOCALAPPDATA": str(tmp)},
        )


def setup_with(server: FakeServer, computer: System | None = None, url: str = "http://127.0.0.1:11434"):
    done = []
    setup = OllamaSetup(
        url, DEFAULT_MODEL, on_done=lambda: done.append(True), transport=httpx.MockTransport(server), system=computer
    )
    setup.run()
    return setup, done


def test_an_installed_ollama_only_downloads_the_model_with_progress():
    server = FakeServer()
    setup, done = setup_with(server, System(which=lambda _: None))
    assert setup.state.stage == SetupStage.DONE and done == [True]
    assert server.pulled == [DEFAULT_MODEL]


def test_download_progress_adds_up_every_layer():
    seen = []

    class Watched(OllamaSetup):
        def _report(self, state):
            seen.append(state)
            super()._report(state)

    Watched("http://127.0.0.1:11434", DEFAULT_MODEL, lambda: None, transport=httpx.MockTransport(FakeServer())).run()
    downloads = [s for s in seen if s.stage == SetupStage.DOWNLOADING and s.total]
    assert (downloads[-1].completed, downloads[-1].total) == (7_000, 7_000)
    assert [s.completed for s in downloads][:3] == [0, 3_000, 6_000]
    assert seen[-1].stage == SetupStage.DONE


def test_a_windows_computer_without_ollama_installs_it_with_winget_and_starts_it(tmp_path):
    server = FakeServer(running=False)
    computer = FakeComputer(server, tmp_path)
    setup, done = setup_with(server, computer)
    assert setup.state.stage == SetupStage.DONE and done == [True]
    assert computer.commands == [WINGET, [str(computer.exe), "serve"]]


def test_setup_says_what_to_do_when_it_cannot_install(tmp_path):
    elsewhere = FakeServer(running=False)
    setup, done = setup_with(elsewhere, FakeComputer(elsewhere, tmp_path, platform="linux"))
    assert setup.state.stage == SetupStage.FAILED and "ollama.com" in setup.state.error and done == []
    remote = FakeServer(running=False)
    setup, _ = setup_with(remote, FakeComputer(remote, tmp_path), url="http://192.168.1.20:11434")
    assert setup.state.stage == SetupStage.FAILED and "on that computer" in setup.state.error
    broken = FakeServer(pull_error="no space left on device")
    setup, done = setup_with(broken, System(which=lambda _: None))
    assert setup.state.stage == SetupStage.FAILED and "no space left" in setup.state.error and done == []


def test_one_button_in_settings_sets_up_and_turns_on_the_offline_model(tmp_path):
    station = Station.open(tmp_path, demo=True)
    server = FakeServer(running=False)
    station.ollama_transport = httpx.MockTransport(server)
    station.ollama_system = FakeComputer(server, tmp_path / "home")
    with TestClient(create_app(station)) as client:
        before = client.get("/api/advisor/local").json()
        assert not before["enabled"] and not before["running"] and before["setup"] is None
        client.post("/api/advisor/local/setup")
        deadline = time.time() + 10
        while client.get("/api/advisor/local").json()["setup"]["stage"] not in ("done", "failed"):
            assert time.time() < deadline
            time.sleep(0.05)
        after = client.get("/api/advisor/local").json()
        assert after["setup"]["stage"] == "done" and after["enabled"] and after["model_installed"]
        assert client.get("/api/settings").json()["local_advisor"]["enabled"] is True
        assert client.get("/api/status").json()["local_advisor"]["enabled"] is True
