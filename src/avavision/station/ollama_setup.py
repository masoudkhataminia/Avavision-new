"""One-click setup of the offline second-opinion model: install Ollama, start it, download the model.

The pharmacist presses one button in Settings instead of typing commands. On Windows Ollama is installed with
winget (its package ``Ollama.Ollama``); elsewhere it must already be installed. Only a model on this computer can
be set up this way; for one on another computer of the pharmacy network, Ollama is installed there by hand.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from urllib.parse import urlsplit

import httpx

WINGET = [
    "winget",
    "install",
    "--id",
    "Ollama.Ollama",
    "-e",
    "--silent",
    "--accept-package-agreements",
    "--accept-source-agreements",
]


class SetupStage(StrEnum):
    IDLE = "idle"
    INSTALLING = "installing"  # winget is installing Ollama
    STARTING = "starting"  # waiting for Ollama to answer
    DOWNLOADING = "downloading"  # the model is being downloaded
    DONE = "done"
    FAILED = "failed"


@dataclass
class SetupState:
    stage: SetupStage = SetupStage.IDLE
    detail: str = ""
    completed: int = 0
    total: int = 0
    error: str | None = None

    def view(self) -> dict:
        return asdict(self)


@dataclass
class System:
    """What setup needs from the computer, replaceable in tests."""

    platform: str = sys.platform
    which: Callable[[str], str | None] = shutil.which
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run
    popen: Callable[..., object] = subprocess.Popen
    sleep: Callable[[float], None] = time.sleep
    environ: dict = field(default_factory=lambda: dict(os.environ))


def is_this_computer(url: str) -> bool:
    return urlsplit(url).hostname in ("127.0.0.1", "localhost", "::1")


class OllamaSetup:
    def __init__(
        self,
        url: str,
        model: str,
        on_done: Callable[[], None],
        transport: httpx.BaseTransport | None = None,
        system: System | None = None,
        start_timeout: float = 90.0,
    ):
        self.url = url.rstrip("/")
        self.model = model
        self.on_done = on_done
        self.system = system or System()
        self.start_timeout = start_timeout
        self.state = SetupState()
        self._client = httpx.Client(transport=transport, trust_env=False, timeout=httpx.Timeout(10.0, read=300.0))
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._report(SetupState(stage=SetupStage.STARTING, detail="Looking for Ollama"))
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()

    def run(self) -> None:
        try:
            if not self._answering():
                if not is_this_computer(self.url):
                    raise SetupError("Ollama is not answering on that computer: install it there from ollama.com")
                self._start_server(self._executable() or self._install())
            self._download()
            # Turn the advisor on before saying so: the interface reloads the settings as soon as it sees "done".
            self.on_done()
            self._report(SetupState(stage=SetupStage.DONE, detail=f"{self.model} is ready"))
        except SetupError as error:
            self._report(SetupState(stage=SetupStage.FAILED, error=str(error)))
        except (httpx.HTTPError, OSError, subprocess.SubprocessError) as error:
            self._report(SetupState(stage=SetupStage.FAILED, error=f"setup failed: {error}"))

    def _report(self, state: SetupState) -> None:
        self.state = state

    # ------------------------------------------------------------------ steps

    def _answering(self) -> bool:
        try:
            return self._client.get(f"{self.url}/api/version", timeout=3.0).status_code == 200
        except httpx.HTTPError:
            return False

    def _executable(self) -> str | None:
        found = self.system.which("ollama")
        if found:
            return found
        if self.system.platform == "win32":
            local = self.system.environ.get("LOCALAPPDATA", "")
            candidate = Path(local) / "Programs" / "Ollama" / "ollama.exe"
            if local and candidate.is_file():
                return str(candidate)
        return None

    def _install(self) -> str:
        if self.system.platform != "win32" or not self.system.which("winget"):
            raise SetupError("Ollama is not installed: install it from ollama.com, then press the button again")
        self._report(SetupState(stage=SetupStage.INSTALLING, detail="Installing Ollama (a few minutes)"))
        done = self.system.run(WINGET, capture_output=True, text=True, timeout=1800)
        executable = self._executable()
        if executable is None:
            output = (getattr(done, "stdout", "") or "") + (getattr(done, "stderr", "") or "")
            raise SetupError(f"installing Ollama did not work: {output.strip()[-300:] or 'no output'}")
        return executable

    def _start_server(self, executable: str) -> None:
        self._report(SetupState(stage=SetupStage.STARTING, detail="Starting Ollama"))
        flags = 0
        if self.system.platform == "win32":
            flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.system.popen(
            [executable, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=flags,
        )
        deadline = time.monotonic() + self.start_timeout
        while not self._answering():
            if time.monotonic() > deadline:
                raise SetupError("Ollama was installed but does not start: restart the computer and try again")
            self.system.sleep(1.0)

    def _download(self) -> None:
        self._report(SetupState(stage=SetupStage.DOWNLOADING, detail=f"Downloading {self.model}"))
        layers: dict[str, tuple[int, int]] = {}
        with self._client.stream("POST", f"{self.url}/api/pull", json={"model": self.model, "stream": True}) as reply:
            if reply.status_code >= 400:
                reply.read()
                raise SetupError(f"the download was refused ({reply.status_code}): {reply.text[:200]}")
            for line in reply.iter_lines():
                if not line.strip():
                    continue
                event = json.loads(line)
                if event.get("error"):
                    raise SetupError(f"the download failed: {event['error']}")
                if event.get("digest") and event.get("total"):
                    layers[event["digest"]] = (int(event.get("completed") or 0), int(event["total"]))
                self._report(
                    SetupState(
                        stage=SetupStage.DOWNLOADING,
                        detail=str(event.get("status", "")),
                        completed=sum(c for c, _ in layers.values()),
                        total=sum(t for _, t in layers.values()),
                    )
                )
                if event.get("status") == "success":
                    return
        raise SetupError("the download ended before it was complete")


class SetupError(RuntimeError):
    pass
