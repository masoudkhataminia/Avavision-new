"""Camera capture on a background thread. Works with USB webcams, document cameras and capture cards.

On Windows, Media Foundation is used (best for high-resolution UVC cameras); DirectShow is the fallback.
"""

from __future__ import annotations

import platform
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CameraSettings:
    index: int = 0
    width: int = 3840
    height: int = 2160
    fps: int = 30
    autofocus: bool = True
    #: Manual exposure (camera units, e.g. -6) for a fixed station light; ``None`` keeps auto exposure.
    exposure: float | None = None


def list_cameras(max_index: int = 6) -> list[dict]:
    found = []
    for index in range(max_index):
        capture = _open(index)
        if capture is not None and capture.isOpened():
            found.append(
                {
                    "index": index,
                    "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                }
            )
            capture.release()
    return found


def _open(index: int) -> cv2.VideoCapture | None:
    backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW] if platform.system() == "Windows" else [cv2.CAP_ANY]
    for backend in backends:
        capture = cv2.VideoCapture(index, backend)
        if capture.isOpened():
            return capture
        capture.release()
    return None


class FrameSource:
    """Anything that yields BGR frames: a real camera or the simulated station."""

    def read(self) -> np.ndarray | None:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:
        pass


class CameraSource(FrameSource):
    def __init__(self, settings: CameraSettings):
        capture = _open(settings.index)
        if capture is None:
            raise RuntimeError(f"camera {settings.index} could not be opened")
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, settings.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.height)
        capture.set(cv2.CAP_PROP_FPS, settings.fps)
        capture.set(cv2.CAP_PROP_AUTOFOCUS, 1 if settings.autofocus else 0)
        if settings.exposure is not None:
            capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            capture.set(cv2.CAP_PROP_EXPOSURE, settings.exposure)
        self.capture = capture

    @property
    def resolution(self) -> tuple[int, int]:
        return int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read(self) -> np.ndarray | None:
        ok, frame = self.capture.read()
        return frame if ok else None

    def close(self) -> None:
        self.capture.release()


class StillSource(FrameSource):
    """Replays fixed images (demo station, photos from disk, tests)."""

    def __init__(self, images: list[np.ndarray], interval: float = 0.05):
        self.images = images
        self.interval = interval
        self.position = 0

    def read(self) -> np.ndarray | None:
        if not self.images:
            return None
        time.sleep(self.interval)
        image = self.images[self.position % len(self.images)]
        self.position += 1
        return image.copy()


class LiveFeed:
    """Keeps only the newest frame so analysis never lags behind the camera."""

    def __init__(self, source: FrameSource, on_frame: Callable[[np.ndarray, float], None] | None = None):
        self.source = source
        self.on_frame = on_frame
        self._latest: tuple[np.ndarray, float] | None = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self.frames_read = 0

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, name="avavision-camera", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2)
        self.source.close()

    def latest(self) -> tuple[np.ndarray, float] | None:
        with self._lock:
            return self._latest

    def _run(self) -> None:
        while self._running:
            frame = self.source.read()
            if frame is None:
                time.sleep(0.01)
                continue
            stamp = time.time()
            with self._lock:
                self._latest = (frame, stamp)
            self.frames_read += 1
            if self.on_frame is not None:
                self.on_frame(frame, stamp)
