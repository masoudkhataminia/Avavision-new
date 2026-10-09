"""The lifecycle of checking one pack: capture → analysis → pharmacist sign-off → audit record."""

from __future__ import annotations

import uuid
from enum import StrEnum

from ..brain.brain import BrainSummary
from .audit import CheckRecord, ModelSummary
from .engine import VerificationEngine, VerificationResult
from .models import Catalog, PackLayout, PackProfile
from .observation import FrameObservation
from .signoff import SignOff, validate_sign_off


class Phase(StrEnum):
    CAPTURING = "capturing"
    ANALYZED = "analyzed"
    COMPLETED = "completed"


class SessionError(ValueError):
    pass


class CheckSession:
    MAXIMUM_RETAINED_FRAMES = 12

    def __init__(self, layout: PackLayout, profile: PackProfile, catalog: Catalog):
        layout_errors = layout.validation_errors()
        if layout_errors:
            raise SessionError(f"invalid layout: {layout_errors}")
        issues = profile.issues(layout, catalog)
        if issues:
            raise SessionError(f"invalid profile: {[i.kind.value for i in issues]}")
        self.id = uuid.uuid4()
        self.layout = layout
        self.profile = profile
        self.phase = Phase.CAPTURING
        self.frames: list[FrameObservation] = []
        self.result: VerificationResult | None = None
        self.sign_off: SignOff | None = None

    @property
    def usable_frame_count(self) -> int:
        return sum(1 for f in self.frames if f.is_usable)

    def record(self, frame: FrameObservation) -> None:
        if self.phase != Phase.CAPTURING:
            raise SessionError(f"cannot record in phase {self.phase}")
        self.frames.append(frame)
        del self.frames[: max(0, len(self.frames) - self.MAXIMUM_RETAINED_FRAMES)]

    def analyze(self, engine: VerificationEngine) -> VerificationResult:
        if self.phase != Phase.CAPTURING:
            raise SessionError(f"cannot analyze in phase {self.phase}")
        self.result = engine.evaluate(self.profile, self.frames)
        self.phase = Phase.ANALYZED
        return self.result

    def retake(self) -> None:
        if self.phase != Phase.ANALYZED:
            raise SessionError(f"cannot retake in phase {self.phase}")
        self.frames.clear()
        self.result = None
        self.phase = Phase.CAPTURING

    def complete(
        self,
        sign_off: SignOff,
        app_version: str,
        station_id: str,
        model: ModelSummary | None,
        brain: BrainSummary | None = None,
        evidence_images: list[str] | None = None,
    ) -> CheckRecord:
        if self.phase != Phase.ANALYZED or self.result is None:
            raise SessionError(f"cannot complete in phase {self.phase}")
        validate_sign_off(sign_off, self.result)
        self.sign_off = sign_off
        self.phase = Phase.COMPLETED
        return CheckRecord(
            app_version=app_version,
            station_id=station_id,
            layout=self.layout,
            profile=self.profile,
            model=model,
            brain=brain,
            result=self.result,
            sign_off=sign_off,
            frame_image_sha256s=[f.image_sha256 for f in self.frames if f.image_sha256],
            evidence_images=evidence_images or [],
        )
