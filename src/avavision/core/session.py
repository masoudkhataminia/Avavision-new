"""The lifecycle of checking one pack: capture → analysis → pharmacist sign-off → audit record."""

from __future__ import annotations

import uuid
from enum import StrEnum

from ..brain.brain import BrainSummary
from .advisory import AdvisoryOpinion, apply_advisories
from .audit import CheckRecord, ModelSummary
from .card import check_card
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
        self.advisories: list[AdvisoryOpinion] = []
        self.card_codes: list[str] = []
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

    def advise(self, opinions: list[AdvisoryOpinion]) -> VerificationResult:
        """Records second opinions before sign-off; disagreements escalate the result to review."""
        if self.phase != Phase.ANALYZED or self.result is None:
            raise SessionError(f"cannot take advice in phase {self.phase}")
        unknown = [o.compartment for o in opinions if not self.layout.contains(o.compartment)]
        if unknown:
            raise SessionError(f"opinions outside the layout: {unknown}")
        self.advisories.extend(opinions)
        self.result = apply_advisories(self.result, opinions)
        return self.result

    def see_card(self, codes: list[str]) -> VerificationResult:
        """Records the header-card codes seen in the analysed frames; another pack's card escalates the result."""
        if self.phase != Phase.ANALYZED or self.result is None:
            raise SessionError(f"cannot read the card in phase {self.phase}")
        self.card_codes = sorted(set(self.card_codes) | set(codes))
        self.result = check_card(self.result, self.profile, self.card_codes)
        return self.result

    def retake(self) -> None:
        if self.phase != Phase.ANALYZED:
            raise SessionError(f"cannot retake in phase {self.phase}")
        self.frames.clear()
        self.result = None
        self.advisories = []
        self.card_codes = []
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
            advisories=self.advisories,
            card_codes=self.card_codes,
        )
