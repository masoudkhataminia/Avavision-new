"""Tamper-evident audit trail: every signed-off check, hash-chained."""

from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from ..brain.brain import BrainSummary
from .engine import VerificationResult
from .gate import ActiveModel, ModelCapability, ModelStage
from .models import PackLayout, PackProfile
from .signoff import SignOff

GENESIS_HASH = "0" * 64


class ModelSummary(BaseModel):
    model_id: str
    version: str
    stage: ModelStage
    model_sha256: str
    capability: ModelCapability

    @classmethod
    def of(cls, model: ActiveModel) -> ModelSummary:
        m = model.manifest
        return cls(
            model_id=m.model_id,
            version=m.version,
            stage=m.stage,
            model_sha256=m.model_sha256,
            capability=model.capability,
        )


class CheckRecord(BaseModel):
    """Everything needed to reconstruct one completed pack check."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    app_version: str
    station_id: str
    layout: PackLayout
    profile: PackProfile
    model: ModelSummary | None
    brain: BrainSummary | None = None
    result: VerificationResult
    sign_off: SignOff
    frame_image_sha256s: list[str] = []
    #: Stored evidence images (relative paths), when image retention is enabled.
    evidence_images: list[str] = []


class AuditEntry(BaseModel):
    sequence: int
    previous_hash: str
    payload: str  # the exact JSON that was hashed
    hash: str


def entry_hash(sequence: int, previous_hash: str, payload: str) -> str:
    return hashlib.sha256(f"{sequence}\n{previous_hash}\n{payload}".encode()).hexdigest()


def make_entry(record: CheckRecord, sequence: int, previous_hash: str) -> AuditEntry:
    payload = record.model_dump_json()
    return AuditEntry(
        sequence=sequence,
        previous_hash=previous_hash,
        payload=payload,
        hash=entry_hash(sequence, previous_hash, payload),
    )


class ChainDefect(BaseModel):
    sequence: int
    defect: str  # sequenceGap | previousHashMismatch | hashMismatch


def verify_chain(entries: list[AuditEntry]) -> ChainDefect | None:
    """Returns the first defect, or ``None`` if the chain is intact."""
    previous = GENESIS_HASH
    for position, entry in enumerate(entries):
        if entry.sequence != position:
            return ChainDefect(sequence=position, defect="sequenceGap")
        if entry.previous_hash != previous:
            return ChainDefect(sequence=position, defect="previousHashMismatch")
        if entry.hash != entry_hash(entry.sequence, entry.previous_hash, entry.payload):
            return ChainDefect(sequence=position, defect="hashMismatch")
        previous = entry.hash
    return None
