"""Station storage: one SQLite database plus an image folder.

SQLite keeps everything in a single file that survives crashes (WAL journal) and handles concurrent reads
from the UI while the camera pipeline writes.
"""

from __future__ import annotations

import base64
import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

import numpy as np

from ..brain.brain import Brain
from ..brain.calibration import CalibrationReport
from ..brain.identity import IdentityEvidence, IdentityPolicy
from ..brain.knowledge import Exemplar, ExemplarSource
from ..brain.learning import LabellingTask, PillSighting
from ..brain.trust import TrustLedger, TrustPolicy
from ..core.audit import GENESIS_HASH, AuditEntry, ChainDefect, CheckRecord, make_entry, verify_chain
from ..core.models import WEEKLY_7X4, Catalog, CompartmentIndex, PackLayout, PackProfile

SCHEMA = """
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS profiles (id TEXT PRIMARY KEY, reference TEXT NOT NULL, json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS audit (
  sequence INTEGER PRIMARY KEY, previous_hash TEXT NOT NULL, payload TEXT NOT NULL, hash TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS exemplars (
  id TEXT PRIMARY KEY, medication_id TEXT NOT NULL, embedder_id TEXT NOT NULL, source TEXT NOT NULL,
  group_id TEXT NOT NULL, created_at TEXT NOT NULL, crop_file TEXT, vector BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS exemplars_medication ON exemplars(medication_id);
CREATE TABLE IF NOT EXISTS labelling_tasks (id TEXT PRIMARY KEY, json TEXT NOT NULL);
"""


class AuditChainBroken(RuntimeError):
    def __init__(self, defect: ChainDefect):
        super().__init__(f"audit chain broken at entry {defect.sequence}: {defect.defect}")
        self.defect = defect


def _vector_to_text(v: np.ndarray | None) -> str | None:
    return None if v is None else base64.b64encode(np.asarray(v, dtype="<f4").tobytes()).decode()


def _vector_from_text(text: str | None) -> np.ndarray | None:
    return None if text is None else np.frombuffer(base64.b64decode(text), dtype="<f4").copy()


class Database:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.images = self.root / "images"
        self.crops = self.images / "crops"
        self.evidence = self.images / "evidence"
        for folder in (self.crops, self.evidence):
            folder.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.root / "avavision.sqlite3", check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.executescript(SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # ------------------------------------------------------------------ key/value

    def _get(self, key: str) -> str | None:
        row = self._db.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def _put(self, key: str, value: str) -> None:
        self._db.execute(
            "INSERT INTO kv(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    # ------------------------------------------------------------------ catalog, layouts, profiles

    def catalog(self) -> Catalog:
        with self._lock:
            text = self._get("catalog")
        return Catalog.model_validate_json(text) if text else Catalog()

    def save_catalog(self, catalog: Catalog) -> None:
        with self._lock, self._db:
            self._put("catalog", catalog.model_dump_json())

    def layouts(self) -> list[PackLayout]:
        """Built-in layouts, replaced by any saved (calibrated) version with the same id."""
        with self._lock:
            text = self._get("layouts")
        saved = {layout.id: layout for layout in (PackLayout.model_validate(d) for d in json.loads(text or "[]"))}
        return [saved.get(WEEKLY_7X4.id, WEEKLY_7X4), *(v for k, v in saved.items() if k != WEEKLY_7X4.id)]

    def save_layout(self, layout: PackLayout) -> None:
        layouts = {lay.id: lay for lay in self.layouts()}
        layouts[layout.id] = layout
        with self._lock, self._db:
            self._put("layouts", json.dumps([lay.model_dump(mode="json") for lay in layouts.values()]))

    def profiles(self) -> list[PackProfile]:
        with self._lock:
            rows = self._db.execute("SELECT json FROM profiles ORDER BY reference").fetchall()
        return [PackProfile.model_validate_json(r[0]) for r in rows]

    def profile(self, profile_id: str) -> PackProfile | None:
        with self._lock:
            row = self._db.execute("SELECT json FROM profiles WHERE id = ?", (profile_id,)).fetchone()
        return PackProfile.model_validate_json(row[0]) if row else None

    def save_profile(self, profile: PackProfile) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO profiles(id, reference, json) VALUES(?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET reference = excluded.reference, json = excluded.json",
                (str(profile.id), profile.reference, profile.model_dump_json()),
            )

    def delete_profile(self, profile_id: str) -> None:
        with self._lock, self._db:
            self._db.execute("DELETE FROM profiles WHERE id = ?", (profile_id,))

    def setting(self, key: str, default=None):
        with self._lock:
            text = self._get(f"setting:{key}")
        return json.loads(text) if text is not None else default

    def save_setting(self, key: str, value) -> None:
        with self._lock, self._db:
            self._put(f"setting:{key}", json.dumps(value))

    # ------------------------------------------------------------------ audit

    def audit_entries(self) -> list[AuditEntry]:
        with self._lock:
            rows = self._db.execute(
                "SELECT sequence, previous_hash, payload, hash FROM audit ORDER BY sequence"
            ).fetchall()
        return [AuditEntry(sequence=r[0], previous_hash=r[1], payload=r[2], hash=r[3]) for r in rows]

    def verify_audit(self) -> ChainDefect | None:
        return verify_chain(self.audit_entries())

    def append_audit(self, record: CheckRecord) -> AuditEntry:
        """Refuses to extend a chain that no longer verifies."""
        with self._lock:
            entries = self.audit_entries()
            defect = verify_chain(entries)
            if defect is not None:
                raise AuditChainBroken(defect)
            entry = make_entry(record, len(entries), entries[-1].hash if entries else GENESIS_HASH)
            with self._db:
                self._db.execute(
                    "INSERT INTO audit(sequence, previous_hash, payload, hash) VALUES(?, ?, ?, ?)",
                    (entry.sequence, entry.previous_hash, entry.payload, entry.hash),
                )
            return entry

    def audit_records(self) -> list[CheckRecord]:
        return [CheckRecord.model_validate_json(e.payload) for e in self.audit_entries()]

    # ------------------------------------------------------------------ brain

    def load_brain(self, embedder_id: str) -> Brain | None:
        """Returns the saved brain (possibly for another embedder), or ``None`` if none was saved."""
        with self._lock:
            meta_text = self._get("brain")
            if meta_text is None:
                return None
            meta = json.loads(meta_text)
            rows = self._db.execute(
                "SELECT id, medication_id, embedder_id, source, group_id, created_at, crop_file, vector FROM exemplars"
            ).fetchall()
            tasks = [json.loads(r[0]) for r in self._db.execute("SELECT json FROM labelling_tasks").fetchall()]
        brain = Brain(meta["embedder_id"], TrustPolicy.model_validate(meta["trust_policy"]))
        for r in rows:
            brain.knowledge.add(
                Exemplar(
                    id=r[0],
                    medication_id=r[1],
                    embedder_id=r[2],
                    source=ExemplarSource(r[3]),
                    group_id=r[4],
                    created_at=datetime.fromisoformat(r[5]),
                    crop_file=r[6],
                    vector=np.frombuffer(r[7], dtype="<f4").copy(),
                )
            )
        brain.identity_policy = IdentityPolicy.model_validate(meta["identity_policy"])
        brain.ledger = TrustLedger.model_validate(meta["ledger"])
        brain.last_calibration = (
            CalibrationReport.model_validate(meta["last_calibration"]) if meta.get("last_calibration") else None
        )
        brain.exemplars_at_calibration = meta.get("exemplars_at_calibration", 0)
        brain.labelling_queue = [self._task_from_dict(t) for t in tasks]
        return brain

    def save_brain(self, brain: Brain) -> None:
        meta = {
            "embedder_id": brain.embedder_id,
            "identity_policy": brain.identity_policy.model_dump(mode="json"),
            "trust_policy": brain.trust_policy.model_dump(mode="json"),
            "ledger": brain.ledger.model_dump(mode="json"),
            "last_calibration": brain.last_calibration.model_dump(mode="json") if brain.last_calibration else None,
            "exemplars_at_calibration": brain.exemplars_at_calibration,
        }
        current = {e.id: e for e in brain.knowledge.exemplars}
        with self._lock, self._db:
            stored = {r[0] for r in self._db.execute("SELECT id FROM exemplars").fetchall()}
            removed = stored - current.keys()
            self._db.executemany("DELETE FROM exemplars WHERE id = ?", [(i,) for i in removed])
            self._db.executemany(
                "INSERT INTO exemplars(id, medication_id, embedder_id, source, group_id, created_at, crop_file, "
                "vector) VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        e.id,
                        e.medication_id,
                        e.embedder_id,
                        e.source.value,
                        e.group_id,
                        e.created_at.isoformat(),
                        e.crop_file,
                        np.asarray(e.vector, dtype="<f4").tobytes(),
                    )
                    for i, e in current.items()
                    if i not in stored
                ],
            )
            self._db.execute("DELETE FROM labelling_tasks")
            self._db.executemany(
                "INSERT INTO labelling_tasks(id, json) VALUES(?, ?)",
                [(t.id, json.dumps(self._task_to_dict(t))) for t in brain.labelling_queue],
            )
            self._put("brain", json.dumps(meta))

    def referenced_crops(self, brain: Brain) -> set[str]:
        return {e.crop_file for e in brain.knowledge.exemplars if e.crop_file} | {
            s.crop_file for t in brain.labelling_queue for s in t.sightings if s.crop_file
        }

    def prune_crops(self, keep: set[str]) -> None:
        for path in self.crops.iterdir():
            if path.name not in keep:
                path.unlink(missing_ok=True)

    @staticmethod
    def _task_to_dict(task: LabellingTask) -> dict:
        return {
            "id": task.id,
            "check_id": task.check_id,
            "compartment_label": task.compartment_label,
            "expected": task.expected,
            "priority": task.priority,
            "created_at": task.created_at.isoformat(),
            "sightings": [
                {
                    "id": s.id,
                    "compartment": s.compartment.model_dump(),
                    "vector": _vector_to_text(s.vector),
                    "embedder_id": s.embedder_id,
                    "crop_file": s.crop_file,
                    "identity": s.identity.model_dump(mode="json") if s.identity else None,
                }
                for s in task.sightings
            ],
        }

    @staticmethod
    def _task_from_dict(d: dict) -> LabellingTask:
        return LabellingTask(
            id=d["id"],
            check_id=d["check_id"],
            compartment_label=d["compartment_label"],
            expected=d["expected"],
            priority=d["priority"],
            created_at=datetime.fromisoformat(d["created_at"]),
            sightings=[
                PillSighting(
                    id=s["id"],
                    compartment=CompartmentIndex.model_validate(s["compartment"]),
                    vector=_vector_from_text(s["vector"]),
                    embedder_id=s["embedder_id"],
                    crop_file=s["crop_file"],
                    identity=IdentityEvidence.model_validate(s["identity"]) if s["identity"] else None,
                )
                for s in d["sightings"]
            ],
        )
