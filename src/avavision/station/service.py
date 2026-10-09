"""The station: one camera, one pack check at a time, and everything a check touches.

Flow of a check: ``start_check`` → ``capture`` (frames → detection → identification once per check,
propagated to the other frames → decision engine) → optional ``review`` / ``explain`` by the expert →
``sign_off`` (pharmacist) → audit entry → the brain learns from the compartments the pharmacist confirmed.
"""

from __future__ import annotations

import os
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel, Field

from .. import __version__
from ..brain.brain import Brain, LearningSummary
from ..brain.learning import PillSighting, plan_learning
from ..core.advisory import AdvisoryOpinion
from ..core.audit import CheckRecord, ModelSummary
from ..core.engine import BrainContext, CompartmentStatus, DecisionPolicy, VerificationEngine
from ..core.gate import (
    ActiveModel,
    CapabilityKind,
    EmbedderManifest,
    ModelManifest,
    builtin_pocket_segmenter,
    evaluate_gate,
)
from ..core.models import WEEKLY_7X4, Catalog, CompartmentIndex, ExpectedItem, PackLayout, PackProfile
from ..core.observation import pill_compartments
from ..core.peers import peer_groups, unlike_peers
from ..core.physical import PhysicalFeatures, PhysicalRange
from ..core.session import CheckSession, Phase
from ..core.signoff import SignOff
from ..expert.charts import MedicationChart, ProfileDraft, chart_to_profile
from ..expert.claude import ClaudeExpert, ExpertCall, ExpertError, ExpertSettings, Explanation
from ..expert.describe import describe_finding, describe_pack_finding
from ..storage.database import Database
from ..vision.camera import CameraSettings, CameraSource, FrameSource, LiveFeed
from ..vision.codes import code_image, read_codes
from ..vision.detector import OnnxDetector
from ..vision.embedder import ClassicEmbedder, OnnxEmbedder
from ..vision.pack_finder import FinderMode
from ..vision.peers import compartment_signatures
from ..vision.pipeline import AnalyzedFrame, FrameAnalyzer, compartment_crop, propagate_identities
from ..vision.runtime import available_providers, file_sha256
from ..vision.segmentation import rectify
from .credentials import load_api_key
from .demo import DemoCamera, DemoFault, demo_catalog, demo_profile
from .phone import PhoneCamera, PhoneLink, new_pairing_key

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$")


class StationError(RuntimeError):
    """A request the station cannot carry out in its current state."""


class CameraKind(StrEnum):
    USB = "usb"  # a camera attached to this computer (OpenCV)
    IPHONE = "iphone"  # an iPhone's Safari page over the local Wi-Fi; see station.phone


class StationSettings(BaseModel):
    station_id: str = "station-1"
    layout_id: str = WEEKLY_7X4.id
    finder_mode: FinderMode = FinderMode.MARKERS
    camera_source: CameraKind = CameraKind.USB
    camera_index: int = 0
    camera_width: int = 3840
    camera_height: int = 2160
    #: Manual exposure for a fixed station light; ``None`` keeps auto exposure.
    camera_exposure: float | None = None
    #: Local-network ports of the iPhone camera page (HTTPS) and its one-time setup page (HTTP).
    phone_port: int = 8766
    phone_setup_port: int = 8767
    keep_evidence_images: bool = True
    expert_enabled: bool = True
    #: Language of the expert's explanations ("en" or "fa").
    language: str = "en"
    expert: ExpertSettings = Field(default_factory=ExpertSettings)


@dataclass
class LiveStatus:
    at: float
    usable: bool
    quality_issues: list[str]
    registration_issues: list[str]
    quad: list[tuple[float, float]] | None
    sharpness: float | None
    locate_ms: float
    codes: list[str] = field(default_factory=list)


@dataclass
class ActiveCheck:
    session: CheckSession
    profile: PackProfile
    frames: list[AnalyzedFrame] = field(default_factory=list)
    evidence: AnalyzedFrame | None = None
    evidence_file: str | None = None
    timings: dict = field(default_factory=dict)
    explanation: Explanation | None = None
    expert_calls: list[ExpertCall] = field(default_factory=list)
    review_errors: list[str] = field(default_factory=list)
    record: CheckRecord | None = None
    audit_sequence: int | None = None
    learning: LearningSummary | None = None


class NoCamera(FrameSource):
    def __init__(self, reason: str):
        self.reason = reason

    def read(self) -> np.ndarray | None:
        time.sleep(0.2)
        return None


def load_embedder(models: Path):
    """The station's ONNX embedder if one is installed and intact, otherwise the classic embedder."""
    path, manifest_path = models / "embedder.onnx", models / "embedder.json"
    if not path.exists() or not manifest_path.exists():
        return ClassicEmbedder(), "no embedder installed (run: avavision fetch-models)"
    manifest = EmbedderManifest.model_validate_json(manifest_path.read_text())
    if not manifest.is_intact(file_sha256(path)):
        return ClassicEmbedder(), "the installed embedder failed its integrity check"
    embedder = OnnxEmbedder(path, manifest.embedder_id, manifest.notes or manifest.embedder_id, manifest.input_size)
    return embedder, None


def migrate_brain(brain: Brain, embedder, crops: Path, batch: int = 64) -> Brain:
    """Moves the brain to a new embedder by re-embedding every stored crop. Pills without a crop are dropped;
    thresholds and track records start again because the vector space changed."""

    def embed(files: list[str]) -> dict[str, np.ndarray]:
        images = {f: cv2.imread(str(crops / f)) for f in files}
        readable = [f for f, image in images.items() if image is not None]
        vectors: dict[str, np.ndarray] = {}
        for start in range(0, len(readable), batch):
            chunk = readable[start : start + batch]
            for name, vector in zip(chunk, embedder.embed([images[f] for f in chunk]), strict=True):
                vectors[name] = vector
        return vectors

    by_crop = embed(sorted({e.crop_file for e in brain.knowledge.exemplars if e.crop_file}))
    migrated = brain.migrated(
        embedder.id, {e.id: by_crop[e.crop_file] for e in brain.knowledge.exemplars if e.crop_file in by_crop}
    )
    task_crops = embed(sorted({s.crop_file for t in brain.labelling_queue for s in t.sightings if s.crop_file}))
    for task in brain.labelling_queue:
        if all(s.crop_file in task_crops for s in task.sightings):
            for sighting in task.sightings:
                sighting.vector = task_crops[sighting.crop_file]
                sighting.embedder_id = embedder.id
                sighting.identity = None  # an opinion from the old space must not count towards new trust
            migrated.labelling_queue.append(task)
    return migrated


def load_detector(models: Path) -> tuple[ActiveModel, OnnxDetector | None, str | None]:
    """The station's own detector if one is installed and the release gate lets it run; otherwise the built-in
    segmenter, which only counts. The gate decides what the detector may do (count only, or identify)."""
    path, manifest_path = models / "detector.onnx", models / "detector.json"
    if not path.exists() or not manifest_path.exists():
        return builtin_pocket_segmenter(), None, None
    manifest = ModelManifest.model_validate_json(manifest_path.read_text())
    decision = evaluate_gate(manifest, file_sha256(path))
    if decision.capability.kind == CapabilityKind.UNAVAILABLE:
        reasons = ", ".join(b.kind.value for b in decision.blockers) or "unavailable"
        return builtin_pocket_segmenter(), None, f"detector {manifest.model_id} not used: {reasons}"
    model = ActiveModel(manifest=manifest, decision=decision)
    return model, OnnxDetector(path, manifest.class_names), None


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


class Station:
    def __init__(
        self,
        db: Database,
        settings: StationSettings,
        source: FrameSource,
        *,
        demo: DemoCamera | None = None,
        embedder=None,
        embedder_note: str | None = None,
        model: ActiveModel | None = None,
        detector: OnnxDetector | None = None,
        model_note: str | None = None,
        expert: ClaudeExpert | None = None,
        phone: PhoneLink | None = None,
    ):
        self.db = db
        self.phone = phone
        self.settings = settings
        self.demo = demo
        self.model = model or builtin_pocket_segmenter()
        self.detector = detector
        self.model_note = model_note
        self.embedder = embedder or ClassicEmbedder()
        self.embedder_note = embedder_note
        self.brain = db.load_brain(self.embedder.id) or Brain(self.embedder.id)
        if self.brain.embedder_id != self.embedder.id:
            self.brain = migrate_brain(self.brain, self.embedder, db.crops)
            db.save_brain(self.brain)
            db.prune_crops(db.referenced_crops(self.brain))
        self.policy = DecisionPolicy()
        self.expert = expert
        self.feed = LiveFeed(source)
        self.camera_error = source.reason if isinstance(source, NoCamera) else None
        self.live: LiveStatus | None = None
        self.check: ActiveCheck | None = None
        self.audit_defect = db.verify_audit()
        self.references = db.images / "references"
        self.references.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._running = False
        self._live_thread: threading.Thread | None = None
        self._apply_layout()

    @classmethod
    def open(cls, data_dir: Path | str, demo: bool = False) -> Station:
        data_dir = Path(data_dir)
        db = Database(data_dir)
        settings = StationSettings.model_validate(db.setting("station") or {})
        demo_camera = None
        phone = None
        if demo:
            demo_camera = DemoCamera()
            layout = demo_camera.station.layout()
            if layout.id not in {lay.id for lay in db.layouts()}:
                db.save_layout(layout)
            if not db.catalog().medications:
                db.save_catalog(demo_catalog())
                db.save_profile(demo_profile(demo_camera.station))
            settings.layout_id = layout.id
            # The demo pack waits on the tray with its header card, as a pharmacist would place it.
            carded = next((p for p in db.profiles() if p.barcode), None)
            if carded is not None:
                demo_camera.show(carded)
            source: FrameSource = demo_camera
        elif settings.camera_source == CameraKind.IPHONE:
            camera = PhoneCamera()
            key = db.setting("phone_key") or new_pairing_key()
            db.save_setting("phone_key", key)
            try:
                phone = PhoneLink(
                    data_dir / "phone", camera, key, port=settings.phone_port, setup_port=settings.phone_setup_port
                )
                source = camera
            except Exception as error:  # certificates could not be made: say so instead of failing to start
                source = NoCamera(f"the iPhone camera could not be prepared: {error}")
        else:
            try:
                source = CameraSource(
                    CameraSettings(
                        index=settings.camera_index,
                        width=settings.camera_width,
                        height=settings.camera_height,
                        exposure=settings.camera_exposure,
                    )
                )
            except RuntimeError as error:
                source = NoCamera(str(error))
        embedder, note = load_embedder(data_dir / "models")
        model, detector, model_note = load_detector(data_dir / "models")
        expert = ClaudeExpert(settings.expert, api_key=load_api_key(db)) if settings.expert_enabled else None
        return cls(
            db,
            settings,
            source,
            demo=demo_camera,
            embedder=embedder,
            embedder_note=note,
            model=model,
            detector=detector,
            model_note=model_note,
            expert=expert,
            phone=phone,
        )

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        if self.phone is not None:
            self.phone.start()
        self.feed.start()
        self._live_thread = threading.Thread(target=self._live_loop, name="avavision-live", daemon=True)
        self._live_thread.start()

    def stop(self) -> None:
        self._running = False
        if self._live_thread is not None:
            self._live_thread.join(timeout=2)
        if self.phone is not None:
            self.phone.stop()
        self.feed.stop()
        self.db.close()

    def _apply_layout(self) -> None:
        layouts = {lay.id: lay for lay in self.db.layouts()}
        self.layout = layouts.get(self.settings.layout_id, WEEKLY_7X4)
        self.analyzer = FrameAnalyzer(self.layout, mode=self.settings.finder_mode, detector=self.detector)

    def _live_loop(self) -> None:
        last = None
        codes: list[str] = []
        last_codes = 0.0
        while self._running:
            latest = self.feed.latest()
            if latest is None or latest[1] == last:
                time.sleep(0.02)
                continue
            last = latest[1]
            start = time.perf_counter()
            observation = self.analyzer.locate(latest[0]).observation
            registration = observation.registration.registration
            metrics = observation.quality.metrics
            if time.time() - last_codes >= 0.5:
                codes, last_codes = read_codes(latest[0]), time.time()
            self.live = LiveStatus(
                codes=codes,
                at=latest[1],
                usable=observation.is_usable,
                quality_issues=[i.value for i in observation.quality.issues],
                registration_issues=[i.value for i in observation.registration.issues],
                quad=[(p.x, p.y) for p in registration.quad.corners] if registration else None,
                sharpness=metrics.sharpness if metrics else None,
                locate_ms=_ms(start),
            )
            time.sleep(max(0.0, 0.1 - (time.perf_counter() - start)))

    def _grab(self, count: int, after: float = 0.0, spacing: float = 0.06, timeout: float = 5.0) -> list[np.ndarray]:
        images: list[np.ndarray] = []
        last = after
        deadline = time.time() + timeout
        while len(images) < count:
            latest = self.feed.latest()
            if latest is not None and latest[1] > last and (not images or latest[1] - last >= spacing):
                images.append(latest[0])
                last = latest[1]
            elif time.time() > deadline:
                raise StationError(self._camera_problem() or "the camera is not delivering frames")
            else:
                time.sleep(0.005)
        return images

    # ------------------------------------------------------------------ status

    @property
    def camera_kind(self) -> str:
        """The camera actually in use (a changed setting applies after a restart)."""
        if self.demo is not None:
            return "demo"
        return CameraKind.IPHONE.value if self.phone is not None else CameraKind.USB.value

    def _camera_problem(self) -> str | None:
        if self.camera_error:
            return self.camera_error
        if self.phone is not None and not self.phone.camera.connected:
            return self.phone.view()["problem"] or "iPhone not connected: open the AvaVision camera page on the iPhone"
        return None

    def status(self) -> dict:
        embedder_providers = getattr(self.embedder, "providers", ["builtin"])
        return {
            "version": __version__,
            "station_id": self.settings.station_id,
            "demo": self.demo is not None,
            "layout": {"id": self.layout.id, "name": self.layout.display_name, "calibrated": self.layout.is_calibrated},
            "camera": {"source": self.camera_kind, "error": self._camera_problem(), "frames": self.feed.frames_read},
            "providers": available_providers(),
            "embedder": {"id": self.embedder.id, "providers": embedder_providers, "note": self.embedder_note},
            "model": {
                "id": self.model.manifest.model_id,
                "version": self.model.manifest.version,
                "capability": self.model.capability.kind.value,
                "note": self.model_note,
            },
            "brain": self.brain.summary.model_dump(mode="json"),
            "expert": {
                "enabled": self.expert is not None,
                "model": self.settings.expert.model,
                "fallbacks": self.settings.expert.fallbacks,
                "key": bool(load_api_key(self.db) or os.environ.get("ANTHROPIC_API_KEY")),
            },
            "audit": {
                "entries": self.db.audit_count(),
                "defect": self.audit_defect.model_dump() if self.audit_defect else None,
            },
            "check": self.check.session.phase.value if self.check else None,
        }

    def live_view(self) -> dict | None:
        return asdict(self.live) if self.live else None

    # ------------------------------------------------------------------ iPhone camera

    def _phone(self) -> PhoneLink:
        if self.phone is None:
            raise StationError("the iPhone camera is not in use: choose it in Settings and restart the station")
        return self.phone

    def phone_view(self) -> dict:
        return self._phone().view()

    def phone_qr(self, which: str) -> bytes:
        """QR code of the setup page or of the camera page (the latter carries the pairing key)."""
        phone = self._phone()
        url = {"setup": phone.setup_url, "camera": phone.camera_url}.get(which)
        if url is None:
            raise StationError("no local network address: connect this computer to the iPhone's Wi-Fi")
        return cv2.imencode(".png", code_image(url, scale=8))[1].tobytes()

    def pair_phone(self) -> dict:
        """A new pairing key: the connected iPhone (and any copy of the old QR code) stops working."""
        phone = self._phone()
        phone.pair_again()
        self.db.save_setting("phone_key", phone.key)
        return phone.view()

    def latest_jpeg(self, width: int = 1280, quality: int = 75) -> bytes | None:
        latest = self.feed.latest()
        if latest is None:
            return None
        image = latest[0]
        if image.shape[1] > width:
            image = cv2.resize(
                image, (width, int(image.shape[0] * width / image.shape[1])), interpolation=cv2.INTER_AREA
            )
        ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buffer.tobytes() if ok else None

    def rectified_jpeg(self) -> bytes:
        """The latest frame flattened to the pack, with the layout's grid drawn on it (for calibration)."""
        images = self._grab(1)
        frame = self.analyzer.locate(images[0])
        registration = frame.observation.registration.registration
        if registration is None:
            issues = ", ".join(i.value for i in frame.observation.registration.issues) or "pack not found"
            raise StationError(f"the pack was not found: {issues}")
        canvas = rectify(images[0], registration, self.layout).canvas.copy()
        h, w = canvas.shape[:2]
        for index in self.layout.all_compartments:
            r = self.layout.cell_rect(index)
            cv2.rectangle(canvas, (int(r.x * w), int(r.y * h)), (int(r.max_x * w), int(r.max_y * h)), (0, 200, 255), 2)
        return cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()

    # ------------------------------------------------------------------ settings, catalog, layouts, profiles

    def update_settings(self, changes: dict) -> StationSettings:
        with self._lock:
            if self.check and self.check.session.phase != Phase.COMPLETED:
                raise StationError("finish the current check first")
            merged = self.settings.model_dump() | changes
            settings = StationSettings.model_validate(merged)
            if self.demo is not None:
                settings.layout_id = self.settings.layout_id
            self.db.save_setting("station", settings.model_dump(mode="json"))
            self.settings = settings
            self._apply_layout()
            self.expert = (
                ClaudeExpert(settings.expert, api_key=load_api_key(self.db)) if settings.expert_enabled else None
            )
            return settings

    def catalog(self) -> Catalog:
        return self.db.catalog()

    def save_layout(self, layout: PackLayout) -> PackLayout:
        errors = layout.validation_errors()
        if errors:
            raise StationError(f"invalid layout: {[e.value for e in errors]}")
        with self._lock:
            self.db.save_layout(layout)
            if layout.id == self.layout.id:
                if self.check and self.check.session.phase != Phase.COMPLETED:
                    raise StationError("finish the current check first")
                self._apply_layout()
        return layout

    def layout_by_id(self, layout_id: str) -> PackLayout:
        found = next((lay for lay in self.db.layouts() if lay.id == layout_id), None)
        if found is None:
            raise StationError(f"unknown layout {layout_id}")
        return found

    def profile_issues(self, profile: PackProfile) -> list[dict]:
        layout = self.layout_by_id(profile.layout_id)
        return [i.model_dump(mode="json") for i in profile.issues(layout, self.catalog())]

    # ------------------------------------------------------------------ checks

    def _engine(self) -> VerificationEngine:
        brain = BrainContext(
            trusted_medications=frozenset(self.brain.trusted_medications),
            well_known_medications=frozenset(self.brain.well_known_medications),
        )
        return VerificationEngine(self.layout, self.model, brain, self.policy)

    def _active(self, phase: Phase) -> ActiveCheck:
        if self.check is None or self.check.session.phase != phase:
            raise StationError(f"no check in phase {phase.value}")
        return self.check

    def start_check(self, profile_id: str) -> dict:
        with self._lock:
            if self.check and self.check.session.phase == Phase.ANALYZED:
                raise StationError("sign off or cancel the current check first")
            profile = self.db.profile(profile_id)
            if profile is None:
                raise StationError(f"unknown profile {profile_id}")
            try:
                session = CheckSession(self.layout, profile, self.catalog())
            except ValueError as error:
                raise StationError(str(error)) from error
            self.check = ActiveCheck(session=session, profile=profile)
            if self.demo is not None:
                self.demo.show(profile, DemoFault.NONE)
            return self.check_view()

    def cancel_check(self) -> None:
        with self._lock:
            self.check = None

    def demo_fault(self, fault: DemoFault) -> None:
        if self.demo is None:
            raise StationError("the demo camera is not active")
        profile = self.check.profile if self.check else None
        self.demo.show(profile, fault, seed=int(time.time()))

    def capture(self) -> dict:
        with self._lock:
            check = self._active(Phase.CAPTURING)
            started = time.perf_counter()
            # Only frames taken after the request: they show the pack as it is now.
            images = self._grab(self.policy.required_consistent_frames, after=time.time())
            grabbed = time.perf_counter()
            with ThreadPoolExecutor(max_workers=len(images)) as pool:
                frames = list(pool.map(lambda image: self.analyzer.detect(self.analyzer.locate(image)), images))
            detected = time.perf_counter()
            usable = [f for f in frames if f.rectified is not None]
            evidence = None
            if usable:
                evidence = self.analyzer.identify(
                    usable[0],
                    self.embedder,
                    self.brain.classifier(),
                    self.model,
                    self.policy.minimum_detection_confidence,
                    self.brain.physical_ranges(),
                )
                for other in usable[1:]:
                    propagate_identities(evidence, other, self.layout)
            identified = time.perf_counter()
            for frame in frames:
                check.session.record(frame.observation)
            check.session.analyze(self._engine())
            check.session.see_card(read_codes(images[0]))
            groups = peer_groups(check.profile, self.layout)
            if evidence is not None and groups:
                # Compartments that should hold the same tablets must look alike; obscured ones cannot be compared.
                obscured = set(evidence.observation.obscured)
                signatures = compartment_signatures(evidence.rectified.canvas, self.layout, obscured)
                check.session.compare_peers(unlike_peers(signatures, groups))
            check.frames, check.evidence = frames, evidence
            check.timings = {
                "grab_ms": round((grabbed - started) * 1000, 1),
                "detect_ms": round((detected - grabbed) * 1000, 1),
                "identify_ms": round((identified - detected) * 1000, 1),
                "decide_ms": _ms(identified),
                "total_ms": _ms(started),
                "pills_identified": len(evidence.pills) if evidence else 0,
                "frames": [f.timings_ms for f in frames],
            }
            if evidence is not None and self.settings.keep_evidence_images:
                name = f"{check.session.id}.jpg"
                cv2.imwrite(str(self.db.evidence / name), evidence.rectified.canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
                check.evidence_file = name
            return self.check_view()

    def retake(self) -> dict:
        with self._lock:
            check = self._active(Phase.ANALYZED)
            check.session.retake()
            check.frames, check.evidence, check.explanation = [], None, None
            check.timings, check.review_errors = {}, []
            return self.check_view()

    def evidence_jpeg(self) -> bytes:
        check = self.check
        if check is None or check.evidence is None or check.evidence.rectified is None:
            raise StationError("no evidence image")
        return cv2.imencode(".jpg", check.evidence.rectified.canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()

    def sign_off(self, sign_off: SignOff) -> dict:
        with self._lock:
            check = self._active(Phase.ANALYZED)
            record = check.session.complete(
                sign_off,
                __version__,
                self.settings.station_id,
                ModelSummary.of(self.model),
                self.brain.summary,
                [check.evidence_file] if check.evidence_file else [],
            )
            entry = self.db.append_audit(record)
            check.record, check.audit_sequence = record, entry.sequence
            pills = check.evidence.pills if check.evidence else []
            sightings = [
                PillSighting(
                    compartment=p.compartment,
                    vector=p.vector,
                    embedder_id=self.embedder.id,
                    identity=p.identity,
                    crop_file=f"{uuid.uuid4()}.jpg",
                    features=p.features,
                )
                for p in pills
            ]
            plan = plan_learning(str(record.id), sign_off, check.profile, self.layout, sightings)
            used = {e.crop_file for e in plan.exemplars} | {s.crop_file for t in plan.tasks for s in t.sightings}
            for pill, sighting in zip(pills, sightings, strict=True):
                if sighting.crop_file in used:
                    cv2.imwrite(str(self.db.crops / sighting.crop_file), pill.crop)
            check.learning = self.brain.learn(plan)
            self.db.save_brain(self.brain)
            return self.check_view()

    # ------------------------------------------------------------------ expert

    def _expert(self) -> ClaudeExpert:
        if self.expert is None:
            raise StationError("the expert is turned off in settings")
        return self.expert

    def import_chart(self, data: bytes, media_type: str, reference: str, layout_id: str | None = None) -> dict:
        answer = self._expert().read_chart(data, media_type)
        layout = self.layout_by_id(layout_id or self.layout.id)
        draft = chart_to_profile(answer.value, self.catalog(), layout, reference)
        return {"draft": self._draft_view(draft, layout), "call": answer.call.model_dump()}

    def redraft(
        self, chart: MedicationChart, reference: str, layout_id: str, overrides: dict[int, str] | None = None
    ) -> dict:
        layout = self.layout_by_id(layout_id)
        draft = chart_to_profile(chart, self.catalog(), layout, reference, overrides)
        return {"draft": self._draft_view(draft, layout)}

    def _draft_view(self, draft: ProfileDraft, layout: PackLayout) -> dict:
        view = draft.model_dump(mode="json")
        view["profile_issues"] = [i.model_dump(mode="json") for i in draft.profile.issues(layout, self.catalog())]
        return view

    def review(self, compartments: list[CompartmentIndex] | None = None) -> dict:
        """Second opinions, by default on every compartment the engine accepted. They can only escalate."""
        expert = self._expert()
        with self._lock:
            check = self._active(Phase.ANALYZED)
            if check.evidence is None or check.evidence.rectified is None:
                raise StationError("no usable photo to review")
            result = check.session.result
            targets = compartments or [
                v.compartment
                for v in result.compartments
                if v.status in (CompartmentStatus.VERIFIED, CompartmentStatus.COUNT_MATCHED)
            ]
            images = {i: compartment_crop(check.evidence.rectified, self.layout, i) for i in targets}
            session_id = check.session.id

        def ask(index: CompartmentIndex):
            try:
                return expert.review_compartment(images[index], index, self.layout, check.profile, self.catalog())
            except ExpertError as error:
                return error

        with ThreadPoolExecutor(max_workers=6) as pool:
            answers = list(pool.map(ask, targets))
        opinions: list[AdvisoryOpinion] = []
        errors = []
        with self._lock:
            check = self._active(Phase.ANALYZED)
            if check.session.id != session_id:
                raise StationError("the check changed during the review")
            for index, answer in zip(targets, answers, strict=True):
                if isinstance(answer, ExpertError):
                    errors.append(f"{self.layout.label(index)}: {answer}")
                else:
                    opinions.append(answer.value)
                    check.expert_calls.append(answer.call)
            check.review_errors = errors
            check.session.advise(opinions)
            return self.check_view()

    def explain(self) -> dict:
        expert = self._expert()
        with self._lock:
            if self.check is None or self.check.session.result is None:
                raise StationError("nothing to explain")
            check = self.check
            result = check.session.result
        answer = expert.explain(result, self.layout, self.catalog(), self.settings.language)
        with self._lock:
            check.explanation = answer.value
            check.expert_calls.append(answer.call)
            return self.check_view()

    # ------------------------------------------------------------------ brain

    def teach(self, medication_id: str) -> dict:
        with self._lock:
            if self.check and self.check.session.phase != Phase.COMPLETED:
                raise StationError("finish or cancel the current check first")
            catalog = self.catalog()
            if medication_id not in catalog:
                raise StationError(f"unknown medication {medication_id}")
            if self.demo is not None:
                profile = PackProfile.empty("teach", self.layout)
                for index in self.layout.all_compartments:
                    profile.set_items(index, [ExpectedItem(medication_id=medication_id, quantity=2)])
                self.demo.show(profile, DemoFault.NONE, seed=int(time.time()))
            image = self._grab(1, after=time.time())[0]
            frame = self.analyzer.detect(self.analyzer.locate(image))
            if frame.rectified is None:
                issues = [i.value for i in frame.observation.quality.issues + frame.observation.registration.issues]
                raise StationError(f"the photo is not usable: {issues}")
            frame = self.analyzer.identify(
                frame, self.embedder, self.brain.classifier(), self.model, self.policy.minimum_detection_confidence
            )
            if not frame.pills:
                raise StationError("no tablets were found")
            vectors = []
            for pill in frame.pills:
                name = f"{uuid.uuid4()}.jpg"
                cv2.imwrite(str(self.db.crops / name), pill.crop)
                vectors.append((pill.vector, name))
            summary = self.brain.teach(
                medication_id,
                vectors,
                group_id=f"teach-{uuid.uuid4()}",
                measurements=[p.features for p in frame.pills if p.features],
            )
            self.db.save_brain(self.brain)
            self.db.prune_crops(self.db.referenced_crops(self.brain))
            return {"pills": len(frame.pills), "learning": _learning(summary)}

    def resolve_task(self, task_id: str, labels: dict[str, str]) -> dict:
        with self._lock:
            summary = self.brain.resolve_task(task_id, labels)
            self.db.save_brain(self.brain)
            return _learning(summary)

    def discard_task(self, task_id: str) -> None:
        with self._lock:
            self.brain.discard_task(task_id)
            self.db.save_brain(self.brain)
            self.db.prune_crops(self.db.referenced_crops(self.brain))

    def forget(self, medication_id: str) -> None:
        with self._lock:
            self.brain.forget(medication_id)
            self.db.save_brain(self.brain)
            self.db.prune_crops(self.db.referenced_crops(self.brain))

    def recalibrate(self) -> dict:
        with self._lock:
            report = self.brain.recalibrate()
            self.db.save_brain(self.brain)
            return report.model_dump(mode="json")

    def pill_jpeg(self, detection_index: int) -> bytes:
        check = self.check
        pill = next(
            (
                p
                for p in (check.evidence.pills if check and check.evidence else [])
                if p.detection_index == detection_index
            ),
            None,
        )
        if pill is None:
            raise StationError("no such pill")
        return cv2.imencode(".jpg", pill.crop, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()

    def reference_path(self, medication_id: str) -> Path | None:
        """The pharmacist-approved reference photo of a medication: one set by hand, otherwise the most typical
        pill in the brain's memory (every pill there was confirmed by a pharmacist)."""
        if not _SAFE_ID.match(medication_id):
            return None
        chosen = self.references / f"{medication_id}.jpg"
        if chosen.is_file():
            return chosen
        exemplars = [
            e
            for e in self.brain.knowledge.exemplars_for(medication_id)
            if e.crop_file and (self.db.crops / e.crop_file).is_file()
        ]
        if not exemplars:
            return None
        centre = np.mean([e.vector for e in exemplars], axis=0)
        typical = max(exemplars, key=lambda e: float(np.dot(e.vector, centre)))
        return self.db.crops / typical.crop_file

    def set_reference(self, medication_id: str, data: bytes | None) -> None:
        if not _SAFE_ID.match(medication_id) or medication_id not in self.catalog():
            raise StationError(f"unknown medication {medication_id}")
        path = self.references / f"{medication_id}.jpg"
        if data is None:
            path.unlink(missing_ok=True)
            return
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise StationError("the file is not an image")
        cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 92])

    def brain_view(self) -> dict:
        catalog = self.catalog()
        knowledge = self.brain.knowledge
        medications = sorted(knowledge.medications | {m.id for m in catalog.medications})
        policy = self.brain.identity_policy
        rows = []
        for medication in medications:
            found = catalog.get(medication)
            status = self.brain.trust_status(medication)
            record = self.brain.ledger.record(medication)
            rows.append(
                {
                    "id": medication,
                    "name": found.display_name if found else medication,
                    "exemplars": knowledge.count(medication),
                    "groups": knowledge.group_count(medication),
                    "trust": status.state.value,
                    "progress": round(status.progress, 3),
                    "correct": record.correct,
                    "false_identifications": record.false_identifications,
                    "streak": record.streak,
                }
            )
        tasks = [
            {
                "id": t.id,
                "compartment": t.compartment_label,
                "expected": t.expected,
                "priority": t.priority,
                "pills": [
                    {
                        "id": s.id,
                        "crop": s.crop_file,
                        "suggestion": s.identity.decision.named if s.identity else None,
                    }
                    for s in t.sightings
                ],
            }
            for t in self.brain.labelling_queue
        ]
        return {
            "summary": self.brain.summary.model_dump(mode="json"),
            "policy": {
                "calibrated": policy.is_calibrated,
                "accept_similarity": policy.accept_similarity,
                "minimum_margin": policy.minimum_margin,
                "version": policy.version,
            },
            "trust_policy": self.brain.trust_policy.model_dump(),
            "calibration": self.brain.last_calibration.model_dump(mode="json") if self.brain.last_calibration else None,
            "medications": rows,
            "tasks": tasks,
        }

    # ------------------------------------------------------------------ views

    def check_view(self) -> dict | None:
        check = self.check
        if check is None:
            return None
        catalog = self.catalog()
        session = check.session
        result = session.result
        view: dict = {
            "session_id": str(session.id),
            "phase": session.phase.value,
            "profile": {
                "id": str(check.profile.id),
                "reference": check.profile.reference,
                "barcode": check.profile.barcode,
            },
            "card_codes": session.card_codes,
            "layout": self.layout.model_dump(mode="json"),
            "timings": check.timings,
            "evidence": check.evidence is not None,
            "explanation": check.explanation.model_dump() if check.explanation else None,
            "advisories": [o.model_dump(mode="json") for o in session.advisories],
            "review_errors": check.review_errors,
            "expert_calls": [c.model_dump() for c in check.expert_calls],
            "record_id": str(check.record.id) if check.record else None,
            "audit_sequence": check.audit_sequence,
            "learning": _learning(check.learning) if check.learning else None,
            "result": None,
        }
        if result is None:
            return view
        spot = set(result.spot_checks)
        requiring = set(result.compartments_requiring_review)
        opinions = {o.compartment: o for o in session.advisories}
        compartments = []
        ranges = self.brain.physical_ranges()
        references = {m.id for m in catalog.medications if self.reference_path(m.id) is not None}
        pills_by_compartment: dict[CompartmentIndex, list[dict]] = {}
        for pill in check.evidence.pills if check.evidence else []:
            detection = check.evidence.observation.detections[pill.detection_index]
            named = pill.identity.decision.named
            physical = detection.physical
            pills_by_compartment.setdefault(pill.compartment, []).append(
                {
                    "index": pill.detection_index,
                    "size": _size(pill.features),
                    "identity": (m.display_name if (m := catalog.get(named)) else named) if named else None,
                    "fits": list(physical.consistent_with) if physical and physical.known else None,
                }
            )
        for verdict in result.compartments:
            expectation = check.profile.expectation(verdict.compartment)
            opinion = opinions.get(verdict.compartment)
            compartments.append(
                {
                    "row": verdict.compartment.row,
                    "column": verdict.compartment.column,
                    "label": self.layout.label(verdict.compartment),
                    "status": verdict.status.value,
                    "findings": [describe_finding(f, catalog) for f in verdict.findings],
                    "expected": [
                        {
                            "id": i.medication_id,
                            "name": (m.display_name if (m := catalog.get(i.medication_id)) else i.medication_id),
                            "quantity": i.quantity,
                            "reference": i.medication_id in references,
                            "range": _range_text(ranges.get(i.medication_id)),
                        }
                        for i in (expectation.items if expectation else [])
                    ],
                    "pills": pills_by_compartment.get(verdict.compartment, []),
                    "expected_count": verdict.expected_count,
                    "observed_count": verdict.observed_count,
                    "spot_check": verdict.compartment in spot,
                    "requires_review": verdict.compartment in requiring,
                    "advisory": opinion.model_dump(mode="json") if opinion else None,
                }
            )
        view["result"] = {
            "id": str(result.id),
            "status": result.status.value,
            "pack_findings": [describe_pack_finding(f) for f in result.pack_findings],
            "has_pack_findings": bool(result.pack_findings),
            "usable_frames": result.usable_frame_count,
            "capability": result.capability.kind.value,
            "trusted_medications": result.trusted_medications,
            "compartments": compartments,
        }
        view["detections"] = self._detections_view(check)
        return view

    def _detections_view(self, check: ActiveCheck) -> list[dict]:
        frame = check.evidence
        if frame is None or frame.rectified is None:
            return []
        catalog = self.catalog()
        located = pill_compartments(
            frame.observation, self.layout, self.policy.minimum_detection_confidence, self.model.meaning
        )
        h, w = frame.rectified.canvas.shape[:2]
        items = []
        for i, detection in enumerate(frame.observation.detections):
            x0, y0, x1, y1 = frame.rectified.image_box_to_canvas(detection.box)
            decision = detection.identity.decision if detection.identity else None
            named = decision.named if decision else None
            items.append(
                {
                    "box": [x0 / w, y0 / h, (x1 - x0) / w, (y1 - y0) / h],
                    "confidence": round(detection.confidence, 3),
                    "counted": i in located,
                    "identity": (m.display_name if (m := catalog.get(named)) else named) if named else None,
                    "decision": decision.kind.value if decision else None,
                }
            )
        return items


def _size(features: PhysicalFeatures | None) -> str | None:
    if features is None:
        return None
    if abs(features.length_mm - features.width_mm) < 0.3:
        return f"{features.length_mm:.1f} mm round"
    return f"{features.length_mm:.1f} × {features.width_mm:.1f} mm"


def _range_text(r: PhysicalRange | None) -> str | None:
    if r is None:
        return None
    return (
        f"{r.length_mm[0]:.1f}–{r.length_mm[1]:.1f} × {r.width_mm[0]:.1f}–{r.width_mm[1]:.1f} mm "
        f"({r.samples} pills, {r.groups} packs)"
    )


def _learning(summary: LearningSummary) -> dict:
    data = {k: v for k, v in asdict(summary).items() if k != "recalibration"}
    data["recalibration"] = summary.recalibration.model_dump(mode="json") if summary.recalibration else None
    return data
