"""The station's local web API and UI. It listens on localhost only; the desktop window is a view of it."""

from __future__ import annotations

import asyncio
import csv
import io
import re
from contextlib import asynccontextmanager
from importlib import resources
from pathlib import Path
from typing import Annotated
from urllib.parse import urlsplit

import cv2
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.trustedhost import TrustedHostMiddleware

from ..brain.learning import LabellingError
from ..core.audit import CheckRecord
from ..core.models import CompartmentIndex, Medication, PackLayout, PackProfile
from ..core.session import SessionError
from ..core.signoff import SignOff, SignOffError
from ..expert.charts import MedicationChart
from ..expert.claude import ExpertIncomplete, ExpertRefused, ExpertUnavailable, media_type_of
from ..storage.database import AuditChainBroken
from ..vision.pack_finder import marker_sheet
from .credentials import save_api_key
from .demo import DemoFault
from .service import Advisor, Station, StationError

_FILE = re.compile(r"^[0-9a-f-]{8,64}\.jpg$")


class StartCheck(BaseModel):
    profile_id: str


class ReviewRequest(BaseModel):
    compartments: list[CompartmentIndex] | None = None
    advisor: Advisor = Advisor.CLAUDE


class FaultRequest(BaseModel):
    fault: DemoFault


class TeachRequest(BaseModel):
    medication_id: str


class LabelRequest(BaseModel):
    labels: dict[str, str]


class RedraftRequest(BaseModel):
    chart: MedicationChart
    reference: str
    layout_id: str
    overrides: dict[int, str] = {}


class ApiKeyRequest(BaseModel):
    key: str | None


def ui_directory() -> Path:
    return Path(str(resources.files("avavision.station") / "ui"))


PLACEHOLDER = """<!doctype html><html><head><meta charset="utf-8"><title>AvaVision</title></head>
<body style="font-family:system-ui;padding:2rem"><h1>AvaVision station</h1>
<p>The interface is not built. Run <code>npm ci &amp;&amp; npm run build</code> in <code>web/</code>.</p>
<p>API: <a href="/docs">/docs</a></p></body></html>"""


def create_app(station: Station, start: bool = True, allowed_hosts: list[str] | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if start:
            station.start()
        yield
        if start:
            station.stop()

    app = FastAPI(title="AvaVision station", version=station.status()["version"], lifespan=lifespan)
    if allowed_hosts:
        # A web page cannot reach the station through DNS rebinding.
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)

    @app.middleware("http")
    async def same_origin(request: Request, call_next):
        """Another web site open on the station's computer must not be able to change anything."""
        origin = request.headers.get("origin")
        if request.method not in ("GET", "HEAD", "OPTIONS") and origin:
            host = request.headers.get("host", "").split(":")[0]
            if urlsplit(origin).hostname != host:
                return JSONResponse(status_code=403, content={"error": "cross-origin request refused"})
        return await call_next(request)

    # ------------------------------------------------------------------ errors

    def error(status: int, message: str, **extra):
        return JSONResponse(status_code=status, content={"error": message, **extra})

    @app.exception_handler(StationError)
    async def _station(_, e: StationError):
        return error(409, str(e))

    @app.exception_handler(SessionError)
    async def _session(_, e: SessionError):
        return error(409, str(e))

    @app.exception_handler(SignOffError)
    async def _sign_off(_, e: SignOffError):
        return error(422, e.kind.value, kind=e.kind.value, compartments=[c.model_dump() for c in e.compartments])

    @app.exception_handler(ExpertUnavailable)
    async def _unavailable(_, e: ExpertUnavailable):
        return error(503, str(e))

    @app.exception_handler(ExpertRefused)
    async def _refused(_, e: ExpertRefused):
        return error(422, str(e), category=e.category)

    @app.exception_handler(ExpertIncomplete)
    async def _incomplete(_, e: ExpertIncomplete):
        return error(502, str(e))

    @app.exception_handler(LabellingError)
    async def _labelling(_, e: LabellingError):
        return error(422, str(e))

    @app.exception_handler(AuditChainBroken)
    async def _audit(_, e: AuditChainBroken):
        return error(500, str(e), defect=e.defect.model_dump())

    # ------------------------------------------------------------------ station and camera

    @app.get("/api/status")
    def status():
        return station.status()

    @app.get("/api/live")
    def live():
        return station.live_view()

    @app.get("/api/camera/snapshot.jpg")
    def snapshot():
        data = station.latest_jpeg(width=1920, quality=88)
        if data is None:
            raise HTTPException(503, "no camera frame yet")
        return Response(data, media_type="image/jpeg")

    @app.get("/api/camera/rectified.jpg")
    def rectified():
        return Response(station.rectified_jpeg(), media_type="image/jpeg")

    @app.get("/api/camera/stream")
    async def stream(width: int = 1280, fps: float = 15):
        boundary = "frame"

        async def frames():
            while True:
                data = await asyncio.to_thread(station.latest_jpeg, width)
                if data is not None:
                    yield (
                        f"--{boundary}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(data)}\r\n\r\n".encode()
                        + data
                        + b"\r\n"
                    )
                await asyncio.sleep(1 / max(1.0, min(fps, 30.0)))

        return StreamingResponse(frames(), media_type=f"multipart/x-mixed-replace; boundary={boundary}")

    @app.get("/api/markers.png")
    def markers():
        return Response(cv2.imencode(".png", marker_sheet())[1].tobytes(), media_type="image/png")

    # ------------------------------------------------------------------ iPhone camera

    @app.get("/api/phone")
    def phone():
        return station.phone_view()

    @app.get("/api/phone/{which}.png")
    def phone_qr(which: str):
        return Response(station.phone_qr(which), media_type="image/png", headers={"Cache-Control": "no-store"})

    @app.post("/api/phone/pair")
    def phone_pair():
        return station.pair_phone()

    # ------------------------------------------------------------------ settings

    @app.get("/api/settings")
    def get_settings():
        return station.settings.model_dump(mode="json")

    @app.put("/api/settings")
    def put_settings(changes: Annotated[dict, Body()]):
        return station.update_settings(changes).model_dump(mode="json")

    @app.put("/api/settings/api-key")
    def put_api_key(request: ApiKeyRequest):
        where = save_api_key(station.db, request.key.strip() if request.key else None)
        station.update_settings({})
        return {"stored_in": where}

    # ------------------------------------------------------------------ catalog, layouts, profiles

    @app.get("/api/catalog")
    def catalog():
        data = station.catalog().model_dump(mode="json")
        data["references"] = [m["id"] for m in data["medications"] if station.reference_path(m["id"]) is not None]
        return data

    @app.put("/api/catalog/medications/{medication_id}")
    def put_medication(medication_id: str, medication: Medication):
        if medication.id != medication_id:
            raise HTTPException(400, "id mismatch")
        catalog = station.catalog()
        catalog.upsert(medication)
        station.db.save_catalog(catalog)
        return medication.model_dump(mode="json")

    @app.delete("/api/catalog/medications/{medication_id}")
    def delete_medication(medication_id: str):
        catalog = station.catalog()
        catalog.remove(medication_id)
        station.db.save_catalog(catalog)
        return {"deleted": medication_id}

    @app.get("/api/layouts")
    def layouts():
        return [lay.model_dump(mode="json") for lay in station.db.layouts()]

    @app.put("/api/layouts/{layout_id}")
    def put_layout(layout_id: str, layout: PackLayout):
        if layout.id != layout_id:
            raise HTTPException(400, "id mismatch")
        return station.save_layout(layout).model_dump(mode="json")

    @app.get("/api/profiles")
    def profiles():
        return [
            {
                "id": str(p.id),
                "reference": p.reference,
                "barcode": p.barcode,
                "layout_id": p.layout_id,
                "doses": p.total_doses,
                "created_at": p.created_at.isoformat(),
                "issues": len(station.profile_issues(p)),
            }
            for p in station.db.profiles()
        ]

    @app.get("/api/profiles/{profile_id}")
    def get_profile(profile_id: str):
        profile = station.db.profile(profile_id)
        if profile is None:
            raise HTTPException(404, "unknown profile")
        return {"profile": profile.model_dump(mode="json"), "issues": station.profile_issues(profile)}

    @app.put("/api/profiles/{profile_id}")
    def put_profile(profile_id: str, profile: PackProfile):
        if str(profile.id) != profile_id:
            raise HTTPException(400, "id mismatch")
        station.layout_by_id(profile.layout_id)
        station.db.save_profile(profile)
        return {"profile": profile.model_dump(mode="json"), "issues": station.profile_issues(profile)}

    @app.delete("/api/profiles/{profile_id}")
    def delete_profile(profile_id: str):
        station.db.delete_profile(profile_id)
        return {"deleted": profile_id}

    @app.post("/api/profiles/import")
    async def import_chart(request: Request, reference: str, filename: str, layout_id: str | None = None):
        data = await request.body()
        if not data:
            raise HTTPException(400, "empty file")
        try:
            media_type = media_type_of(filename)
        except ValueError as e:
            raise HTTPException(415, str(e)) from e
        return await asyncio.to_thread(station.import_chart, data, media_type, reference, layout_id)

    @app.post("/api/profiles/redraft")
    def redraft(request: RedraftRequest):
        return station.redraft(request.chart, request.reference, request.layout_id, request.overrides)

    # ------------------------------------------------------------------ checks

    @app.get("/api/check")
    def get_check():
        return station.check_view()

    @app.post("/api/check/start")
    def start_check(request: StartCheck):
        return station.start_check(request.profile_id)

    @app.post("/api/check/capture")
    def capture():
        return station.capture()

    @app.post("/api/check/retake")
    def retake():
        return station.retake()

    @app.post("/api/check/cancel")
    def cancel():
        station.cancel_check()
        return None

    @app.post("/api/check/review")
    def review(request: ReviewRequest):
        return station.review(request.compartments, request.advisor)

    @app.post("/api/check/review/stop")
    def stop_review():
        return station.stop_review()

    @app.get("/api/advisor/local")
    def local_advisor_status():
        return station.local_model_view()

    @app.post("/api/advisor/local/setup")
    def setup_local_advisor():
        return station.setup_local_model()

    @app.post("/api/check/explain")
    def explain():
        return station.explain()

    @app.post("/api/check/sign-off")
    def sign_off(request: SignOff):
        return station.sign_off(request)

    @app.get("/api/check/pills/{index}.jpg")
    def pill_image(index: int):
        return Response(station.pill_jpeg(index), media_type="image/jpeg")

    @app.get("/api/medications/{medication_id}/reference.jpg")
    def reference_image(medication_id: str):
        path = station.reference_path(medication_id)
        if path is None:
            raise HTTPException(404, "no reference image")
        return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.put("/api/medications/{medication_id}/reference")
    async def put_reference(medication_id: str, request: Request):
        data = await request.body()
        await asyncio.to_thread(station.set_reference, medication_id, data or None)
        return {"medication_id": medication_id, "custom": bool(data)}

    @app.get("/api/check/evidence.jpg")
    def evidence():
        return Response(station.evidence_jpeg(), media_type="image/jpeg")

    @app.post("/api/demo/fault")
    def demo_fault(request: FaultRequest):
        station.demo_fault(request.fault)
        return {"fault": request.fault.value}

    # ------------------------------------------------------------------ brain

    @app.get("/api/brain")
    def brain():
        return station.brain_view()

    @app.post("/api/brain/teach")
    def teach(request: TeachRequest):
        return station.teach(request.medication_id)

    @app.post("/api/brain/tasks/{task_id}/resolve")
    def resolve_task(task_id: str, request: LabelRequest):
        try:
            return station.resolve_task(task_id, request.labels)
        except KeyError as e:
            raise HTTPException(404, "unknown task") from e

    @app.delete("/api/brain/tasks/{task_id}")
    def discard_task(task_id: str):
        station.discard_task(task_id)
        return {"deleted": task_id}

    @app.delete("/api/brain/medications/{medication_id}")
    def forget(medication_id: str):
        station.forget(medication_id)
        return {"forgotten": medication_id}

    @app.post("/api/brain/recalibrate")
    def recalibrate():
        return station.recalibrate()

    def image_file(folder: Path, name: str) -> FileResponse:
        path = folder / name
        if not _FILE.match(name) or not path.is_file():
            raise HTTPException(404, "no such image")
        return FileResponse(path, media_type="image/jpeg")

    @app.get("/api/crops/{name}")
    def crop(name: str):
        return image_file(station.db.crops, name)

    @app.get("/api/evidence/{name}")
    def evidence_file(name: str):
        return image_file(station.db.evidence, name)

    # ------------------------------------------------------------------ audit

    def summary(entry) -> dict:
        record = CheckRecord.model_validate_json(entry.payload)
        return {
            "sequence": entry.sequence,
            "hash": entry.hash,
            "created_at": record.created_at.isoformat(),
            "reference": record.profile.reference,
            "status": record.result.status.value,
            "decision": record.sign_off.decision.value,
            "pharmacist": record.sign_off.pharmacist,
            "evidence": record.evidence_images,
        }

    @app.get("/api/audit")
    def audit(limit: int = 50, before: int | None = None):
        return {
            "total": station.db.audit_count(),
            "entries": [summary(e) for e in station.db.audit_page(min(limit, 500), before)],
        }

    @app.get("/api/audit/export.csv")
    def export_records(since: str | None = None, until: str | None = None):
        """Checking records for QCPP and pharmacy audits: one row per signed-off pack."""
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(
            [
                "entry",
                "signed_at",
                "pack_reference",
                "header_card",
                "station",
                "result",
                "decision",
                "pharmacist",
                "compartments_inspected",
                "compartments_flagged",
                "pack_findings",
                "spot_checks",
                "evidence_photo",
                "record_hash",
            ]
        )
        for entry in station.db.audit_entries():
            record = CheckRecord.model_validate_json(entry.payload)
            signed = record.sign_off.signed_at.isoformat()
            if (since and signed < since) or (until and signed > until):
                continue
            flagged = [v for v in record.result.compartments if v.status.value in ("needsReview", "mismatch")]
            writer.writerow(
                [
                    entry.sequence,
                    signed,
                    record.profile.reference,
                    record.profile.barcode or "",
                    record.station_id,
                    record.result.status.value,
                    record.sign_off.decision.value,
                    record.sign_off.pharmacist,
                    len(record.sign_off.reviews),
                    len(flagged),
                    "; ".join(f.kind.value for f in record.result.pack_findings),
                    len(record.result.spot_checks),
                    "; ".join(record.evidence_images),
                    entry.hash,
                ]
            )
        return Response(
            buffer.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="avavision-checking-records.csv"'},
        )

    @app.get("/api/audit/{sequence}")
    def audit_entry(sequence: int):
        page = station.db.audit_page(1, sequence + 1)
        if not page or page[0].sequence != sequence:
            raise HTTPException(404, "no such entry")
        record = CheckRecord.model_validate_json(page[0].payload)
        return {"entry": summary(page[0]), "record": record.model_dump(mode="json")}

    @app.post("/api/audit/verify")
    def verify():
        defect = station.db.verify_audit()
        station.audit_defect = defect
        return {"intact": defect is None, "defect": defect.model_dump() if defect else None}

    # ------------------------------------------------------------------ interface

    ui = ui_directory()
    if (ui / "index.html").is_file():
        app.mount("/", StaticFiles(directory=ui, html=True), name="ui")
    else:

        @app.get("/", response_class=HTMLResponse)
        def placeholder():
            return PLACEHOLDER

    return app
