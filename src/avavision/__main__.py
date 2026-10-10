"""``avavision`` command line: run the station, install models, benchmark and check records."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

from . import __version__

#: Pinned so a changed upstream file can never silently change the brain's vector space.
DINOV2 = {
    "url": "https://huggingface.co/onnx-community/dinov2-with-registers-small/resolve/"
    "ec7abea1a8757ec4f9f7b26399d31234ffaa6e6a/onnx/model.onnx",
    "sha256": "815e440d222e60294bb5b165491c4760e7f08dc7ecd2488a5450e93d50887efe",
    "embedder_id": "dinov2s-reg-815e440d",
    "license": "Apache-2.0 (facebook/dinov2-with-registers-small)",
}


def default_data_dir() -> Path:
    if platform.system() == "Windows":
        return Path(os.environ.get("LOCALAPPDATA", Path.home())) / "AvaVision"
    return Path.home() / ".avavision"


def _serve(args, open_window: bool) -> None:
    import uvicorn

    from .station.api import create_app
    from .station.service import Station

    station = Station.open(args.data, demo=args.demo)
    loopback = args.host in ("127.0.0.1", "localhost")
    app = create_app(station, allowed_hosts=["127.0.0.1", "localhost"] if loopback else None)
    config = uvicorn.Config(app, host=args.host, port=args.port, log_level="warning")
    server = uvicorn.Server(config)
    url = f"http://{args.host}:{args.port}/"
    if not open_window:
        print(f"AvaVision {__version__} on {url} (data: {args.data})")
        server.run()
        return
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.05)
    try:
        import webview
    except ImportError:
        print(f"pywebview is not installed; opening {url} in the browser")
        webbrowser.open(url)
        thread.join()
        return
    webview.create_window("AvaVision", url, width=1440, height=900, min_size=(1100, 720))
    webview.start()
    server.should_exit = True
    thread.join(timeout=5)


def _fetch_models(args) -> None:
    from .core.gate import EmbedderManifest
    from .vision.runtime import file_sha256

    models = Path(args.data) / "models"
    models.mkdir(parents=True, exist_ok=True)
    target = models / "embedder.onnx"
    if not (target.exists() and file_sha256(target) == DINOV2["sha256"]):
        partial = models / "embedder.onnx.part"
        print(f"downloading {DINOV2['url']}")
        with urllib.request.urlopen(DINOV2["url"]) as response, open(partial, "wb") as out:
            total = int(response.headers.get("Content-Length", 0))
            done = 0
            while chunk := response.read(1 << 20):
                out.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r{done * 100 // total}%", end="", flush=True)
        print()
        digest = file_sha256(partial)
        if digest != DINOV2["sha256"]:
            partial.unlink()
            sys.exit(f"checksum mismatch: {digest}")
        partial.replace(target)
    manifest = EmbedderManifest(
        embedder_id=DINOV2["embedder_id"],
        version="1",
        model_sha256=DINOV2["sha256"],
        input_size=224,
        output_name="last_hidden_state",
        base_model_license=DINOV2["license"],
        notes="DINOv2-S with registers (general-purpose; replace with AvaVision's fine-tuned embedder when trained)",
    )
    (models / "embedder.json").write_text(manifest.model_dump_json(indent=2))
    print(f"embedder installed in {models}")


def _benchmark(args) -> None:
    import numpy as np

    from .core.gate import builtin_pocket_segmenter
    from .station.service import load_embedder
    from .vision.pipeline import FrameAnalyzer
    from .vision.runtime import available_providers
    from .vision.synthetic import PALETTE, Station, full_scene

    embedder, note = load_embedder(Path(args.data) / "models")
    station = Station()
    layout = station.layout()
    analyzer = FrameAnalyzer(layout)
    model = builtin_pocket_segmenter()
    from .brain.brain import Brain

    classifier = Brain(embedder.id).classifier()
    medications = list(PALETTE)[:2]
    timings: dict[str, list[float]] = {"locate": [], "detect": [], "identify": []}
    for k in range(args.frames):
        frame = analyzer.detect(analyzer.locate(station.render(full_scene(layout, medications), seed=k)))
        frame = analyzer.identify(frame, embedder, classifier, model, 0.6)
        for key in timings:
            timings[key].append(frame.timings_ms.get(key, 0.0))
    print(f"providers: {available_providers()}")
    print(f"embedder: {embedder.id} {getattr(embedder, 'providers', '')} {note or ''}")
    for key, values in timings.items():
        warm = values[1:] or values
        print(f"{key:>9}: {np.median(warm):7.1f} ms median ({len(frame.pills)} pills per frame)")


def _verify_audit(args) -> None:
    from .storage.database import Database

    db = Database(args.data)
    defect = db.verify_audit()
    count = db.audit_count()
    if defect:
        sys.exit(f"audit chain BROKEN at entry {defect.sequence}: {defect.defect} ({count} entries)")
    print(f"audit chain intact ({count} entries)")


def _export_training(args) -> None:
    """Pills the brain learned from pharmacist-confirmed checks, in the format ``training/`` reads."""
    import shutil

    from .storage.database import Database

    db = Database(args.data)
    brain = db.load_brain("")
    if brain is None:
        sys.exit("the brain is empty")
    out = Path(args.out)
    (out / "crops").mkdir(parents=True, exist_ok=True)
    count = 0
    with open(out / "labels.jsonl", "w", encoding="utf-8") as labels:
        for exemplar in brain.knowledge.exemplars:
            source = db.crops / exemplar.crop_file if exemplar.crop_file else None
            if source is None or not source.is_file():
                continue
            shutil.copyfile(source, out / "crops" / exemplar.crop_file)
            row = {
                "crop": f"crops/{exemplar.crop_file}",
                "medicationID": exemplar.medication_id,
                "source": exemplar.source.value,
                "groupID": exemplar.group_id,
                "createdAt": exemplar.created_at.isoformat(),
            }
            labels.write(json.dumps(row) + "\n")
            count += 1
    print(f"exported {count} pill images of {len(brain.knowledge.medications)} medications to {out}")


def _evaluate(args) -> None:
    """Holdout samples (JSON list) → the evaluation report that goes into a detector manifest."""
    from pydantic import TypeAdapter

    from .core.evaluation import EvaluationSample, evaluate

    samples = TypeAdapter(list[EvaluationSample]).validate_json(Path(args.samples).read_text())
    print(evaluate(samples, args.dataset_id).model_dump_json(indent=2))


def _gate(args) -> None:
    from .core.gate import ModelManifest, evaluate_gate
    from .vision.runtime import file_sha256

    manifest = ModelManifest.model_validate_json(Path(args.manifest).read_text())
    decision = evaluate_gate(manifest, file_sha256(args.model))
    print(json.dumps(decision.model_dump(mode="json"), indent=2))
    sys.exit(0 if not decision.blockers else 1)


def _markers(args) -> None:
    import cv2

    from .vision.pack_finder import marker_sheet

    cv2.imwrite(args.out, marker_sheet())
    print(f"marker sheet written to {args.out} (print at 100%, cut out, stick ids 0-3 clockwise from top-left)")


def _compare_advisors(args) -> None:
    """Second-opinion advisors scored on the compartments pharmacists already confirmed or corrected."""
    import csv

    import cv2

    from .core.audit import CheckRecord
    from .expert.claude import ClaudeExpert
    from .expert.compare import compare_advisors, report, samples_from_records
    from .expert.local import LocalVisionAdvisor
    from .station.credentials import load_api_key
    from .station.service import StationSettings
    from .storage.database import Database

    db = Database(args.data)
    settings = StationSettings.model_validate(db.setting("station") or {})
    records = [CheckRecord.model_validate_json(entry.payload) for entry in db.audit_entries()]

    def load(name: str):
        path = db.evidence / name
        return cv2.imread(str(path)) if path.is_file() else None

    samples = samples_from_records(records, load)
    if args.limit:
        wrong = [s for s in samples if not s.correct]
        samples = wrong + [s for s in samples if s.correct][: max(0, args.limit - len(wrong))]
    if not samples:
        sys.exit("no signed-off check with an evidence photo and reviewed compartments yet")
    advisors = {}
    for name in args.advisor or ["local"]:
        if name == "local":
            local = settings.local_advisor.model_copy(update={"model": args.model or settings.local_advisor.model})
            advisors[f"{local.model} (local)"] = LocalVisionAdvisor(local)
        else:
            advisors[settings.expert.model] = ClaudeExpert(settings.expert, api_key=load_api_key(db))
    wrong = sum(not s.correct for s in samples)
    print(f"{len(samples)} compartments from {len(records)} checks: {wrong} wrong, {len(samples) - wrong} correct")
    scores, rows = compare_advisors(advisors, samples, db.catalog())
    print(report(scores))
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as out:
            writer = csv.DictWriter(out, fieldnames=list(rows[0].model_dump()))
            writer.writeheader()
            writer.writerows(r.model_dump() for r in rows)
        print(f"every answer written to {args.csv}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="avavision", description="AvaVision pack-verification station")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--data", type=Path, default=default_data_dir(), help="station data folder")
    commands = parser.add_subparsers(dest="command", required=True)

    for name, help_text in (("serve", "run the station server"), ("desktop", "run the station in its own window")):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--demo", action="store_true", help="simulated camera and demo data")
        command.add_argument("--host", default="127.0.0.1")
        command.add_argument("--port", type=int, default=8765)

    commands.add_parser("fetch-models", help="download and verify the appearance embedder")
    benchmark = commands.add_parser("benchmark", help="time the pipeline on simulated frames")
    benchmark.add_argument("--frames", type=int, default=6)
    commands.add_parser("verify-audit", help="verify the audit hash chain")
    export = commands.add_parser("export-training", help="export confirmed pill images for training/")
    export.add_argument("out")
    evaluation = commands.add_parser("evaluate", help="evaluation report from holdout samples")
    evaluation.add_argument("samples")
    evaluation.add_argument("dataset_id")
    gate = commands.add_parser("gate", help="evaluate a detector model against the release gate")
    gate.add_argument("manifest")
    gate.add_argument("model")
    markers = commands.add_parser("markers", help="write the printable tray marker sheet")
    markers.add_argument("--out", default="avavision-markers.png")
    advisors = commands.add_parser(
        "compare-advisors", help="score second-opinion advisors on compartments pharmacists already checked"
    )
    advisors.add_argument("--advisor", action="append", choices=["local", "claude"], help="repeat to compare")
    advisors.add_argument("--model", help="local model to try instead of the one in settings")
    advisors.add_argument("--limit", type=int, default=0, help="every wrong compartment plus correct ones up to this")
    advisors.add_argument("--csv", help="write every answer to this file")

    args = parser.parse_args(argv)
    if args.command in ("serve", "desktop"):
        _serve(args, open_window=args.command == "desktop")
    elif args.command == "fetch-models":
        _fetch_models(args)
    elif args.command == "benchmark":
        _benchmark(args)
    elif args.command == "verify-audit":
        _verify_audit(args)
    elif args.command == "export-training":
        _export_training(args)
    elif args.command == "evaluate":
        _evaluate(args)
    elif args.command == "gate":
        _gate(args)
    elif args.command == "markers":
        _markers(args)
    elif args.command == "compare-advisors":
        _compare_advisors(args)


if __name__ == "__main__":
    main()
