"""Exports AvaVision's appearance embedder to ONNX for the Windows station.

    # Pretrained DINOv2 only (before any fine-tuning)
    python export_onnx.py --out build --embedder-id avavision-dinov2s-base --version 2026.10.1

    # A fine-tuned checkpoint from finetune.py
    python export_onnx.py --checkpoint runs/v1/embedder.pt --out build --embedder-id avavision-dinov2s-v1 \
        --version 2026.10.1

Writes ``embedder.onnx`` and its manifest ``embedder.json`` (with the SHA-256 the station verifies). Copy both
into the station's ``models`` folder. A new embedder ID makes the station re-learn its memory from stored crops
and rebuild trust from zero. The export is rejected unless ONNX Runtime reproduces PyTorch's embeddings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from avavision_training.data import IMAGE_SIZE
from avavision_training.model import BACKBONE_LICENSE, DEFAULT_BACKBONE, ExportWrapper, PillEmbedder


def build(checkpoint: str | None) -> PillEmbedder:
    if checkpoint is None:
        return PillEmbedder(DEFAULT_BACKBONE, embedding_size=None)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = PillEmbedder(state["backbone"], state["embedding_size"])
    model.load_state_dict(state["state_dict"])
    return model


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, wrapper: ExportWrapper) -> float:
    """Lowest cosine similarity between ONNX Runtime and PyTorch over a few random batches."""
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    worst = 1.0
    for batch in (1, 5):
        pixels = (rng.random((batch, 3, IMAGE_SIZE, IMAGE_SIZE)) * 255).astype(np.float32)
        with torch.no_grad():
            expected = wrapper(torch.from_numpy(pixels)).numpy()
        actual = session.run(["embedding"], {"image": pixels})[0]
        cosines = (expected * actual).sum(1) / (np.linalg.norm(expected, axis=1) * np.linalg.norm(actual, axis=1))
        worst = min(worst, float(cosines.min()))
    return worst


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", help="Checkpoint from finetune.py; omit to export the pretrained backbone")
    parser.add_argument("--out", required=True, help="Output folder")
    parser.add_argument("--embedder-id", required=True)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "embedder.onnx"
    wrapper = ExportWrapper(build(args.checkpoint)).eval()
    example = torch.rand(2, 3, IMAGE_SIZE, IMAGE_SIZE) * 255
    torch.onnx.export(
        wrapper,
        (example,),
        str(path),
        input_names=["image"],
        output_names=["embedding"],
        dynamic_axes={"image": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    cosine = verify(path, wrapper)
    print(f"ONNX Runtime vs PyTorch: lowest cosine similarity {cosine:.6f}")
    if cosine < 0.999:
        path.unlink()
        raise SystemExit("ONNX output does not match PyTorch; do not ship this model.")

    manifest = {
        "schema_version": 1,
        "embedder_id": args.embedder_id,
        "version": args.version,
        "model_sha256": sha256(path),
        "input_size": IMAGE_SIZE,
        "output_name": "embedding",
        "base_model_license": BACKBONE_LICENSE,
        "notes": f"AvaVision appearance embedder {args.embedder_id}"
        + (" (fine-tuned)" if args.checkpoint else " (pretrained backbone)"),
    }
    (out / "embedder.json").write_text(json.dumps(manifest, indent=2))
    print(f"Saved {path} and embedder.json ({manifest['model_sha256'][:12]}…)")


if __name__ == "__main__":
    main()
