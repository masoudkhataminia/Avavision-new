"""Exports AvaVision's appearance embedder to Core ML.

    # Pretrained DINOv2 only (useful before any fine-tuning: a much stronger eye than the built-in one)
    python export_coreml.py --out build/AvaVisionEmbedder.mlpackage --embedder-id avavision-dinov2s-base

    # A fine-tuned checkpoint from finetune.py
    python export_coreml.py --checkpoint runs/embedder-v1/embedder.pt --out build/AvaVisionEmbedder.mlpackage \
        --embedder-id avavision-dinov2s-v1

Then, on a Mac, run ``scripts/package-embedder.sh build/AvaVisionEmbedder.mlpackage <embedder-id> <version>``
to compile it, hash it and install it with its manifest in the app.
Changing the embedder ID makes every installed app re-learn its memory from stored crops.
"""

from __future__ import annotations

import argparse
import sys

import coremltools as ct
import numpy as np
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", help="Checkpoint from finetune.py; omit to export the pretrained backbone")
    parser.add_argument("--out", required=True, help="Output .mlpackage")
    parser.add_argument("--embedder-id", required=True)
    parser.add_argument("--precision", choices=["float16", "float32"], default="float16")
    args = parser.parse_args()

    wrapper = ExportWrapper(build(args.checkpoint)).eval()
    example = torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE) * 255
    with torch.no_grad():
        reference = wrapper(example)
        traced = torch.jit.trace(wrapper, example)

    model = ct.convert(
        traced,
        inputs=[ct.ImageType(name="image", shape=(1, 3, IMAGE_SIZE, IMAGE_SIZE), color_layout=ct.colorlayout.RGB)],
        outputs=[ct.TensorType(name="embedding")],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16 if args.precision == "float16" else ct.precision.FLOAT32,
    )
    model.author = "AvaVision"
    model.license = f"Proprietary. Base weights: {BACKBONE_LICENSE}"
    model.short_description = f"AvaVision pill appearance embedder ({args.embedder_id})"
    model.user_defined_metadata["embedderID"] = args.embedder_id
    model.save(args.out)

    norm = float(np.linalg.norm(reference.numpy()))
    print(f"Saved {args.out}: embedding size {reference.shape[-1]}, reference norm {norm:.4f}")

    if sys.platform == "darwin":
        verify(model, wrapper)


def verify(model: ct.models.MLModel, wrapper: ExportWrapper) -> None:
    """Core ML must reproduce PyTorch's embedding (cosine similarity ≥ 0.99), or the export is rejected."""
    from PIL import Image

    pixels = (np.random.default_rng(0).random((IMAGE_SIZE, IMAGE_SIZE, 3)) * 255).astype(np.uint8)
    image = Image.fromarray(pixels, "RGB")
    with torch.no_grad():
        expected = wrapper(torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).float()).numpy().ravel()
    actual = np.asarray(model.predict({"image": image})["embedding"]).ravel()
    cosine = float(expected @ actual / (np.linalg.norm(expected) * np.linalg.norm(actual)))
    print(f"Core ML vs PyTorch cosine similarity: {cosine:.5f}")
    if cosine < 0.99:
        raise SystemExit("Core ML output does not match PyTorch; do not ship this model.")


if __name__ == "__main__":
    main()
