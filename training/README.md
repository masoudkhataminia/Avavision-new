# AvaVision training

Offline tools that turn pharmacist-confirmed pill images into AvaVision's own appearance embedder.
Persian guide: [`docs/TRAINING_GUIDE_FA.md`](../docs/TRAINING_GUIDE_FA.md).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Data: in the app, Brain › Export training data, then copy the folder from the Files app.
#    (Or make fake data to try the pipeline: python tools/make_synthetic_export.py /tmp/export)

# 2. Fine-tune (Apple Silicon uses the GPU via MPS automatically)
python finetune.py --data AvaVision-training-data --out runs/v1

# 3. Export to Core ML (on a Mac the export is checked against PyTorch)
python export_coreml.py --checkpoint runs/v1/embedder.pt --out build/AvaVisionEmbedder.mlpackage \
    --embedder-id avavision-dinov2s-v1

# 4. Install into the app (Mac): compiles, hashes, writes the manifest
../scripts/package-embedder.sh build/AvaVisionEmbedder.mlpackage avavision-dinov2s-v1 2026.10.1
```

Rules:

- Only permissively licensed weights and libraries (DINOv2 Apache-2.0, PyTorch BSD, pytorch-metric-learning MIT,
  coremltools BSD-3). Never AGPL (Ultralytics), never non-commercial or research-only weights
  (DINOv3 needs legal review; MobileCLIP and ConvNeXt V2 weights are excluded).
- Exported data contains single-pill crops only. Keep it on encrypted storage and out of Git.
- A new embedder ID makes every app re-learn its memory from stored crops and rebuild trust from zero.
