# AvaVision training

Offline tools that turn pharmacist-confirmed pill images into AvaVision's own appearance embedder.
Persian guide: [`docs/TRAINING_GUIDE_FA.md`](../docs/TRAINING_GUIDE_FA.md).

```bash
python -m venv .venv && .venv\Scripts\activate        # Linux: source .venv/bin/activate
pip install -r requirements.txt                        # NVIDIA: install the CUDA build of torch first

# 1. Data: on the station, export what the brain learned from pharmacist-confirmed checks
avavision export-training D:\AvaVision-training-data
#    (or fake data to try the pipeline: python tools/make_synthetic_export.py D:\fake-export)

# 2. Fine-tune (uses the NVIDIA GPU automatically when present)
python finetune.py --data D:\AvaVision-training-data --out runs\v1

# 3. Export to ONNX; the export is rejected unless ONNX Runtime reproduces PyTorch
python export_onnx.py --checkpoint runs\v1\embedder.pt --out build --embedder-id avavision-dinov2s-v1 --version 2026.10.1

# 4. Install: copy build\embedder.onnx and build\embedder.json into the station's models folder
#    (%LOCALAPPDATA%\AvaVision\models) and restart the station.
```

Rules:

- Only permissively licensed weights and libraries (DINOv2 Apache-2.0, PyTorch BSD, pytorch-metric-learning MIT,
  ONNX Apache-2.0). Never AGPL (Ultralytics), never non-commercial or research-only weights
  (DINOv3 needs legal review; MobileCLIP and ConvNeXt V2 weights are excluded).
- Exported data contains single-pill crops only. Keep it out of Git.
- A new embedder ID makes the station re-learn its memory from stored crops and rebuild trust from zero.
