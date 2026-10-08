# Bundled detection model

Place exactly these two files here before building (they are not committed to Git):

- `AvaVisionDetector.mlmodelc/` — the compiled Core ML object-detection model
- `AvaVisionDetector.json` — its `ModelManifest` (stage, SHA-256, label meanings, holdout evaluation)

Without them the app runs with automatic checking off. See `docs/MODEL_GUIDE_FA.md`.
