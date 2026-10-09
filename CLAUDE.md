# AvaVision — guidance for coding agents

Webster-pak verification assistant for pharmacists (iOS). Founder-facing docs are Persian (`docs/*_FA.md`); app UI and code are English.

## Layout

- `Sources/AvaVisionCore` — all safety-relevant logic (Foundation only, builds on Linux).
- `Sources/AvaVisionImaging` — capture-quality metrics on grayscale images.
- `Sources/AvaVisionCore/Brain` — the learning brain (memory, open-set identification, calibration, trust ledger).
- `Sources/AvaVisionCLI` — `avavision` tool (model hash, holdout evaluation, gate, audit verification, embedder manifest, brain report).
- `training/` — offline Python pipeline: DINOv2 (Apache-2.0) metric learning → Core ML; CI in `.github/workflows/training.yml`.
- `App/` — SwiftUI app; the Xcode project is generated from `App/project.yml` with XcodeGen and is not committed.

## Commands

```bash
swift test                                                   # core tests (macOS or Linux)
swift format --in-place --recursive Package.swift Sources Tests App
swift format lint --strict --recursive Package.swift Sources Tests App   # CI fails on any finding
scripts/ios-test.sh                                          # Mac only: xcodegen + simulator tests
```

## Rules

- Safety decisions live in `AvaVisionCore` and must have tests. The app only gathers evidence and displays results.
- Never relax a fail-safe default (see `docs/SAFETY_RULES_FA.md`) without recording it in `docs/DECISIONS_FA.md`.
- Weak, missing or conflicting evidence must never produce `verified` or `countMatched`.
- No patient data, images, model weights or secrets in Git.
- The product is proprietary (see LICENSE). Do not add third-party Swift dependencies; in `training/` use only
  permissive licences and never AGPL or non-commercial weights (see `docs/DECISIONS_FA.md` D-116).
- Brain safety: the brain may only escalate until a medication earns trust in `TrustLedger`; it learns only from
  compartments a pharmacist inspected. Never feed automatically accepted results back as training data.
- App target compiles in Swift 5 language mode; packages use Swift 6.
