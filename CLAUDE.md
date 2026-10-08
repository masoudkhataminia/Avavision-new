# AvaVision — guidance for coding agents

Webster-pak verification assistant for pharmacists (iOS). Founder-facing docs are Persian (`docs/*_FA.md`); app UI and code are English.

## Layout

- `Sources/AvaVisionCore` — all safety-relevant logic (Foundation + swift-crypto only, builds on Linux).
- `Sources/AvaVisionImaging` — capture-quality metrics on grayscale images.
- `Sources/AvaVisionCLI` — `avavision` tool (model hash, holdout evaluation, gate, audit verification).
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
- Do not add AGPL dependencies (e.g. Ultralytics code); the product is commercial.
- App target compiles in Swift 5 language mode; packages use Swift 6.
