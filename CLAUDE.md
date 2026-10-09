# AvaVision — guidance for coding agents

Webster-pak verification station for pharmacists, on Windows with a fixed camera. Founder-facing docs are
Persian (`docs/*_FA.md`); the interface and code are English.

Start with `docs/PROJECT_OVERVIEW_FA.md`: the whole project, its history, every decision, the current state and the
open items in one place. The founder's machine keeps the build chat, photos and raw history in `data/project-archive/`
(git-ignored; never commit anything from it).

## Layout

- `src/avavision/core` — all safety-relevant logic: decision engine, model gate, sign-off, audit chain,
  advisory opinions. Pure Python + pydantic; no I/O.
- `src/avavision/brain` — the learning brain (memory, open-set identification, calibration, trust ledger).
- `src/avavision/vision` — camera, pack finding (ArUco), quality, segmentation, ONNX Runtime embedder/detector,
  synthetic station renderer.
- `src/avavision/expert` — the Claude expert (chart reading, explanations, second opinions). Advisory only.
- `src/avavision/storage` — SQLite (WAL) and image folders.
- `src/avavision/station` — station service, FastAPI app (localhost), demo camera.
- `web/` — React + TypeScript interface, built into `src/avavision/station/ui` (git-ignored).
- `training/` — offline PyTorch fine-tuning of the embedder → ONNX.
- `packaging/` — PyInstaller build of the Windows app.

## Commands

```bash
pip install -e ".[dev]"
pytest -q
ruff check . && ruff format --check .        # CI fails on any finding
cd web && npm ci && npm run build            # typecheck + build the interface
avavision serve --demo                       # station with the simulated camera on http://127.0.0.1:8765
```

## Rules

- Safety decisions live in `src/avavision/core` (and the brain's trust rules) and must have tests. The station,
  interface and expert only gather evidence and display results.
- Never relax a fail-safe default (see `docs/SAFETY_RULES_FA.md`) without recording it in `docs/DECISIONS_FA.md`.
- Weak, missing or conflicting evidence must never produce `verified` or `countMatched`.
- The language-model expert is never in the acceptance path: its opinions can only escalate (`core/advisory.py`).
- Brain safety: the brain may only escalate until a medication earns trust in `TrustLedger`; it learns only from
  compartments a pharmacist inspected. Never feed automatically accepted results back as training data.
- No secrets, API keys, images or model weights in Git.
- The product is proprietary (see LICENSE). Only permissively licensed dependencies and weights (MIT / BSD /
  Apache-2.0); never AGPL (e.g. Ultralytics) or non-commercial weights (see `docs/DECISIONS_FA.md` D-116).
- Claude API code uses the official `anthropic` SDK, model `claude-opus-5-5`, explicit `effort`, structured
  outputs, refusal handling and server-side fallbacks.
