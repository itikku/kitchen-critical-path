# Project State — Kitchen Critical Path

Last updated: 2026-11-01

## Current status

- Phase 0: complete except 0.5 (phone placement, user) and 0.7 (Jetson inventory, blocked). See the table below.
- **Phase 1: complete.** All acceptance criteria pass (see "Verification").
- **Phase 2a: in progress.** Code complete and smoke-tested. The full embedding run is in progress (~1,389 videos; started 06:37). Then: train, evaluate, write `reports/phase2a.md`, delete the extracted videos (keep the 3.7 GB archive, per the user).
- Git initialized; **no commits** (the user has not asked for any).

## Phase 0 results

| Step | Status | Output |
|---|---|---|
| 0.1 Task graph | Done. Rice times confirmed by the user; other durations are authored heuristics. | `data/graphs/` |
| 0.2 Breakfast labels | Done. 48 coarse labels, 52 persons, 1,712 annotation files. Checksums match. | `docs/breakfast_labels.md` |
| 0.3 Label mapping | Done. 8 camera labels. `wash_greens` and `press_tofu` use user input (no public data). | `docs/label_mapping.md` |
| 0.4 License and citation | Done. CC BY 4.0 (dataset card). | `docs/data_provenance.md` |
| 0.4b 50 Salads | Blocked: official server down; mirror has features only. Substitute: T2. | `docs/data_provenance.md` |
| 0.5 Phone placement | Open (user). Needed before Phase 3 only. | — |
| 0.6 Privacy plan | Done. | `docs/privacy.md` |
| 0.7 Jetson inventory | Blocked until the Jetson setup. | `docs/jetson_inventory.md` |
| 0.8 AI-review design | Approved. | `IMPLEMENTATION_PLAN.md` |

## Phase 1 results (`reports/phase1.md`)

- Frozen rice: scheduler 2.1 min faster per session (never slower); sessions with a window miss 96% → 46%.
- Instant Pot: same total time (−0.1 min); rice-freshness misses 60% → 0%; any-miss sessions 78% → 34%.
- All remaining scheduler misses are the tomato (median overshoot 2 min). Structural: the tomato is cut before the ~19-min wok phase. The window was not tuned after the run.
- Product implication: lead the pitch with timing quality and recovery, not minutes saved (Risk 2).

## Verification (2026-11-01)

```text
uv run ruff format --check .   -> all files formatted
uv run ruff check .            -> All checks passed
uv run mypy                    -> no issues in 15 source files
uv run pytest                  -> 35 passed
```

## Implementation notes

- The `.avi.labels` files inside the video archive contain two extra labels (`walk_in`, `walk_out`). The pipeline uses the official coarse annotation files instead.
- Breakfast camera IDs do not mean the same position in each kitchen. T2 = camera ID `webcam02`, never used in training.
- Embedding speed on the M1: ~135 frames/s at native 320×240 (MPS, fp16, fixed batch shape). Embeddings verified identical (cosine 1.0) before and after the speed fix.
- Some Breakfast videos are 320×180; the encoder accepts any size.

## Decisions

- Recipe family: tofu-vegetable stir-fry, two rice variants. Vegetables grouped by cut and cook time. Tofu pressed.
- Camera covers the prep zone only. Stove, wok, sink, and tofu-press events come from user input: voice first, buttons as fallback (Phase 2b/3).
- Public training data: Breakfast Actions (CC BY 4.0).
- Scale principle: no per-kitchen human labels.
- v1 tests: T1 (new people), T2 (new people + unseen camera ID), simulated sessions.
- Phase 3: a 3-dinner human test set, used once to validate the label-free signals.

## Disk use

- Archive `data/raw/breakfast/BreakfastII_15fps_qvga_sync.tar.gz`: 3.7 GB, **keep** (user decision).
- Extracted videos: 3.8 GB, **delete after the embedding run**.
- Embedding cache: ~1 GB. Python environment: ~1 GB.

## Blockers

- Jetson setup (USB drive + adapter). Affects 0.7 and Phase 2b.
- 50 Salads server down. Affects only the cross-dataset test.

## Next step

Finish Phase 2a (train, evaluate, report). Then the user reviews `reports/phase1.md` and `reports/phase2a.md`.
