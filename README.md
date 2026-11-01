# Kitchen Critical Path

**Mise en place is a compiler.** A critical-path planner for cooking that watches what has actually happened, tracks dependencies and quality windows, and recommends the next best task.

Target hardware: NVIDIA Jetson Orin Nano Super (8 GB). Current status: the planner and the first PyTorch perception model run offline on a Mac. Jetson deployment is next.

## What it does

Given a structured recipe and the events of a cooking session, the planner tells you what to do next and why:

```text
[17:00] complete 'Cut the leafy vegetables'
        next: Cook the rice in the Instant Pot — Start it now: it runs by itself for 30 min while you do other work.
        hold: Cut the tomato — wait: start when 'plate the food' is ~25 min away (now ~30 min)
        plate ≈ 49:00 · bottleneck: cook the rice in the instant pot (30 min left)
```

It is not a recipe chatbot or a timer. It replans from what actually happened: a late start, a step done out of order, or a correction.

## Quick start

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run pytest                      # 36 tests
uv run kcp replay tests/fixtures/sessions/ip_late_rice.jsonl \
    --graph data/graphs/tofu_stir_fry.instant_pot.v0.json
uv run kcp simulate --graph data/graphs/tofu_stir_fry.frozen_rice.v0.json --sessions 200
```

## How it works

```text
camera (prep zone) ──► PyTorch action model ──► action events ─┐
voice / button ────────────────────────────────► task events ──┼─► scheduler ──► next task + reason
recipe task graph (dependencies, durations, quality windows) ──┘
```

**Scheduler rules** (`src/kcp/scheduler.py`): a task needs its prerequisites; one active task at a time; start passive tasks (timers, machines) first; hold a task until its quality window opens, just in time for passive tasks; rank the rest by the longest remaining path; ask to confirm a low-confidence state.

**Perception** (`src/kcp/perception/`): a frozen, ImageNet-pretrained MobileNetV3-Large encodes frames once; a small PyTorch temporal head classifies 8-frame windows (~4 s) into 8 prep-zone actions (cut, peel, pour, stir, transfer, take object, background, other). It trains on [Breakfast Actions](https://serre.lab.brown.edu/breakfast-actions-dataset.html) (CC BY 4.0).

## Results so far

| Result | Value | Details |
|---|---|---|
| Simulated sessions, frozen-rice variant | 2.1 min faster; window-miss sessions 96% → 46% | `reports/phase1.md` |
| Simulated sessions, Instant Pot variant | same total time; rice-freshness misses 60% → 0% | `reports/phase1.md` |
| Action model (8 classes), unseen people / unseen camera | macro-F1 0.20 / 0.20 (frame baseline 0.17 / 0.11; majority 0.09) | `reports/phase2a.md` |

## Reproduce the perception results

```sh
uv sync --extra train
# Download Breakfast Actions (see docs/data_provenance.md) into data/raw/breakfast/
uv run kcp-train embed      # frame embeddings, cached in data/cache/ (~1 h on an M1)
uv run kcp-train train      # frame baseline + temporal head
uv run kcp-train evaluate   # writes reports/phase2a.md
```

## Limits

- Durations and quality windows are authored heuristics, not measurements.
- Public training data shows breakfast prep (fruit, bread), not stir-fry vegetables. The camera detects *that* you are cutting, not *what*; the planner infers the task from the recipe graph.
- The action model is weak in absolute terms (macro-F1 0.20), and weakest on `cut`. See the next steps in `reports/phase2a.md`.
- Simulated results are not real cooking measurements. No result yet comes from the developer's own kitchen.
- Food safety is out of scope: the system never judges doneness, temperature, or freshness for safety.

## Repository layout

```text
src/kcp/            scheduler, replay, simulator, schemas, CLI
src/kcp/perception/ embeddings, training, evaluation (PyTorch)
src/kcp/adapters/   Breakfast Actions -> action events
data/graphs/        recipe task graphs
docs/               label mapping, data provenance, privacy plan
reports/            generated results
tests/              unit tests, fixtures, golden replays
```

## Data and citation

Breakfast Actions: H. Kuehne, A. B. Arslan, T. Serre, "The Language of Actions: Recovering the Syntax and Semantics of Goal-Directed Human Activities," CVPR 2014. No video is stored in this repository. See `docs/data_provenance.md`.
