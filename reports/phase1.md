# Phase 1 — Scheduler, Replay, and Simulation

Date: 2026-11-01. Reproduce with the commands at the end. All numbers come from simulated sessions; no real cooking was measured.

## What was built

- A typed recipe task graph (`data/graphs/`), with two rice variants: frozen rice reheated, and Instant Pot.
- A deterministic scheduler (`src/kcp/scheduler.py`). Rules: prerequisites first; one active task at a time; start passive tasks first; hold a task until its quality window opens (just in time for passive tasks); rank active tasks by the longest remaining path; ask to confirm low-confidence states.
- A replay tool that shows, after each event, the next task, the reason, held tasks, warnings, the plating estimate, and the bottleneck.
- A session simulator that compares the scheduler with the plain recipe order.

## Simulation method

- 200 sessions per graph variant, seed 0. Both policies cook with the same random durations in each session (paired comparison).
- Active tasks: duration × log-normal noise (σ = 0.25); 15% of active tasks get an interruption of 1–5 min. Passive tasks (timers, machines) take exactly their graph duration.
- **Baseline ("recipe order"):** the cook follows the steps in the order a typical written recipe lists them, and starts the next step during passive waits (for example, cuts vegetables while the tofu presses). This is a fair baseline, not a cook who waits idly.
- Durations, windows, and noise are authored heuristics.

## Results

| Variant | Policy | Mean total | p90 total | Idle hands | Sessions with a window miss |
|---|---|---:|---:|---:|---:|
| Frozen rice | scheduler | 55.7 min | 63.6 min | 3.9 min | 46% |
| Frozen rice | recipe order | 57.8 min | 65.7 min | 6.0 min | 96% |
| Instant Pot | scheduler | 57.2 min | 64.5 min | 3.1 min | 34% |
| Instant Pot | recipe order | 57.1 min | 64.5 min | 3.0 min | 78% |

Paired time saving: **2.1 min per session** with frozen rice (the scheduler is never slower); **−0.1 min** with the Instant Pot (not slower in 91% of sessions).

Window misses by task (sessions out of 200; median overshoot past the window):

| Variant | Task | Scheduler | Recipe order |
|---|---|---:|---:|
| Frozen rice | cut the tomato | 93 (2 min) | 191 (5 min) |
| Instant Pot | cut the tomato | 69 (2 min) | 138 (3 min) |
| Instant Pot | cook the rice | **0** | 120 (5 min) |

## What this means

1. **The value is mostly quality, not speed.** Total time improves a little (frozen rice) or not at all (Instant Pot). The main gain is timing: the rice finishes when it is needed, and the tomato is cut later.
2. **Speed gain comes from overlapping the rice reheat** with the last wok steps, instead of reheating after cooking.
3. **The remaining tomato misses are structural.** The tomato must be cut before the wok starts (all prep before the wok), and the wok phase alone is about 19 min. When the wok phase runs long, the tomato misses its 25-min window by about 2 min. A looser window would remove these misses, but the window was set before the run and was not tuned.
4. **Instant Pot just-in-time has a small cost:** in 9% of sessions the cook finishes before the rice and waits briefly. The scheduler uses nominal durations; a faster cook finishes early.

Product implication (Risk 2 in the plan): the pitch should lead with **"the right thing at the right time"** (fresh rice, late-cut tomato, no idle wok) and with recovery from mistakes (late rice, a wok started too early), more than with minutes saved.

## Replay fixtures

Golden outputs in `tests/expected/` cover: on-time sessions for both variants, late rice, a wok started before prep is done, a low-confidence camera event, a user correction, and an early-cut tomato.

## Reproduce

```sh
uv sync
uv run pytest
uv run kcp simulate --graph data/graphs/tofu_stir_fry.frozen_rice.v0.json --sessions 200 --seed 0
uv run kcp simulate --graph data/graphs/tofu_stir_fry.instant_pot.v0.json --sessions 200 --seed 0
uv run kcp replay tests/fixtures/sessions/ip_late_rice.jsonl --graph data/graphs/tofu_stir_fry.instant_pot.v0.json
```
