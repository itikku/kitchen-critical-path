"""Simulated cooking sessions: the scheduler versus the plain recipe order.

Both policies cook with the same random task durations in each session (common random numbers).
Active-task durations vary around the graph value; some tasks get an interruption. Passive tasks
(machines and timers) take exactly their graph duration. Values are heuristics, not measurements."""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass
from typing import Literal

from kcp.scheduler import build_state, recommend
from kcp.schema import Event, TaskGraph

Policy = Literal["scheduler", "recipe_order"]
IDLE_TICK_S = 10.0
DURATION_SIGMA = 0.25  # log-normal spread of active-task durations
INTERRUPT_P = 0.15  # chance that an active task is interrupted
INTERRUPT_S = (60.0, 300.0)


@dataclass
class SessionResult:
    total_s: float
    idle_hands_s: float
    window_misses: int
    events: list[Event]


def sample_durations(graph: TaskGraph, rng: random.Random) -> dict[str, float]:
    out = {}
    for t in graph.tasks:
        d = float(t.duration_s)
        if t.attention == "active":
            d *= rng.lognormvariate(0.0, DURATION_SIGMA)
            if rng.random() < INTERRUPT_P:
                d += rng.uniform(*INTERRUPT_S)
        out[t.task_id] = d
    return out


def run_session(
    graph: TaskGraph, policy: Policy, durations: dict[str, float], session_id: str = "sim"
) -> SessionResult:
    events: list[Event] = []

    def emit(task_id: str, status: Literal["started", "complete"], t: float) -> None:
        events.append(
            Event(
                event_id=f"{session_id}-e{len(events):03d}",
                session_id=session_id,
                t_start_s=t,
                task_id=task_id,
                status=status,
                source="simulated",
            )
        )

    t, idle = 0.0, 0.0
    busy: str | None = None
    busy_until = 0.0
    pointer = 0
    final = graph.final_task
    while True:
        if busy is not None and t >= busy_until:
            emit(busy, "complete", busy_until)
            busy = None
        state = build_state(graph, events, t)
        if state.tasks[final].status == "complete":
            break
        if busy is None:
            started_active = None
            if policy == "scheduler":
                rec = recommend(graph, state, t)
                for p in rec.start_passive:
                    emit(p, "started", t)
                if rec.ranked_active:
                    started_active = rec.ranked_active[0]
            else:
                while pointer < len(graph.recipe_order):
                    task_id = graph.recipe_order[pointer]
                    task = graph.task(task_id)
                    if any(state.tasks[r].status != "complete" for r in task.requires):
                        break  # blocked: the recipe says this step is next
                    pointer += 1
                    if task.attention == "passive":
                        emit(task_id, "started", t)
                        continue
                    started_active = task_id
                    break
            if started_active is not None:
                emit(started_active, "started", t)
                busy, busy_until = started_active, t + durations[started_active]
                t = busy_until
                continue
            idle += IDLE_TICK_S
        t = busy_until if busy is not None else t + IDLE_TICK_S
    final_state = build_state(graph, events, t)
    done_at = final_state.tasks[final].completed_at
    assert done_at is not None
    misses = sum(1 for _, w in final_state.warnings if w.startswith("Missed quality window"))
    return SessionResult(total_s=done_at, idle_hands_s=idle, window_misses=misses, events=events)


@dataclass
class PolicySummary:
    mean_total_min: float
    p90_total_min: float
    mean_idle_min: float
    sessions_with_window_miss: float  # fraction


@dataclass
class Comparison:
    graph_id: str
    sessions: int
    seed: int
    scheduler: PolicySummary
    recipe_order: PolicySummary
    mean_saving_min: float  # recipe_order total - scheduler total, paired
    scheduler_not_slower: float  # fraction of sessions


def _summary(results: list[SessionResult]) -> PolicySummary:
    totals = sorted(r.total_s / 60 for r in results)
    return PolicySummary(
        mean_total_min=statistics.fmean(totals),
        p90_total_min=totals[int(0.9 * (len(totals) - 1))],
        mean_idle_min=statistics.fmean(r.idle_hands_s / 60 for r in results),
        sessions_with_window_miss=sum(r.window_misses > 0 for r in results) / len(results),
    )


def compare(graph: TaskGraph, sessions: int, seed: int) -> Comparison:
    sched, recipe = [], []
    for i in range(sessions):
        durations = sample_durations(graph, random.Random(seed * 100_003 + i))
        sched.append(run_session(graph, "scheduler", durations, f"sched-{i}"))
        recipe.append(run_session(graph, "recipe_order", durations, f"recipe-{i}"))
    savings = [(r.total_s - s.total_s) / 60 for s, r in zip(sched, recipe, strict=True)]
    return Comparison(
        graph_id=graph.graph_id,
        sessions=sessions,
        seed=seed,
        scheduler=_summary(sched),
        recipe_order=_summary(recipe),
        mean_saving_min=statistics.fmean(savings),
        scheduler_not_slower=sum(x >= -1e-9 for x in savings) / sessions,
    )


def format_comparison(c: Comparison) -> str:
    rows = [
        f"Simulation: {c.graph_id} — {c.sessions} sessions, seed {c.seed}",
        f"{'policy':14s} {'mean total':>11s} {'p90 total':>10s} {'idle hands':>11s} "
        f"{'window miss':>12s}",
    ]
    for name, s in (("scheduler", c.scheduler), ("recipe order", c.recipe_order)):
        rows.append(
            f"{name:14s} {s.mean_total_min:8.1f} min {s.p90_total_min:7.1f} min "
            f"{s.mean_idle_min:8.1f} min {s.sessions_with_window_miss:11.0%}"
        )
    rows.append(
        f"paired saving: {c.mean_saving_min:.1f} min per session; "
        f"scheduler not slower in {c.scheduler_not_slower:.0%} of sessions"
    )
    return "\n".join(rows) + "\n"
