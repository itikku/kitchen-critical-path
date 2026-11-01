"""Deterministic critical-path scheduler.

Rules (IMPLEMENTATION_PLAN.md, "Scheduling rules"):
1. A task is eligible when all its prerequisites are complete.
2. The cook does one active task at a time; passive tasks run in the background.
3. Start eligible passive tasks first.
4. Defer a task whose quality window has not opened yet.
5. Rank the remaining active tasks by the longest remaining path to the final task.
6. If a task's state is unknown, ask the cook to confirm it before relying on it. A completion
   reported with low confidence makes the state unknown; a low-confidence start is accepted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from kcp.schema import Event, TaskGraph

LOW_CONFIDENCE = 0.6
JIT_MARGIN_S = 60  # a windowed passive task starts when it would finish ~1 min before its anchor
Status = Literal["pending", "started", "complete", "unknown"]


@dataclass
class TaskState:
    status: Status = "pending"
    started_at: float | None = None
    completed_at: float | None = None


@dataclass
class SessionState:
    tasks: dict[str, TaskState]
    warnings: list[tuple[float, str]] = field(default_factory=list)


def fmt(seconds: float) -> str:
    sign = "-" if seconds < 0 else ""
    s = int(round(abs(seconds)))
    return f"{sign}{s // 60:02d}:{s % 60:02d}"


def build_state(graph: TaskGraph, events: list[Event], now: float) -> SessionState:
    """Fold events up to `now` into task states. Corrected events are ignored.
    Passive tasks complete by themselves when their duration has passed."""
    corrected = {e.corrects_event_id for e in events if e.corrects_event_id}
    state = SessionState(tasks={t.task_id: TaskState() for t in graph.tasks})
    timeline = sorted(
        (e for e in events if e.event_id not in corrected and e.t_start_s <= now),
        key=lambda e: e.t_start_s,
    )
    for e in timeline:
        _auto_complete(graph, state, e.t_start_s)
        _apply(graph, state, e)
    _auto_complete(graph, state, now)
    return state


def _auto_complete(graph: TaskGraph, state: SessionState, now: float) -> None:
    for t in graph.tasks:
        ts = state.tasks[t.task_id]
        if t.attention == "passive" and ts.status == "started" and ts.started_at is not None:
            if now - ts.started_at >= t.duration_s:
                ts.status, ts.completed_at = "complete", ts.started_at + t.duration_s


def _apply(graph: TaskGraph, state: SessionState, e: Event) -> None:
    task = graph.task(e.task_id)
    ts = state.tasks[e.task_id]
    if e.confidence is not None and e.confidence < LOW_CONFIDENCE and e.status == "complete":
        ts.status = "unknown"
        return
    if e.status == "unknown":
        ts.status = "unknown"
        return
    if ts.status == "pending":
        missing = [r for r in task.requires if state.tasks[r].status != "complete"]
        if missing:
            names = ", ".join(graph.task(r).display_name.lower() for r in missing)
            state.warnings.append((e.t_start_s, f"{task.display_name} started before: {names}."))
    if ts.started_at is None:
        ts.started_at = e.t_start_s
    if e.status == "started":
        ts.status = "started"
    else:
        ts.status = "complete"
        ts.completed_at = e.t_end_s if e.t_end_s is not None else e.t_start_s
    _check_windows(graph, state, e)


def _check_windows(graph: TaskGraph, state: SessionState, e: Event) -> None:
    """When an anchor task starts, report tasks that started too early for their window."""
    anchor_start = state.tasks[e.task_id].started_at
    if anchor_start != e.t_start_s:
        return
    for t in graph.tasks:
        w = t.window
        started = state.tasks[t.task_id].started_at
        if w and w.anchor == e.task_id and started is not None:
            lead = anchor_start - started
            if lead > w.max_lead_s:
                anchor = graph.task(w.anchor).display_name.lower()
                state.warnings.append(
                    (
                        e.t_start_s,
                        f"Missed quality window: {t.display_name.lower()} started "
                        f"{lead / 60:.0f} min before '{anchor}'; "
                        f"best within {w.max_lead_s / 60:.0f} min.",
                    )
                )


def remaining_s(graph: TaskGraph, state: SessionState, task_id: str, now: float) -> float:
    ts = state.tasks[task_id]
    duration = graph.task(task_id).duration_s
    if ts.status == "complete":
        return 0.0
    if ts.status == "started" and ts.started_at is not None:
        return max(0.0, duration - (now - ts.started_at))
    return float(duration)


def earliest_finish(graph: TaskGraph, state: SessionState, now: float) -> dict[str, float]:
    """Earliest finish time of every task, relative to now, ignoring the one-pair-of-hands limit."""
    ef: dict[str, float] = {}
    for tid in graph.topological_order():
        ts = state.tasks[tid]
        before = (
            0.0
            if ts.status == "started"
            else max((ef[r] for r in graph.task(tid).requires), default=0.0)
        )
        ef[tid] = before + remaining_s(graph, state, tid, now)
    return ef


def path_to_end_s(graph: TaskGraph, state: SessionState, task_id: str, now: float) -> float:
    """Longest chain of remaining work from this task to the final task (inclusive)."""
    tail = max((path_to_end_s(graph, state, s, now) for s in graph.successors(task_id)), default=0)
    return remaining_s(graph, state, task_id, now) + tail


@dataclass
class StartEstimate:
    seconds: float  # from now until the task can start
    bottleneck: str


def estimate_start(
    graph: TaskGraph, state: SessionState, task_id: str, now: float
) -> StartEstimate:
    """Time until `task_id` can start: the larger of the dependency chain and the active work
    that the cook's hands must still do before it."""
    ef = earliest_finish(graph, state, now)
    requires = graph.task(task_id).requires
    chain = max((ef[r] for r in requires), default=0.0)
    hands = sum(
        remaining_s(graph, state, a, now)
        for a in graph.ancestors(task_id)
        if graph.task(a).attention == "active"
    )
    if hands >= chain:
        return StartEstimate(hands, f"your hands ({hands / 60:.0f} min of active work left)")
    # Walk the critical chain back to the first task that is running or ready to run.
    critical = max(requires, key=lambda r: ef[r])
    while state.tasks[critical].status in ("pending", "unknown"):
        open_preds = [
            p for p in graph.task(critical).requires if state.tasks[p].status != "complete"
        ]
        if not open_preds:
            break
        critical = max(open_preds, key=lambda r: ef[r])
    name = graph.task(critical).display_name
    return StartEstimate(chain, f"{name.lower()} ({ef[critical] / 60:.0f} min left)")


@dataclass
class Recommendation:
    now: float
    next_task: str | None
    reason: str
    start_passive: list[str]
    ranked_active: list[str]
    waiting: list[tuple[str, str]]
    confirm: list[str]
    plate_eta: float  # absolute session time when the final task is expected to finish
    bottleneck: str
    busy_with: str | None


def recommend(graph: TaskGraph, state: SessionState, now: float) -> Recommendation:
    final = graph.final_task
    tasks = state.tasks
    confirm = [
        f"Is '{graph.task(t).display_name.lower()}' done?"
        for t in graph.recipe_order
        if tasks[t].status == "unknown"
    ]
    eligible = [
        t
        for t in graph.recipe_order
        if tasks[t].status == "pending"
        and all(tasks[r].status == "complete" for r in graph.task(t).requires)
    ]
    ready: list[str] = []
    waiting: list[tuple[str, str]] = []
    for t in eligible:
        w = graph.task(t).window
        if w is None:
            ready.append(t)
            continue
        lead = estimate_start(graph, state, w.anchor, now).seconds
        task = graph.task(t)
        limit = w.max_lead_s
        if task.attention == "passive":
            limit = min(limit, task.duration_s + JIT_MARGIN_S)  # just in time
        if lead > limit:
            waiting.append(
                (
                    t,
                    f"wait: start when '{graph.task(w.anchor).display_name.lower()}' is "
                    f"~{limit / 60:.0f} min away (now ~{lead / 60:.0f} min)",
                )
            )
        else:
            ready.append(t)

    in_progress = [t for t in graph.recipe_order if tasks[t].status == "started"]
    busy_with = next((t for t in in_progress if graph.task(t).attention == "active"), None)
    if not ready and not in_progress and waiting:
        # Deadlock guard: nothing else can move, so the window cannot be met by waiting.
        forced = waiting[0][0]
        waiting = [tw for tw in waiting if tw[0] != forced]
        ready.append(forced)

    start_passive = [t for t in ready if graph.task(t).attention == "passive"]
    order = {t: i for i, t in enumerate(graph.recipe_order)}
    ranked_active = sorted(
        (t for t in ready if graph.task(t).attention == "active"),
        # Windowed tasks go last among equals: doing them later keeps them fresher.
        key=lambda t: (
            graph.task(t).window is not None,
            -path_to_end_s(graph, state, t, now),
            order[t],
        ),
    )

    est = estimate_start(graph, state, final, now)
    plate_eta = now + est.seconds + remaining_s(graph, state, final, now)

    next_task, reason = None, "Nothing to start now."
    if tasks[final].status == "complete":
        reason = "Done. Enjoy the meal."
    elif start_passive:
        next_task = start_passive[0]
        d = graph.task(next_task).duration_s / 60
        reason = f"Start it now: it runs by itself for {d:.0f} min while you do other work."
    elif ranked_active:
        next_task = ranked_active[0]
        p = path_to_end_s(graph, state, next_task, now) / 60
        reason = f"It is on the longest remaining path ({p:.0f} min of work to the end)."
        if busy_with:
            reason = f"After '{graph.task(busy_with).display_name.lower()}': {reason}"
    elif busy_with:
        reason = f"Keep going with '{graph.task(busy_with).display_name.lower()}'."
    elif in_progress:
        reason = f"Wait for '{graph.task(in_progress[0]).display_name.lower()}'."
    elif waiting:
        reason = waiting[0][1]
    return Recommendation(
        now=now,
        next_task=next_task,
        reason=reason,
        start_passive=start_passive,
        ranked_active=ranked_active,
        waiting=waiting,
        confirm=confirm,
        plate_eta=plate_eta,
        bottleneck=est.bottleneck,
        busy_with=busy_with,
    )
