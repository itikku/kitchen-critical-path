"""Replay a session event log and show how the recommendation changes after each event.

Convention: every event is a state change at `t_start_s` ("started" or "complete" at that time)."""

from __future__ import annotations

from kcp.scheduler import build_state, fmt, recommend
from kcp.schema import Event, TaskGraph


def _describe(graph: TaskGraph, e: Event) -> str:
    conf = f", confidence {e.confidence:.2f}" if e.confidence is not None else ""
    fix = f", corrects {e.corrects_event_id}" if e.corrects_event_id else ""
    return f"{e.status} '{graph.task(e.task_id).display_name}' ({e.source}{conf}{fix})"


def replay(graph: TaskGraph, events: list[Event]) -> str:
    events = sorted(events, key=lambda e: e.t_start_s)
    lines = [f"Replay: {graph.display_name} [{graph.graph_id}]", ""]
    steps: list[tuple[float, str]] = [(0.0, "session start")]
    steps += [(e.t_start_s, _describe(graph, e)) for e in events]
    seen: set[tuple[float, str]] = set()  # corrections can remove warnings, so track by content
    for i, (t, what) in enumerate(steps):
        visible = events[:i]
        state = build_state(graph, visible, t)
        rec = recommend(graph, state, t)
        lines.append(f"[{fmt(t)}] {what}")
        if rec.next_task:
            lines.append(f"        next: {graph.task(rec.next_task).display_name} — {rec.reason}")
        else:
            lines.append(f"        next: — {rec.reason}")
        for task_id, why in rec.waiting:
            lines.append(f"        hold: {graph.task(task_id).display_name} — {why}")
        for prompt in rec.confirm:
            lines.append(f"        ? confirm: {prompt}")
        for item in state.warnings:
            if item not in seen:
                seen.add(item)
                lines.append(f"        ! {item[1]}")
        if state.tasks[graph.final_task].status != "complete":
            lines.append(f"        plate ≈ {fmt(rec.plate_eta)} · bottleneck: {rec.bottleneck}")
    return "\n".join(lines) + "\n"
