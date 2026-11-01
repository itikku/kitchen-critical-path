"""Action events -> task events (the rule in docs/label_mapping.md, "From actions to tasks").

The camera can tell that someone is cutting, not what they cut. So:
1. A cutting run is credited to the currently recommended cutting task; if none is recommended,
   to the open cutting task with the longest remaining path.
2. When the run ends, the task is reported complete with low confidence. The scheduler turns a
   low-confidence event into a "Is <task> done?" prompt, so the cook confirms with one word or tap.
3. Mixing (pour or stir) is credited to `mix_sauce` the same way.
"""

from __future__ import annotations

from kcp.edge.runs import RunEvent
from kcp.scheduler import LOW_CONFIDENCE, Recommendation, SessionState
from kcp.schema import Event, TaskGraph

CUT_TASKS = ("cut_aromatics", "cut_firm_veg", "cut_leafy_veg", "cut_tomato", "cube_tofu")
MIX_TASKS = ("mix_sauce",)
END_CONFIDENCE = LOW_CONFIDENCE - 0.1  # below the threshold on purpose: ask the cook to confirm


class Interpreter:
    def __init__(self, graph: TaskGraph, session_id: str, model_version: str) -> None:
        self.graph, self.session_id, self.model_version = graph, session_id, model_version
        self.active: dict[str, str] = {}  # action -> task credited with the current run
        self.n = 0

    def _candidates(self, action: str) -> tuple[str, ...]:
        tasks = CUT_TASKS if action == "cut" else MIX_TASKS
        return tuple(t for t in tasks if t in self.graph.recipe_order)

    def _event(self, task_id: str, status: str, t_s: float, confidence: float) -> Event:
        self.n += 1
        return Event.model_validate(
            {
                "event_id": f"{self.session_id}-cam{self.n:03d}",
                "session_id": self.session_id,
                "t_start_s": t_s,
                "task_id": task_id,
                "status": status,
                "source": "student",
                "source_version": self.model_version,
                "confidence": round(confidence, 3),
            }
        )

    def on_run(self, run: RunEvent, state: SessionState, rec: Recommendation) -> Event | None:
        candidates = self._candidates(run.action)
        if run.kind == "start":
            open_tasks = [t for t in candidates if state.tasks[t].status == "pending"]
            if not open_tasks:
                return None  # nothing left to credit; ignore the run
            ranked = [t for t in rec.ranked_active if t in open_tasks]
            if rec.next_task in open_tasks:
                task = rec.next_task
            elif ranked:
                task = ranked[0]
            else:
                task = open_tasks[0]
            self.active[run.action] = task
            return self._event(task, "started", run.t_s, run.confidence)
        if run.action not in self.active:
            return None
        done = self.active.pop(run.action)
        return self._event(done, "complete", run.t_s, min(run.confidence, END_CONFIDENCE))
