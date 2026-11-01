"""Typed schemas for recipe task graphs and session events."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Attention = Literal["active", "passive"]
EventStatus = Literal["started", "complete", "unknown"]
EventSource = Literal[
    "human", "user", "public_dataset", "student", "teacher", "user_correction", "simulated"
]


class QualityWindow(BaseModel):
    """Start this task no earlier than `max_lead_s` seconds before `anchor` starts.

    Values are authored heuristics, not learned from data."""

    model_config = ConfigDict(extra="forbid")
    anchor: str
    max_lead_s: int = Field(gt=0)
    rationale: str


class Task(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_id: str
    display_name: str
    requires: list[str] = []
    duration_s: int = Field(gt=0)
    attention: Attention
    observed_by: Literal["camera", "user"]
    window: QualityWindow | None = None


class TaskGraph(BaseModel):
    model_config = ConfigDict(extra="forbid")
    graph_id: str
    display_name: str
    final_task: str
    recipe_order: list[str]
    tasks: list[Task]
    note: str | None = None

    @model_validator(mode="after")
    def _check(self) -> TaskGraph:
        ids = [t.task_id for t in self.tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate task_id")
        known = set(ids)
        for t in self.tasks:
            missing = set(t.requires) - known
            if missing:
                raise ValueError(f"{t.task_id} requires unknown tasks: {sorted(missing)}")
            if t.window and t.window.anchor not in known:
                raise ValueError(f"{t.task_id} window anchor is unknown: {t.window.anchor}")
        if self.final_task not in known:
            raise ValueError(f"final_task is unknown: {self.final_task}")
        if sorted(self.recipe_order) != sorted(ids):
            raise ValueError("recipe_order must list every task exactly once")
        self.topological_order()  # raises on a cycle
        orphans = known - self.ancestors(self.final_task) - {self.final_task}
        if orphans:
            raise ValueError(f"tasks do not lead to {self.final_task}: {sorted(orphans)}")
        return self

    def task(self, task_id: str) -> Task:
        for t in self.tasks:
            if t.task_id == task_id:
                return t
        raise KeyError(task_id)

    def successors(self, task_id: str) -> list[str]:
        return [t.task_id for t in self.tasks if task_id in t.requires]

    def ancestors(self, task_id: str) -> set[str]:
        seen: set[str] = set()
        stack = list(self.task(task_id).requires)
        while stack:
            current = stack.pop()
            if current not in seen:
                seen.add(current)
                stack.extend(self.task(current).requires)
        return seen

    def topological_order(self) -> list[str]:
        order: list[str] = []
        done: set[str] = set()
        pending = [t.task_id for t in self.tasks]
        while pending:
            ready = [t for t in pending if set(self.task(t).requires) <= done]
            if not ready:
                raise ValueError(f"cycle among tasks: {sorted(pending)}")
            for t in ready:
                order.append(t)
                done.add(t)
                pending.remove(t)
        return order

    @classmethod
    def load(cls, path: Path) -> TaskGraph:
        return cls.model_validate(json.loads(path.read_text()))


class Event(BaseModel):
    """One observed change in task state. Provenance fields support later AI-label audits."""

    model_config = ConfigDict(extra="forbid")
    event_id: str
    session_id: str
    t_start_s: float = Field(ge=0)
    t_end_s: float | None = None
    task_id: str
    status: EventStatus
    source: EventSource
    source_version: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    corrects_event_id: str | None = None

    @model_validator(mode="after")
    def _check_times(self) -> Event:
        if self.t_end_s is not None and self.t_end_s < self.t_start_s:
            raise ValueError("t_end_s is before t_start_s")
        return self


def load_events(path: Path) -> list[Event]:
    events = []
    for n, line in enumerate(path.read_text().splitlines(), start=1):
        if line.strip():
            try:
                events.append(Event.model_validate_json(line))
            except ValueError as err:
                raise ValueError(f"{path}:{n}: invalid event: {err}") from err
    return events
