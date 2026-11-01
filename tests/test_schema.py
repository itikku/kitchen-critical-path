import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from kcp.schema import Event, TaskGraph, load_events
from tests.conftest import GRAPHS


def _graph_dict() -> dict:  # type: ignore[type-arg]
    return json.loads((GRAPHS / "tofu_stir_fry.frozen_rice.v0.json").read_text())


def test_both_graphs_load(frozen: TaskGraph, instant_pot: TaskGraph) -> None:
    assert frozen.final_task == "plate"
    assert "reheat_rice" in frozen.recipe_order
    assert "cook_rice_instant_pot" in instant_pot.recipe_order


def test_cycle_is_rejected() -> None:
    g = _graph_dict()
    for t in g["tasks"]:
        if t["task_id"] == "press_tofu":
            t["requires"] = ["cube_tofu"]
    with pytest.raises(ValidationError, match="cycle"):
        TaskGraph.model_validate(g)


def test_unknown_prerequisite_is_rejected() -> None:
    g = _graph_dict()
    g["tasks"][0]["requires"] = ["no_such_task"]
    with pytest.raises(ValidationError, match="unknown tasks"):
        TaskGraph.model_validate(g)


def test_recipe_order_must_cover_every_task() -> None:
    g = _graph_dict()
    g["recipe_order"] = g["recipe_order"][:-1]
    with pytest.raises(ValidationError, match="recipe_order"):
        TaskGraph.model_validate(g)


def test_event_end_before_start_is_rejected() -> None:
    with pytest.raises(ValidationError, match="before t_start_s"):
        Event(
            event_id="e",
            session_id="s",
            t_start_s=10,
            t_end_s=5,
            task_id="plate",
            status="complete",
            source="user",
        )


def test_load_events_reports_the_bad_line(tmp_path: Path) -> None:
    common = '"session_id":"s","t_start_s":0,"task_id":"plate","source":"user"'
    good = '{"event_id":"a",' + common + ',"status":"started"}'
    bad = '{"event_id":"b",' + common + ',"status":"done"}'
    path = tmp_path / "events.jsonl"
    path.write_text(good + "\n" + bad + "\n")
    with pytest.raises(ValueError, match="events.jsonl:2"):
        load_events(path)
