"""One test per scheduling rule (IMPLEMENTATION_PLAN.md), plus recovery after a delay."""

from typing import Any

from kcp.scheduler import build_state, recommend
from kcp.schema import Event, TaskGraph, load_events
from tests.conftest import FIXTURES


def ev(i: int, t: float, task: str, status: str, **kw: Any) -> Event:
    return Event(
        event_id=f"e{i}",
        session_id="t",
        t_start_s=t,
        task_id=task,
        status=status,
        source=kw.pop("source", "user"),
        **kw,
    )


def at(graph: TaskGraph, events: list[Event], now: float):  # type: ignore[no-untyped-def]
    return recommend(graph, build_state(graph, events, now), now)


def test_rule1_task_needs_prerequisites(frozen: TaskGraph) -> None:
    rec = at(frozen, [], 0)
    assert "cube_tofu" not in rec.ranked_active  # needs the pressed tofu
    assert "cut_leafy_veg" not in rec.ranked_active  # needs washed greens


def test_rule2_one_active_task_at_a_time(frozen: TaskGraph) -> None:
    events = [ev(0, 0, "press_tofu", "started"), ev(1, 0, "cut_firm_veg", "started")]
    rec = at(frozen, events, 60)
    assert rec.busy_with == "cut_firm_veg"
    assert rec.reason.startswith("After 'cut the firm vegetables")


def test_rule3_passive_tasks_start_first(frozen: TaskGraph) -> None:
    rec = at(frozen, [], 0)
    assert rec.next_task == "press_tofu"
    assert rec.start_passive == ["press_tofu"]


def test_rule4_windowed_tasks_wait(frozen: TaskGraph) -> None:
    waiting = {t for t, _ in at(frozen, [], 0).waiting}
    assert waiting == {"cut_tomato", "reheat_rice"}


def test_rule4_passive_window_is_just_in_time(instant_pot: TaskGraph) -> None:
    on_time = load_events(FIXTURES / "sessions" / "ip_on_time.jsonl")
    before = [e for e in on_time if e.t_start_s < 17 * 60]
    early = at(instant_pot, before, 2 * 60)
    assert "cook_rice_instant_pot" in {t for t, _ in early.waiting}
    ready = at(instant_pot, before, 17 * 60)
    assert ready.next_task == "cook_rice_instant_pot"


def test_rule5_longest_path_first(frozen: TaskGraph) -> None:
    rec = at(frozen, [ev(0, 0, "press_tofu", "started")], 0)
    assert rec.next_task == "cut_firm_veg"
    assert rec.ranked_active[:2] == ["cut_firm_veg", "wash_greens"]


def test_rule6_low_confidence_asks_to_confirm(frozen: TaskGraph) -> None:
    events = [
        ev(0, 0, "cut_firm_veg", "started"),
        ev(1, 480, "cut_firm_veg", "complete", source="student", confidence=0.4),
    ]
    rec = at(frozen, events, 480)
    assert rec.confirm and "cut the firm vegetables" in rec.confirm[0]
    confirmed = at(frozen, [*events, ev(2, 500, "cut_firm_veg", "complete")], 500)
    assert confirmed.confirm == []


def test_passive_task_completes_by_itself(frozen: TaskGraph) -> None:
    state = build_state(frozen, [ev(0, 0, "press_tofu", "started")], 20 * 60)
    assert state.tasks["press_tofu"].status == "complete"


def test_recovery_after_late_rice(instant_pot: TaskGraph) -> None:
    on_time = load_events(FIXTURES / "sessions" / "ip_on_time.jsonl")
    late = load_events(FIXTURES / "sessions" / "ip_late_rice.jsonl")
    t = 33 * 60
    rec_on_time = at(instant_pot, [e for e in on_time if e.t_start_s <= t], t)
    rec_late = at(instant_pot, [e for e in late if e.t_start_s <= t], t)
    assert rec_late.plate_eta - rec_on_time.plate_eta >= 14 * 60
    assert rec_late.bottleneck.startswith("cook the rice")


def test_early_start_warns_about_missing_prep(frozen: TaskGraph) -> None:
    state = build_state(frozen, [ev(0, 0, "heat_wok", "started")], 0)
    assert len(state.warnings) == 1
    assert "Heat the wok started before" in state.warnings[0][1]


def test_missed_window_is_reported_when_anchor_starts(frozen: TaskGraph) -> None:
    events = load_events(FIXTURES / "sessions" / "early_tomato.jsonl")
    state = build_state(frozen, events, 10_000)
    misses = [w for _, w in state.warnings if w.startswith("Missed quality window")]
    assert misses == [
        "Missed quality window: cut the tomato started 47 min before "
        "'plate the food'; best within 25 min."
    ]


def test_correction_replaces_the_wrong_event(frozen: TaskGraph) -> None:
    wrong = ev(0, 0, "cut_leafy_veg", "started", source="student", confidence=0.8)
    fix = ev(1, 10, "cut_tomato", "started", source="user_correction", corrects_event_id="e0")
    state = build_state(frozen, [wrong, fix], 10)
    assert state.tasks["cut_leafy_veg"].status == "pending"
    assert state.tasks["cut_tomato"].status == "started"
    assert state.warnings == []  # the wrong event's warning goes away with it


def test_deadlock_guard_starts_a_waiting_task() -> None:
    graph = TaskGraph.model_validate(
        {
            "graph_id": "tiny",
            "display_name": "tiny",
            "final_task": "b",
            "recipe_order": ["a", "b"],
            "tasks": [
                {
                    "task_id": "a",
                    "display_name": "A",
                    "duration_s": 60,
                    "attention": "active",
                    "observed_by": "user",
                    "window": {
                        "anchor": "b",
                        "max_lead_s": 1,
                        "rationale": "impossible on purpose",
                    },
                },
                {
                    "task_id": "b",
                    "display_name": "B",
                    "requires": ["a"],
                    "duration_s": 600,
                    "attention": "active",
                    "observed_by": "user",
                },
            ],
        }
    )
    rec = at(graph, [], 0)
    assert rec.next_task == "a"
