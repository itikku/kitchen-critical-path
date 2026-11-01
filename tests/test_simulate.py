import random

from kcp.scheduler import build_state
from kcp.schema import TaskGraph
from kcp.simulate import compare, run_session, sample_durations


def test_same_seed_gives_same_results(frozen: TaskGraph) -> None:
    assert compare(frozen, sessions=5, seed=7) == compare(frozen, sessions=5, seed=7)


def test_simulated_sessions_never_break_a_prerequisite(
    frozen: TaskGraph, instant_pot: TaskGraph
) -> None:
    for graph in (frozen, instant_pot):
        for i in range(20):
            durations = sample_durations(graph, random.Random(i))
            for policy in ("scheduler", "recipe_order"):
                result = run_session(graph, policy, durations)
                state = build_state(graph, result.events, result.total_s + 1)
                assert state.tasks[graph.final_task].status == "complete"
                assert not [w for _, w in state.warnings if "started before" in w]


def test_both_policies_do_the_same_work(frozen: TaskGraph) -> None:
    durations = sample_durations(frozen, random.Random(0))
    a = run_session(frozen, "scheduler", durations)
    b = run_session(frozen, "recipe_order", durations)
    started = lambda r: sorted(e.task_id for e in r.events if e.status == "started")  # noqa: E731
    assert started(a) == started(b) == sorted(t.task_id for t in frozen.tasks)
