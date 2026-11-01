"""Golden-output tests: each fixture's replay must match tests/expected/<fixture>.txt.

To update after an intended change: `uv run python -m tests.test_replay` and review the diff."""

import pytest

from kcp.replay import replay
from kcp.schema import TaskGraph, load_events
from tests.conftest import FIXTURES, GRAPHS

CASES = {
    "frozen_on_time": "frozen_rice",
    "early_wok": "frozen_rice",
    "unknown_state": "frozen_rice",
    "user_correction": "frozen_rice",
    "early_tomato": "frozen_rice",
    "ip_on_time": "instant_pot",
    "ip_late_rice": "instant_pot",
}
EXPECTED = FIXTURES.parent / "expected"


def render(name: str) -> str:
    graph = TaskGraph.load(GRAPHS / f"tofu_stir_fry.{CASES[name]}.v0.json")
    return replay(graph, load_events(FIXTURES / "sessions" / f"{name}.jsonl"))


@pytest.mark.parametrize("name", CASES)
def test_replay_matches_golden(name: str) -> None:
    assert render(name) == (EXPECTED / f"{name}.txt").read_text()


if __name__ == "__main__":
    for case in CASES:
        (EXPECTED / f"{case}.txt").write_text(render(case))
        print(f"wrote {case}")
