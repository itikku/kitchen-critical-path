from pathlib import Path

import pytest

from kcp.schema import TaskGraph

ROOT = Path(__file__).resolve().parents[1]
GRAPHS = ROOT / "data" / "graphs"
FIXTURES = ROOT / "tests" / "fixtures"


@pytest.fixture
def frozen() -> TaskGraph:
    return TaskGraph.load(GRAPHS / "tofu_stir_fry.frozen_rice.v0.json")


@pytest.fixture
def instant_pot() -> TaskGraph:
    return TaskGraph.load(GRAPHS / "tofu_stir_fry.instant_pot.v0.json")
