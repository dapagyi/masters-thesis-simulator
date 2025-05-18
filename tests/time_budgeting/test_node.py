import random

import pytest

from rl_playground.vrp.time_budgeting.custom_types import Node


@pytest.fixture(autouse=True)
def fixed_seed():
    random.seed(42)


def test_fixed_seed():
    # Test that the random seed is fixed.
    assert random.randint(0, 100) == 81
    assert random.randint(0, 100) == 14
    assert random.randint(0, 100) == 3


def test_fixed_seed_different_runs():
    # Test that the random seed is fixed across different runs.
    assert random.randint(0, 100) == 81
    assert random.randint(0, 100) == 14
    assert random.randint(0, 100) == 3


@pytest.mark.parametrize(
    "node1, node2, expected_distance",
    [
        (Node(0, 0), Node(3, 4), 5.0),
        (Node(1, 1), Node(4, 5), 5.0),
        (Node(-1, -1), Node(-4, -5), 5.0),
        (Node(0, 0), Node(0, 0), 0.0),
    ],
)
def test_distance_to(node1, node2, expected_distance):
    assert node1.distance_to(node2) == expected_distance
