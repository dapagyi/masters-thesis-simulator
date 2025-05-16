import random

import pytest

from rl_playground.vrp.time_budgeting.environment import TimeBudgetingEnv
from rl_playground.vrp.time_budgeting.types import Customer, Node, ResetOptions


@pytest.fixture(autouse=True)
def fixed_seed():
    random.seed(42)


def test_fixed_seed():
    # Test that the random seed is fixed
    assert random.randint(0, 100) == 81
    assert random.randint(0, 100) == 14
    assert random.randint(0, 100) == 3


def test_fixed_seed_different_runs():
    # Test that the random seed is fixed across different runs
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


@pytest.fixture
def reset_options():
    initial_customers = [
        Customer(Node(1, 1)),
        Customer(Node(1, 2)),
        Customer(Node(2, 2)),
    ]
    future_customers = [
        Customer(Node(3, 3), 2),
        Customer(Node(4, 4), 4),
        Customer(Node(5, 5), 5),
    ]
    return ResetOptions(
        initial_customers=initial_customers,
        future_customers=future_customers,
    )


@pytest.fixture
def env(reset_options: ResetOptions):
    _env = TimeBudgetingEnv(
        t_max=20,
        number_of_initial_requests=len(reset_options.initial_customers),
        number_of_future_requests=len(reset_options.future_customers),
        grid_size=10,
        depot=Node(0, 0),
    )
    return _env


def test_env_init(env: TimeBudgetingEnv, reset_options: ResetOptions):
    obs, info = env.reset(options=reset_options)
    # Expected route:
    # (0, 0) -> (2, 2) -> (1, 2) -> (1, 1) -> (0, 0)
    # Travel times: 3 + 1 + 1 + 2 = 7
    # Remaining time: 20 - 7 = 13
    assert obs.vehicle_position == Node(2, 2)
    assert obs.remaining_route == [Node(1, 2), Node(1, 1), Node(0, 0)]
    assert obs.current_time == info.point_of_time == 3
    assert info.free_time_budget == 13  # 20 - 3 - ()
