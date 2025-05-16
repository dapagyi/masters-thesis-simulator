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
        Customer(Node(3, 3), 3),
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
        t_max=10,
        number_of_initial_requests=len(reset_options.initial_customers),
        number_of_future_requests=len(reset_options.future_customers),
        grid_size=10,
        depot=Node(0, 0),
    )
    _env.reset(options=reset_options)
    return _env


def test_env_init(env: TimeBudgetingEnv):
    print(env._route)
    print(env._new_customers)
    print(env._future_customers)
    print(env._get_info())
    print(env._get_obs())
    # assert False
