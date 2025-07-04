import pytest

from rl_playground.vrp.time_budgeting.custom_types import Action, Customer, Node
from rl_playground.vrp.time_budgeting.customers.generators import CustomerGenerator, UniformCustomerGenerator
from rl_playground.vrp.time_budgeting.environment import TimeBudgetingEnv
from rl_playground.vrp.time_budgeting.policies import go_action, greedy_policy, reject_policy, wait_action


def test_invalid_route():
    class TestCustomerGenerator(CustomerGenerator):
        def reset(self) -> None:
            self._initial_customers = [
                Customer(Node(3, 4)),  # Takes 5 time units to reach
            ]
            self._future_customers = [
                Customer(Node(3, 3), 3),
            ]

    env = TimeBudgetingEnv(
        customer_generator=TestCustomerGenerator(),
        t_max=10,
        grid_size=10,
        depot=Node(0, 0),
    )

    env.reset()  # This should not raise an error.

    with pytest.raises(ValueError, match="Route exceeds maximum travel time"):
        env = TimeBudgetingEnv(
            customer_generator=TestCustomerGenerator(),
            t_max=9,  # Not enough time to visit the customer and return to the depot
            grid_size=10,
            depot=Node(0, 0),
        )
        env.reset()


@pytest.fixture
def env():
    class TestCustomerGenerator(CustomerGenerator):
        def reset(self) -> None:
            self._initial_customers = [
                Customer(Node(1, 1)),
                Customer(Node(1, 2)),
                Customer(Node(2, 2)),
            ]
            self._future_customers = [
                Customer(Node(3, 3), 2),
                Customer(Node(3, 3), 3),
                Customer(Node(4, 4), 4),
                Customer(Node(5, 5), 5),
            ]

    _env = TimeBudgetingEnv(
        customer_generator=TestCustomerGenerator(),
        t_max=20,
        grid_size=10,
        depot=Node(0, 0),
    )
    return _env


def test_raise_error_when_not_moving(env: TimeBudgetingEnv):
    observation, info = env.reset()
    assert info.vehicle_position == Node(2, 2)
    with pytest.raises(ValueError, match="Route exceeds maximum travel time"):
        while True:
            env.step(wait_action)


def test_env_init(env: TimeBudgetingEnv):
    observation, info = env.reset()
    # Route: (0, 0) -> (2, 2) -> (1, 2) -> (1, 1) -> (0, 0)
    # Travel times: 3 + 1 + 1 + 2 = 7
    # Remaining time: 20 - 7 = 13
    assert info.vehicle_position == Node(2, 2)
    assert info.remaining_route == [Node(1, 2), Node(1, 1), Node(0, 0)]
    assert info.current_time == 3
    _, free_time_budget = observation
    assert free_time_budget == 13  # 20 - 3 - (1 + 1 + 2)
    assert info.new_customers == [Customer(Node(3, 3), 2), Customer(Node(3, 3), 3)]


def test_env_step_stay_at_current_location(env: TimeBudgetingEnv):
    observation, info = env.reset()
    assert info.vehicle_position == Node(2, 2)
    action = Action(
        accepted_customers=[Customer(Node(3, 3), 2)],
        wait_at_current_location=True,
    )
    observation, reward, terminated, truncated, info = env.step(action)
    assert reward == 1
    assert info.vehicle_position == Node(2, 2)  # Vehicle stays at (2, 2)
    assert info.remaining_route == [Node(3, 3), Node(1, 2), Node(1, 1), Node(0, 0)]
    assert info.current_time == 4
    assert info.new_customers == [Customer(Node(4, 4), 4)]


def test_env_step(env: TimeBudgetingEnv):
    observation, info = env.reset()
    assert info.vehicle_position == Node(2, 2)
    action = Action(
        accepted_customers=[Customer(Node(3, 3), 2)],
        wait_at_current_location=False,
    )
    observation, reward, terminated, truncated, info = env.step(action)
    assert reward == 1
    assert info.vehicle_position == Node(3, 3)
    assert info.remaining_route == [Node(1, 2), Node(1, 1), Node(0, 0)]
    assert info.current_time == 5
    assert info.new_customers == [Customer(Node(4, 4), 4), Customer(Node(5, 5), 5)]


def test_env_never_accepts_customers(env: TimeBudgetingEnv):
    observation, info = env.reset()
    done = False
    while not done:
        observation, reward, terminated, truncated, info = env.step(go_action)
        done = terminated and not truncated
    assert info.vehicle_position == Node(0, 0)
    assert info.current_time == 7  # Travel times: 3 + 1 + 1 + 2 = 7
    assert info.remaining_route == []
    _, free_time_budget = observation
    assert free_time_budget == env._t_max - info.current_time


@pytest.mark.parametrize(
    "action, expected_reward",
    [
        (Action(accepted_customers=[], wait_at_current_location=True), 0),
        (Action(accepted_customers=[], wait_at_current_location=False), 0),
        (Action(accepted_customers=[Customer(Node(3, 3), 2)], wait_at_current_location=True), 1),
        (Action(accepted_customers=[Customer(Node(3, 3), 2)], wait_at_current_location=False), 1),
    ],
)
def test_env_action(env: TimeBudgetingEnv, action: Action, expected_reward: int):
    observation, info = env.reset()
    assert info.vehicle_position == Node(2, 2)
    observation, reward, terminated, truncated, info = env.step(action)
    assert reward == expected_reward
    assert info.vehicle_position == Node(2, 2) if action.wait_at_current_location else Node(3, 3)


@pytest.mark.parametrize(
    "t_max, grid_size, initial_customers, future_customers",
    [
        (10000, 10, 5, 5),
        (20000, 20, 10, 10),
        (30000, 30, 15, 15),
        (40000, 40, 20, 20),
    ],
)
def test_greedy_policy(t_max: int, grid_size: int, initial_customers: int, future_customers: int):
    env = TimeBudgetingEnv(
        customer_generator=UniformCustomerGenerator(
            number_of_initial_customers=initial_customers,
            number_of_future_customers=future_customers,
            grid_size=grid_size,
            t_max=t_max,
        ),
        t_max=t_max,
        grid_size=grid_size,
    )
    observation, info = env.reset()

    done = False
    total_reward = 0
    while not done:
        action = greedy_policy(observation, info, env)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated and not truncated
    assert info.vehicle_position == env._depot
    assert info.remaining_route == []
    assert info.current_time <= t_max
    # Since t_max is large, we expect to accept most of the future customers.
    # (Because request times are generated uniformly, some may occur near the end of the time horizon,
    # but we should still be able to accept most of them.)
    assert total_reward >= 0.8 * future_customers


@pytest.mark.parametrize(
    "wait_at_depot_for_one_step",
    [True, False],
)
def test_return_to_depot_then_accept_new_customers(wait_at_depot_for_one_step: bool):
    class TestCustomerGenerator(CustomerGenerator):
        def reset(self) -> None:
            self._initial_customers = [
                Customer(Node(1, 1)),  # Takes 4 seconds to visit and return to the depot
            ]
            self._future_customers = [
                Customer(Node(0, 2), 3),  # Becomes available only when the vehicle returns to the depot
                Customer(Node(2, 2), 3),
            ]

    env = TimeBudgetingEnv(
        customer_generator=TestCustomerGenerator(),
        t_max=20,
        grid_size=10,
        depot=Node(0, 0),
    )
    observation, info = env.reset()
    assert info.vehicle_position == Node(1, 1)
    assert info.remaining_route == [Node(0, 0)]
    assert info.current_time == 2
    assert info.new_customers == []

    observation, reward, terminated, truncated, info = env.step(go_action)
    assert info.vehicle_position == Node(0, 0)
    assert info.remaining_route == []
    assert info.current_time == 4
    assert info.new_customers == [Customer(Node(0, 2), 3), Customer(Node(2, 2), 3)]

    action = Action(
        accepted_customers=[Customer(Node(0, 2), 3), Customer(Node(2, 2), 3)],
        wait_at_current_location=wait_at_depot_for_one_step,
    )
    observation, reward, terminated, truncated, info = env.step(action)
    # The planned route will be ((0, 0) ->) (2, 2) -> (0, 2) -> (0, 0)
    # The vehicle will wait at the depot for one step if wait_at_depot_for_one_step is True

    if wait_at_depot_for_one_step:
        assert info.vehicle_position == Node(0, 0)
        assert info.remaining_route == [Node(2, 2), Node(0, 2), Node(0, 0)]
        assert info.current_time == 5
        observation, reward, terminated, truncated, info = env.step(go_action)

    assert info.vehicle_position == Node(2, 2)
    assert info.remaining_route == [Node(0, 2), Node(0, 0)]
    assert info.current_time == 7 + (1 if wait_at_depot_for_one_step else 0)

    done = False
    while not done:
        action = reject_policy(observation, info)
        observation, reward, terminated, truncated, info = env.step(go_action)
        done = terminated and not truncated

    assert info.vehicle_position == env._depot
    assert info.remaining_route == []
    assert info.current_time <= env._t_max
    assert info.current_time == 11 + (1 if wait_at_depot_for_one_step else 0)

    assert env.final_route == [Node(0, 0), Node(1, 1), Node(0, 0), Node(2, 2), Node(0, 2), Node(0, 0)]
