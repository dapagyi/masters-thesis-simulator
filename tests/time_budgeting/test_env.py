import pytest

from rl_playground.vrp.time_budgeting.custom_types import Action, Customer, Node, ResetOptions
from rl_playground.vrp.time_budgeting.environment import TimeBudgetingEnv


def test_invalid_route():
    reset_options = ResetOptions(
        initial_customers=[
            Customer(Node(3, 4)),  # Takes 5 time units to reach
        ],
        future_customers=[
            Customer(Node(3, 3), 3),
        ],
    )
    env = TimeBudgetingEnv(
        t_max=10,
        number_of_initial_customers=len(reset_options.initial_customers),
        number_of_future_customers=len(reset_options.future_customers),
        grid_size=10,
        depot=Node(0, 0),
    )
    env.reset(options=reset_options)  # This should not raise an error

    with pytest.raises(ValueError, match="Route exceeds time budget"):
        env = TimeBudgetingEnv(
            t_max=9,  # Not enough time to visit the customer and return to the depot
            number_of_initial_customers=len(reset_options.initial_customers),
            number_of_future_customers=len(reset_options.future_customers),
            grid_size=10,
            depot=Node(0, 0),
        )
        env.reset(options=reset_options)


@pytest.fixture
def reset_options():
    initial_customers = [
        Customer(Node(1, 1)),
        Customer(Node(1, 2)),
        Customer(Node(2, 2)),
    ]
    future_customers = [
        Customer(Node(3, 3), 2),
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
        t_max=20,
        number_of_initial_customers=len(reset_options.initial_customers),
        number_of_future_customers=len(reset_options.future_customers),
        grid_size=10,
        depot=Node(0, 0),
    )
    return _env


def test_env_init(env: TimeBudgetingEnv, reset_options: ResetOptions):
    observation, info = env.reset(options=reset_options)
    # Route: (0, 0) -> (2, 2) -> (1, 2) -> (1, 1) -> (0, 0)
    # Travel times: 3 + 1 + 1 + 2 = 7
    # Remaining time: 20 - 7 = 13
    assert observation.vehicle_position == Node(2, 2)
    assert observation.remaining_route == [Node(1, 2), Node(1, 1), Node(0, 0)]
    assert observation.current_time == info.point_of_time == 3
    assert info.free_time_budget == 13  # 20 - 3 - (1 + 1 + 2)
    assert observation.new_customers == [Customer(Node(3, 3), 2), Customer(Node(3, 3), 3)]


def test_env_step_stay_at_current_location(env: TimeBudgetingEnv, reset_options: ResetOptions):
    observation, info = env.reset(options=reset_options)
    assert observation.vehicle_position == Node(2, 2)
    action = Action(
        accepted_customers=[Customer(Node(3, 3), 2)],
        wait_at_current_location=True,
    )
    observation, reward, terminated, truncated, info = env.step(action)
    assert reward == 1
    assert observation.vehicle_position == Node(2, 2)  # Vehicle stays at (2, 2)
    assert observation.remaining_route == [Node(3, 3), Node(1, 2), Node(1, 1), Node(0, 0)]
    assert observation.current_time == info.point_of_time == 4
    assert observation.new_customers == [Customer(Node(4, 4), 4)]


def test_env_step(env: TimeBudgetingEnv, reset_options: ResetOptions):
    observation, info = env.reset(options=reset_options)
    assert observation.vehicle_position == Node(2, 2)
    action = Action(
        accepted_customers=[Customer(Node(3, 3), 2)],
        wait_at_current_location=False,
    )
    observation, reward, terminated, truncated, info = env.step(action)
    assert reward == 1
    assert observation.vehicle_position == Node(3, 3)
    assert observation.remaining_route == [Node(1, 2), Node(1, 1), Node(0, 0)]
    assert observation.current_time == info.point_of_time == 5
    assert observation.new_customers == [Customer(Node(4, 4), 4), Customer(Node(5, 5), 5)]


def test_env_never_accepts_customers(env: TimeBudgetingEnv, reset_options: ResetOptions):
    observation, info = env.reset(options=reset_options)
    action = Action(
        accepted_customers=[],
        wait_at_current_location=False,
    )
    done = False
    while not done:
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated and not truncated
    assert observation.vehicle_position == Node(0, 0)
    assert observation.current_time == 7  # Travel times: 3 + 1 + 1 + 2 = 7
    assert observation.remaining_route == []
    assert info.point_of_time + info.free_time_budget == env._t_max


@pytest.mark.parametrize(
    "action, expected_reward",
    [
        (Action(accepted_customers=[], wait_at_current_location=True), 0),
        (Action(accepted_customers=[], wait_at_current_location=False), 0),
        (Action(accepted_customers=[Customer(Node(3, 3), 2)], wait_at_current_location=True), 1),
        (Action(accepted_customers=[Customer(Node(3, 3), 2)], wait_at_current_location=False), 1),
    ],
)
def test_env_action(env: TimeBudgetingEnv, reset_options: ResetOptions, action: Action, expected_reward: int):
    observation, info = env.reset(options=reset_options)
    assert observation.vehicle_position == Node(2, 2)
    observation, reward, terminated, truncated, info = env.step(action)
    assert reward == expected_reward
    assert observation.vehicle_position == Node(2, 2) if action.wait_at_current_location else Node(3, 3)


@pytest.mark.parametrize(
    "t_max, grid_size, initial_customers, future_customers",
    [
        (1000, 10, 5, 5),
        (2000, 20, 10, 10),
        (3000, 30, 15, 15),
        (4000, 40, 20, 20),
    ],
)
def test_large_env(t_max: int, grid_size: int, initial_customers: int, future_customers: int):
    env = TimeBudgetingEnv(
        t_max=t_max,
        number_of_initial_customers=initial_customers,
        number_of_future_customers=future_customers,
        grid_size=grid_size,
    )
    observation, info = env.reset()
    action = Action(
        accepted_customers=[],
        wait_at_current_location=False,
    )
    done = False
    while not done:
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated and not truncated
    assert observation.vehicle_position == Node(grid_size // 2, grid_size // 2)
    assert observation.current_time <= t_max
