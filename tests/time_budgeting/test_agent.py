import numpy as np

from rl_playground.vrp.time_budgeting.agent import TabularAgent
from rl_playground.vrp.time_budgeting.customers.generators import UniformCustomerGenerator
from rl_playground.vrp.time_budgeting.environment import TimeBudgetingEnv


def test_init_value_table():
    t_max = 10
    scale_factor = 2
    agent = TabularAgent(alpha=0.1, epsilon=0.2, gamma=0.3, t_max=t_max, scale_factor=scale_factor)
    expected_shape = (t_max // scale_factor + 1, t_max // scale_factor + 1)
    assert agent.value_table.shape == expected_shape
    assert np.all(agent.value_table == 0)


def test_training():
    t_max = 60
    scale_factor = 2
    agent = TabularAgent(alpha=0.1, epsilon=0.2, gamma=0.3, t_max=t_max, scale_factor=scale_factor)
    customer_generator = UniformCustomerGenerator(
        number_of_initial_customers=5,
        number_of_future_customers=10,
        grid_size=20,
        t_max=t_max,
    )

    env = TimeBudgetingEnv(
        customer_generator=customer_generator,
        t_max=t_max,
        grid_size=20,
    )

    env.reset()
    observation, info = env.reset(seed=0)

    terminated = False
    truncated = False
    rl_episode_reward = 0
    while not terminated and not truncated:
        action = agent.choose_action(env, observation, info)
        next_observation, reward, terminated, truncated, info = env.step(action)
        agent.update(observation, reward, next_observation, terminated)
        observation = next_observation
        rl_episode_reward += reward
