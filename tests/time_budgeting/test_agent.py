import numpy as np

from rl_playground.vrp.time_budgeting.agent import TabularAgent


def test_init_value_table():
    t_max = 10
    scale_factor = 2
    agent = TabularAgent(alpha=0.1, epsilon=0.2, gamma=0.3, t_max=t_max, scale_factor=scale_factor)
    expected_shape = (t_max // scale_factor + 1, t_max // scale_factor + 1)
    assert agent.value_table.shape == expected_shape
    assert np.all(agent.value_table == 0)
