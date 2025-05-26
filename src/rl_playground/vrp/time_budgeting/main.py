import click

from rl_playground.vrp.time_budgeting.agent import TabularAgent
from rl_playground.vrp.time_budgeting.environment import TimeBudgetingEnv
from rl_playground.vrp.time_budgeting.training import save_results_and_plots, train


@click.command()
@click.option("--episodes", default=1000, help="Number of episodes to run.")
@click.option("--alpha", default=0.1, help="Learning rate for the agent.")
@click.option("--epsilon", default=0.1, help="Epsilon for epsilon-greedy policy.")
@click.option("--gamma", default=0.99, help="Discount factor for future rewards.")
@click.option("--t_max", default=120, help="Maximum time for an episode.")
@click.option("--grid_size", default=20, help="Size of the grid for customer locations.")
@click.option("--initial_customers", default=5, help="Number of initial customers.")
@click.option("--future_customers", default=50, help="Number of future customers.")
@click.option("--seed", type=int, default=0, help="Random seed for reproducibility.")
@click.option("--scale_factor", default=4, help="Scale factor for discretizing state space in TabularAgent.")
@click.option("--results_dir", default="./results", type=str, help="Directory to save plots.")
def main(
    episodes: int,
    alpha: float,
    epsilon: float,
    gamma: float,
    t_max: int,
    grid_size: int,
    initial_customers: int,
    future_customers: int,
    seed: int | None,
    scale_factor: int,
    results_dir: str,
):
    """Train a Tabular agent on the TimeBudgetingEnv and compare with a greedy agent."""

    env = TimeBudgetingEnv(
        t_max=t_max,
        number_of_initial_customers=initial_customers,
        number_of_future_customers=future_customers,
        grid_size=grid_size,
    )
    # Create an identical environment for the greedy agent
    greedy_env = TimeBudgetingEnv(
        t_max=t_max,
        number_of_initial_customers=initial_customers,
        number_of_future_customers=future_customers,
        grid_size=grid_size,
    )
    agent = TabularAgent(alpha=alpha, epsilon=epsilon, gamma=gamma, t_max=t_max, scale_factor=scale_factor)
    rl_rewards_per_episode, greedy_rewards_per_episode = train(episodes, seed, env, greedy_env, agent)
    save_results_and_plots(rl_rewards_per_episode, greedy_rewards_per_episode, agent, results_dir)


if __name__ == "__main__":
    main()
