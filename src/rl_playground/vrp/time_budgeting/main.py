from pathlib import Path

import click

from rl_playground.vrp.time_budgeting.agent import TabularAgent
from rl_playground.vrp.time_budgeting.customers.generators import UniformCustomerGenerator
from rl_playground.vrp.time_budgeting.environment import TimeBudgetingEnv
from rl_playground.vrp.time_budgeting.training import save_results_and_plots, train


@click.command()
@click.option("--episodes", default=1200, help="Number of episodes to run.")
@click.option("--alpha", default=0.1, help="Learning rate for the agent.")
@click.option("--epsilon", default=0.1, help="Epsilon for epsilon-greedy policy.")
@click.option("--gamma", default=0.99, help="Discount factor for future rewards.")
@click.option("--t_max", default=127, help="Maximum time for an episode.")
@click.option("--grid_size", default=20, help="Size of the grid for customer locations.")
@click.option("--initial_customers", default=5, help="Number of initial customers.")
@click.option("--future_customers", default=50, help="Number of future customers.")
@click.option("--seed", type=int, default=0, help="Random seed for reproducibility.")
# Scale factor for 4x4 table with t_max=128 is 128/4 = 32
@click.option(
    "--initial_scale_factor", default=32, help="Initial scale factor for discretizing state space in TabularAgent."
)
@click.option("--results_dir", default="./results", type=str, help="Directory to save plots.")
@click.option("--neighborhood_size", default=3, type=int, help="Neighborhood size for get_value in TabularAgent.")
@click.option(
    "--refinement_episodes", default=200, type=int, help="Number of episodes between value table refinements."
)
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
    initial_scale_factor: int,
    results_dir: str,
    neighborhood_size: int,
    refinement_episodes: int,
):
    """Train a Tabular agent on the TimeBudgetingEnv and compare with a greedy agent."""

    customer_generator = UniformCustomerGenerator(
        number_of_initial_customers=initial_customers,
        number_of_future_customers=future_customers,
        grid_size=grid_size,
        t_max=t_max,
    )

    env = TimeBudgetingEnv(
        customer_generator=customer_generator,
        t_max=t_max,
        grid_size=grid_size,
    )
    # Create an identical environment for the greedy agent
    greedy_env = TimeBudgetingEnv(
        customer_generator=customer_generator,
        t_max=t_max,
        grid_size=grid_size,
    )
    agent = TabularAgent(
        alpha=alpha,
        epsilon=epsilon,
        gamma=gamma,
        t_max=t_max,
        scale_factor=initial_scale_factor,
        neighborhood_size=neighborhood_size,
    )

    num_refinements = 0
    current_sf = initial_scale_factor
    while current_sf > 1:
        if current_sf % 2 != 0 and current_sf != 1:  # Handle cases where sf is not power of 2 initially
            print(
                f"Warning: initial_scale_factor ({initial_scale_factor}) is not a power of 2. "
                f"Refinement might not reach exactly scale_factor=1 for all t_max values."
            )
        current_sf //= 2
        if current_sf < 1:  # Ensure it doesn't go below 1 if initial_sf wasn't a power of 2
            current_sf = 1
        num_refinements += 1

    adjusted_episodes = (num_refinements + 1) * refinement_episodes
    if episodes != adjusted_episodes:
        print(f"Adjusting total episodes from {episodes} to {adjusted_episodes} to accommodate refinement schedule.")
        episodes = adjusted_episodes

    rl_rewards_per_episode, greedy_rewards_per_episode = train(
        episodes, seed, env, greedy_env, agent, results_dir=Path(results_dir), refinement_interval=refinement_episodes
    )
    save_results_and_plots(rl_rewards_per_episode, greedy_rewards_per_episode, agent, Path(results_dir))


if __name__ == "__main__":
    main()
