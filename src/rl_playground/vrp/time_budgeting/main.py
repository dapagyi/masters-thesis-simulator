from pathlib import Path

import click
import mlflow
from dotenv import load_dotenv

from rl_playground.vrp.time_budgeting.agent import TabularAgent
from rl_playground.vrp.time_budgeting.custom_types import Node
from rl_playground.vrp.time_budgeting.customers.generators import (
    Cluster,
    ClusteredCustomerGenerator,
)
from rl_playground.vrp.time_budgeting.environment import TimeBudgetingEnv
from rl_playground.vrp.time_budgeting.training import save_results_and_plots, train


@click.command()
@click.option("--episodes", default=200, help="Number of episodes to run.")
@click.option("--alpha", default=0.2, help="Learning rate for the agent.")
@click.option("--epsilon", default=0.4, help="Epsilon for epsilon-greedy policy.")
@click.option("--gamma", default=0.999, help="Discount factor for future rewards.")
@click.option("--seed", type=int, default=0, help="Random seed for reproducibility.")
@click.option(
    "--initial_scale_factor", default=8, help="Initial scale factor for discretizing state space in TabularAgent."
)
@click.option("--results_dir", default="./tmp", type=str, help="Directory to save plots.")
@click.option("--neighborhood_size", default=3, type=int, help="Neighborhood size for get_value in TabularAgent.")
@click.option("--refinement_episodes", default=50, type=int, help="Number of episodes between value table refinements.")
def main(
    episodes: int,
    alpha: float,
    epsilon: float,
    gamma: float,
    seed: int | None,
    initial_scale_factor: int,
    results_dir: str,
    neighborhood_size: int,
    refinement_episodes: int,
):
    """Train a Tabular agent on the TimeBudgetingEnv and compare with a greedy agent."""
    load_dotenv()

    with mlflow.start_run():
        mlflow.set_tags({
            "experiment": "time_budgeting",
            "topic": "tabular_agent_training",
        })
        t_max = 31
        grid_size = 20
        initial_customers = 2
        future_customers_cluster_1 = 23
        future_customers_cluster_2 = 10
        future_customers_cluster_3 = 5

        customer_generator = ClusteredCustomerGenerator(
            clusters=[
                Cluster(Node(x=5, y=5), 1.5, grid_size, initial_customers, initial=True),
                Cluster(
                    Node(x=5, y=5), 1.5, grid_size, future_customers_cluster_1, t_min=t_max // 4, t_max=3 * t_max // 4
                ),
                Cluster(Node(x=12, y=12), 1.5, grid_size, future_customers_cluster_2, t_min=0, t_max=t_max // 2),
                Cluster(Node(x=11, y=6), 1.5, grid_size, future_customers_cluster_3, t_min=0, t_max=t_max),
            ]
        )
        depot = Node(x=10, y=10)

        mlflow.log_params({
            "episodes": episodes,
            "alpha": alpha,
            "epsilon": epsilon,
            "gamma": gamma,
            "t_max": t_max,
            "grid_size": grid_size,
            "initial_customers": initial_customers,
            "future_customers_cluster_1": future_customers_cluster_1,
            "future_customers_cluster_2": future_customers_cluster_2,
            "seed": seed,
            "initial_scale_factor": initial_scale_factor,
            "neighborhood_size": neighborhood_size,
            "refinement_episodes": refinement_episodes,
        })

        env = TimeBudgetingEnv(
            customer_generator=customer_generator,
            t_max=t_max,
            grid_size=grid_size,
            depot=depot,
        )
        # Create an identical environment for the greedy agent
        greedy_env = TimeBudgetingEnv(
            customer_generator=customer_generator,
            t_max=t_max,
            grid_size=grid_size,
            depot=depot,
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
            print(
                f"Adjusting total episodes from {episodes} to {adjusted_episodes} to accommodate refinement schedule."
            )
            episodes = adjusted_episodes

        rl_rewards_per_episode, greedy_rewards_per_episode = train(
            episodes,
            seed,
            env,
            greedy_env,
            agent,
            results_dir=Path(results_dir),
            refinement_interval=refinement_episodes,
        )
        save_results_and_plots(rl_rewards_per_episode, greedy_rewards_per_episode, agent, Path(results_dir))


if __name__ == "__main__":
    main()
