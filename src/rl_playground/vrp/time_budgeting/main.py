from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

from rl_playground.vrp.time_budgeting.agent import TabularAgent
from rl_playground.vrp.time_budgeting.environment import TimeBudgetingEnv
from rl_playground.vrp.time_budgeting.policies import greedy_policy


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
def train(
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

    rl_rewards_per_episode = []
    greedy_rewards_per_episode = []

    for episode in (pb := trange(episodes, desc="Training Episodes")):
        episode_seed = seed + episode if seed is not None else None
        # episode_seed = 0

        # RL Agent Run
        observation, info = env.reset(seed=episode_seed)

        terminated = False
        truncated = False
        rl_episode_reward = 0
        while not terminated and not truncated:
            action = agent.choose_action(env, observation, info)
            next_observation, reward, terminated, truncated, info = env.step(action)
            agent.update(observation, reward, next_observation, terminated)
            observation = next_observation
            rl_episode_reward += reward
        rl_rewards_per_episode.append(rl_episode_reward)

        # Greedy Agent Run
        greedy_observation, greedy_info = greedy_env.reset(seed=episode_seed)
        greedy_terminated = False
        greedy_truncated = False
        greedy_episode_reward = 0
        while not greedy_terminated and not greedy_truncated:
            greedy_action = greedy_policy(greedy_observation, greedy_info, greedy_env)
            (
                greedy_observation,
                greedy_step_reward,
                greedy_terminated,
                greedy_truncated,
                greedy_info,
            ) = greedy_env.step(greedy_action)
            greedy_episode_reward += greedy_step_reward
        greedy_rewards_per_episode.append(greedy_episode_reward)

        if (episode + 1) % 100 == 0:
            avg_rl_reward = np.mean(rl_rewards_per_episode[-100:])
            avg_greedy_reward = np.mean(greedy_rewards_per_episode[-100:])
            pb.write(
                f"Episode {episode + 1}/{episodes}, Avg RL Reward (last 100): {avg_rl_reward:.2f}, "
                f"Avg Greedy Reward (last 100): {avg_greedy_reward:.2f}"
            )

    # Save heatmap
    results_dir_path = Path(results_dir)
    results_dir_path.mkdir(parents=True, exist_ok=True)  # Ensure results directory exists
    heatmap_save_path = results_dir_path / "value_heatmap.png"
    agent.save_value_table_heatmap(heatmap_save_path)

    # Create and save rewards comparison plot
    rewards_plot_save_path = results_dir_path / "rewards_comparison.png"
    fig, ax = plt.subplots(figsize=(12, 6), dpi=144)
    ax.plot(rl_rewards_per_episode, label="RL Agent Reward", alpha=0.7)
    ax.plot(greedy_rewards_per_episode, label="Greedy Agent Reward", alpha=0.7, linestyle="--")

    # Calculate and plot moving average for RL agent
    if len(rl_rewards_per_episode) >= 100:
        moving_avg_rl = np.convolve(rl_rewards_per_episode, np.ones(100) / 100, mode="valid")
        ax.plot(
            np.arange(99, len(rl_rewards_per_episode)),
            moving_avg_rl,
            label="RL Agent Moving Avg (100 episodes)",
            color="blue",
            linewidth=2,
        )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("RL Agent vs. Greedy Agent Performance")
    ax.legend()
    ax.grid(True)
    fig.savefig(rewards_plot_save_path)
    plt.close(fig)


if __name__ == "__main__":
    train()
