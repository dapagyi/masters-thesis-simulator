from pathlib import Path

import mlflow
import numpy as np
from matplotlib import pyplot as plt
from tqdm import trange

from rl_playground.vrp.time_budgeting.agent import TabularAgent
from rl_playground.vrp.time_budgeting.customers.customer_logging import log_customers
from rl_playground.vrp.time_budgeting.environment import TimeBudgetingEnv
from rl_playground.vrp.time_budgeting.policies import greedy_policy


def save_rewards_plot(
    rl_rewards_per_episode: list[float],
    greedy_rewards_per_episode: list[float],
    results_dir: Path = Path("./results"),
    filename: str = "rewards_comparison.png",
):
    """Creates and saves a plot comparing RL and Greedy agent rewards."""
    results_dir.mkdir(parents=True, exist_ok=True)
    rewards_plot_save_path = results_dir / filename
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


def train(
    episodes: int,
    seed: int | None,
    env: TimeBudgetingEnv,
    greedy_env: TimeBudgetingEnv,
    agent: TabularAgent,
    results_dir: Path = Path("./results"),
    refinement_interval: int = 100,
):
    rl_rewards_per_episode = []
    greedy_rewards_per_episode = []
    results_dir.mkdir(parents=True, exist_ok=True)

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
        mlflow.log_metric("rl_episode_reward", rl_episode_reward, step=episode)

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
        mlflow.log_metric("greedy_episode_reward", greedy_episode_reward, step=episode)

        if (episode + 1) % refinement_interval == 0:
            avg_rl_reward = np.mean(rl_rewards_per_episode[-refinement_interval:])
            avg_greedy_reward = np.mean(greedy_rewards_per_episode[-refinement_interval:])
            pb.write(
                f"Episode {episode + 1}/{episodes}, Avg RL Reward (last {refinement_interval}): {avg_rl_reward:.2f}, "
                f"Avg Greedy Reward (last {refinement_interval}): {avg_greedy_reward:.2f}"
            )
            agent.epsilon -= 0.15
            log_customers(env._customer_generator, env._depot, env.final_route, Path(".") / "RL" / str(episode + 1))
            log_customers(
                greedy_env._customer_generator,
                greedy_env._depot,
                greedy_env.final_route,
                Path(".") / "Greedy" / str(episode + 1),
            )

        # Refinement and heatmap saving logic
        if refinement_interval > 0 and (episode + 1) % refinement_interval == 0 and (episode + 1) < episodes:
            heatmap_filename = f"value_heatmap_episode_{episode + 1}.png"
            agent.save_value_table_heatmap(
                results_dir,
                filename=heatmap_filename,
                artifact_file_path=Path(".") / "RL" / str(episode + 1) / heatmap_filename,
            )
            pb.write(f"Saved heatmap to {results_dir / heatmap_filename}")
            if agent.scale_factor > 1:  # Check if refinement is possible
                agent.refine_value_table()
                pb.write(f"Refined value table. New scale_factor: {agent.scale_factor}")

    return rl_rewards_per_episode, greedy_rewards_per_episode


def save_results_and_plots(
    rl_rewards_per_episode: list[float],
    greedy_rewards_per_episode: list[float],
    agent: TabularAgent,
    results_dir: Path = Path("./results"),
):
    results_dir.mkdir(parents=True, exist_ok=True)
    agent.save_value_table_heatmap(
        results_dir, filename="value_heatmap_final.png", artifact_file_path=Path(".") / "RL" / "value_heatmap_final.png"
    )
    save_rewards_plot(
        rl_rewards_per_episode, greedy_rewards_per_episode, results_dir, filename="rewards_comparison_final.png"
    )
