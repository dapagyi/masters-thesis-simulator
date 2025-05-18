"""
Dummy Routing Environment

The requests are the same for each episode, so the agent does not learn any generalizable behavior.
"""

from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from matplotlib import pyplot as plt
from tqdm import trange


class VehicleRoutingEnv(gym.Env):
    def __init__(self, requests=None, fuel_cost_per_unit=1.0):
        super().__init__()

        self.fuel_cost_per_unit = fuel_cost_per_unit
        self.requests = requests or self._generate_requests(100)
        self.num_requests = len(self.requests)

        self.observation_space = spaces.Dict({
            "current_index": spaces.Discrete(self.num_requests),
            "vehicle_pos": spaces.Box(low=-100, high=100, shape=(2,), dtype=np.int32),
        })
        self.action_space = spaces.Discrete(2)  # 0 = reject, 1 = accept
        self.current_index = 0

        self.reset()

    def _generate_requests(self, n, seed=0):
        np.random.seed = seed
        return [(np.random.randint(-10, 10), np.random.randint(-10, 10), np.random.randint(5, 20)) for _ in range(n)]

    def reset(self, seed=None, options=None):  # type: ignore
        self.vehicle_pos = np.array([0.0, 0.0])
        self.current_index = 0
        self.route_plan = []
        self.done = False
        # self.requests = self._generate_requests(self.num_requests, seed=seed)
        return self._get_obs(), {}

    def _get_obs(self):
        return {"vehicle_pos": self.vehicle_pos, "current_index": self.current_index}

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode has already ended")

        request = self.requests[self.current_index]
        x, y, reward = request

        reward_out = 0.0

        if action == 1:
            self.route_plan.append((x, y, reward))  # Accept and add to plan
            reward_out = (
                reward - (abs(x - self.vehicle_pos[0]) + abs(y - self.vehicle_pos[1])) * self.fuel_cost_per_unit
            )
            self.vehicle_pos = np.array([x, y])

        self.current_index += 1
        terminated = self.current_index >= self.num_requests

        if terminated:
            self.done = True

        return self._get_obs(), reward_out, terminated, False, {}

    def _calculate_profit(self):
        total_reward = 0.0
        total_distance = 0.0
        pos = np.array([0.0, 0.0])
        for x, y, r in self.route_plan:
            target = np.array([x, y])
            dist = np.linalg.norm(pos - target)
            total_distance += dist
            total_reward += r
            pos = target
        return total_reward - total_distance * self.fuel_cost_per_unit

    def render(self):
        print(f"Vehicle position: {self.vehicle_pos}, current request: {self.current_index}, route: {self.route_plan}")


if __name__ == "__main__":
    env = VehicleRoutingEnv()

    def discretize_obs(obs):
        idx = obs["current_index"]
        pos = tuple((obs["vehicle_pos"] // 5).astype(int))
        return (idx, pos)

    q_table = {}
    alpha = 0.1
    gamma = 0.9
    epsilon = 1.0
    epsilon_decay = 0.995
    min_epsilon = 0.05
    # min_epsilon = 1e-5
    episodes = 5000

    reward_history = []

    for episode in trange(episodes):
        obs, _ = env.reset(seed=episode)
        state = discretize_obs(obs)
        done = False
        total_reward = 0

        while not done:
            if state not in q_table:
                q_table[state] = np.zeros(env.action_space.n)  # type: ignore

            action = env.action_space.sample() if np.random.rand() < epsilon else np.argmax(q_table[state])

            next_obs, reward, terminated, _, _ = env.step(action)
            next_state = discretize_obs(next_obs)

            if next_state not in q_table:
                q_table[next_state] = np.zeros(env.action_space.n)  # type: ignore

            target = reward
            if not terminated:
                target += gamma * np.max(q_table[next_state])

            q_table[state][action] += alpha * (target - q_table[state][action])
            state = next_state
            done = terminated
            total_reward += reward

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        reward_history.append(total_reward)

    print("Training complete.")

    # Visualize the results
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(reward_history, label="Profit per Episode", alpha=0.7)
    ax.plot(
        np.convolve(reward_history, np.ones(10) / 10, mode="valid"),
        label="Smoothed (10-episode avg)",
        linewidth=2,
        color="orange",
    )
    ax.set_title("Training Reward (Profit) per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Profit")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    # Save the plot
    results_dir = Path(".") / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(results_dir / f"dummy_routing_{episodes}.png")
    plt.close(fig)
    print(f"Plot saved to {results_dir / f'dummy_routing_{episodes}.png'}")
