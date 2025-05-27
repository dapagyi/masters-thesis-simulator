import random
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rl_playground.vrp.time_budgeting.custom_types import Action, Info, Observation
from rl_playground.vrp.time_budgeting.environment import TimeBudgetingEnv


class TabularAgent:
    def __init__(
        self,
        alpha: float,
        epsilon: float,
        gamma: float,
        t_max: int,
        scale_factor: int = 1,
        neighborhood_size: int = 1,  # New parameter
    ):
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.t_max = t_max
        self.scale_factor = scale_factor
        self.scaled_t_max = t_max // scale_factor
        self.value_table = np.zeros((self.scaled_t_max + 1, self.scaled_t_max + 1))
        self.neighborhood_size = neighborhood_size
        if self.neighborhood_size % 2 == 0:
            # Ensure neighborhood_size is odd for a clear center
            self.neighborhood_size += 1
            print(f"Warning: neighborhood_size was even, adjusted to {self.neighborhood_size}")

    def get_value(self, observation: Observation) -> float:
        point_of_time, free_time_budget = observation

        # Scale and clamp indices for the center of the neighborhood
        scaled_point_of_time = point_of_time // self.scale_factor
        scaled_free_time_budget = free_time_budget // self.scale_factor

        center_time_idx = min(scaled_point_of_time, self.scaled_t_max)
        center_budget_idx = min(scaled_free_time_budget, self.scaled_t_max)

        if self.neighborhood_size == 1:
            return self.value_table[center_time_idx, center_budget_idx]
        else:
            half_hood = (self.neighborhood_size - 1) // 2

            start_row = max(0, center_time_idx - half_hood)
            end_row = min(self.scaled_t_max, center_time_idx + half_hood)

            start_col = max(0, center_budget_idx - half_hood)
            end_col = min(self.scaled_t_max, center_budget_idx + half_hood)

            neighborhood = self.value_table[start_row : end_row + 1, start_col : end_col + 1]

            if neighborhood.size == 0:  # Should not happen with proper clamping
                return 0.0
            return float(np.mean(neighborhood))

    def _get_valid_actions(self, env: TimeBudgetingEnv, info: Info) -> list[Action]:
        valid_actions: list[Action] = []
        new_customers = info.new_customers

        for i in range(len(new_customers) + 1):
            for accepted_customer_tuple in combinations(new_customers, i):
                accepted_customers = list(accepted_customer_tuple)
                for wait_at_current_location in [True, False]:
                    action = Action(
                        accepted_customers=accepted_customers,
                        wait_at_current_location=wait_at_current_location,
                    )
                    try:
                        # Check if the action is valid. Invalid actions will raise a ValueError.
                        env.calculate_post_decison_state(action)
                        valid_actions.append(action)
                    except ValueError:
                        # Action is invalid
                        continue
        return valid_actions

    def choose_action(self, env: TimeBudgetingEnv, observation: Observation, info: Info) -> Action:
        valid_actions = self._get_valid_actions(env, info)

        if random.random() < self.epsilon:
            # Explore
            return random.choice(valid_actions)

        # Exploit
        best_action = valid_actions[0]
        max_q_value = -float("inf")

        for action in valid_actions:
            reward = len(action.accepted_customers)
            next_route_state, next_point_of_time, _ = env.calculate_post_decison_state(action)
            next_free_time_budget = env.free_time_budget(route=next_route_state, point_of_time=next_point_of_time)
            next_observation: Observation = (next_point_of_time, next_free_time_budget)

            q_value = reward + self.gamma * self.get_value(next_observation)

            if q_value > max_q_value:
                max_q_value = q_value
                best_action = action

        return best_action

    def update(self, observation: Observation, reward: float, next_observation: Observation, terminated: bool) -> None:
        current_point_of_time, current_free_time_budget = observation

        # Scale and clamp indices for current state
        scaled_current_point_of_time = current_point_of_time // self.scale_factor
        scaled_current_free_time_budget = current_free_time_budget // self.scale_factor

        current_time_idx = min(scaled_current_point_of_time, self.scaled_t_max)
        # Ensure budget_idx is non-negative before min clamping
        current_budget_idx = min(max(0, scaled_current_free_time_budget), self.scaled_t_max)
        # For update, we use the direct cell value, not the neighborhood average
        current_value_for_update = self.value_table[current_time_idx, current_budget_idx]

        next_value = self.get_value(next_observation) if not terminated else 0.0  # get_value uses neighborhood
        td_error = reward + self.gamma * next_value - current_value_for_update
        self.value_table[current_time_idx, current_budget_idx] += self.alpha * td_error

    def refine_value_table(self) -> None:
        """Halves the scale_factor and doubles the value_table dimensions."""
        if self.scale_factor <= 1:
            print("Scale factor is already 1 or less. Cannot refine further.")
            return

        assert self.scale_factor // 2 >= 1, "Scale factor cannot be reduced to less than 1."

        old_scale_factor = self.scale_factor
        old_scaled_t_max = self.scaled_t_max
        old_value_table = self.value_table.copy()

        self.scale_factor //= 2
        self.scaled_t_max = self.t_max // self.scale_factor

        new_value_table = np.zeros((self.scaled_t_max + 1, self.scaled_t_max + 1))

        for r_new in range(self.scaled_t_max + 1):
            for c_new in range(self.scaled_t_max + 1):
                # Determine which old cell this new cell corresponds to
                # Effective point_of_time/budget for the center of the new cell
                effective_pot = r_new * self.scale_factor + self.scale_factor // 2
                effective_ftb = c_new * self.scale_factor + self.scale_factor // 2

                r_old = min(effective_pot // old_scale_factor, old_scaled_t_max)
                c_old = min(effective_ftb // old_scale_factor, old_scaled_t_max)

                new_value_table[r_new, c_new] = old_value_table[r_old, c_old]

        self.value_table = new_value_table
        print(
            f"Value table refined. Scale factor: {old_scale_factor} -> {self.scale_factor}. "
            + f"Table size: ({old_scaled_t_max + 1}x{old_scaled_t_max + 1}) -> "
            + f"({self.scaled_t_max + 1}x{self.scaled_t_max + 1})"
        )

    def save_value_table_heatmap(self, results_dir: Path, filename: str = "value_heatmap.png") -> None:
        """Saves a heatmap of the agent's value table."""
        heatmap_save_path = results_dir / filename
        fig, ax = plt.subplots(figsize=(10, 8), dpi=144)
        im = ax.imshow(self.value_table, aspect="auto", origin="lower", cmap="viridis")

        fig.colorbar(im, ax=ax, label="Value")
        ax.set_xlabel(f"Scaled Free Time Budget (/{self.scale_factor})")
        ax.set_ylabel(f"Scaled Point of Time (/{self.scale_factor})")
        ax.set_title(f"Value Table Heatmap (t_max={self.t_max}, scale={self.scale_factor})")
        fig.savefig(heatmap_save_path)
        plt.close(fig)
