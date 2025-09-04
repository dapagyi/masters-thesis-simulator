import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from rl_playground.vrp.time_budgeting.custom_types import Action, Info, Node, Observation
from rl_playground.vrp.time_budgeting.customers.generators import (
    Cluster,
    ClusteredCustomerGenerator,
)
from rl_playground.vrp.time_budgeting.environment import TimeBudgetingEnv

go_action = Action(
    accepted_customers=[],
    wait_at_current_location=False,
)

wait_action = Action(
    accepted_customers=[],
    wait_at_current_location=True,
)


def reject_policy(observation: Observation, info: Info) -> Action:
    """Reject all new customers and go to the next customer in the route if there is one."""
    return go_action if info.remaining_route else wait_action


def greedy_policy(observation: Observation, info: Info, env: TimeBudgetingEnv) -> Action:
    if not info.new_customers:
        return go_action if info.remaining_route else wait_action

    for k in reversed(range(len(info.new_customers) + 1)):
        for accepted_customers in combinations(info.new_customers, k):
            action = Action(
                accepted_customers=list(accepted_customers),
                wait_at_current_location=False,
            )
            try:
                env.calculate_post_decison_state(action)
                return action
            except ValueError:
                continue

    raise ValueError("No valid action found.")


def modified_objective_function_policy(  # noqa: C901
    observation: Observation, info: Info, env: TimeBudgetingEnv, weights: np.ndarray | None = None
) -> Action:
    if weights is None:
        weights = np.array([3.0, 1.0, 1.0])

    progress = info.current_time / env.t_max
    progress = min(max(progress, 0.2), 0.8)

    adjusted_weights = weights * np.array([progress, 1 - progress, 1 - progress])

    def _calculate_objective_value_components(
        action: Action, planned_route: list[Node], point_of_time: int
    ) -> np.ndarray:
        true_objective_value = len(action.accepted_customers) / len(info.new_customers) if info.new_customers else 0
        free_time_budget = env.free_time_budget(route=planned_route, point_of_time=point_of_time) / env.t_max
        # spatial_factor = -(
        #     sum(
        #         sum(u.distance_to(v) ** 2 for v in info.all_arrived_customer_nodes_so_far)
        #         / (
        #             max(u.distance_to(v) ** 2 for v in info.all_arrived_customer_nodes_so_far)
        #             * len(info.all_arrived_customer_nodes_so_far)
        #         )
        #         for u in planned_route
        #     )
        #     / len(planned_route)
        #     if planned_route
        #     else 0
        # )

        closest_customers_per_route_node = defaultdict(list)
        for v in info.all_arrived_customer_nodes_so_far:
            closest_route_node = min(planned_route, key=lambda u: u.distance_to(v), default=None)
            closest_customers_per_route_node[closest_route_node].append(v)

        spatial_factor = 0

        for route_node, closest_customers in closest_customers_per_route_node.items():
            max_dist = max(route_node.distance_to(v) ** 2 for v in closest_customers)
            spatial_factor += sum(route_node.distance_to(c) ** 2 for c in closest_customers) / (
                (max_dist * len(closest_customers)) if max_dist > 0 else 1
            )

        spatial_factor = (
            1 - (spatial_factor / len(closest_customers_per_route_node.keys()))
            if closest_customers_per_route_node
            else 0
        )

        return np.array([true_objective_value, free_time_budget, spatial_factor])

    best_action = None
    _best_action_objective_components = None
    best_action_value = None

    for i in range(len(info.new_customers) + 1):
        for accepted_customer_tuple in combinations(info.new_customers, i):
            accepted_customers = list(accepted_customer_tuple)
            for wait_at_current_location in [True, False]:
                action = Action(
                    accepted_customers=accepted_customers,
                    wait_at_current_location=wait_at_current_location,
                )
                try:
                    # Check if the action is valid. Invalid actions will raise a ValueError.
                    planned_route, point_of_time, _ = env.calculate_post_decison_state(action)

                    comps = _calculate_objective_value_components(action, planned_route, point_of_time)
                    value = np.sum(adjusted_weights * comps)

                    if best_action is None or value > best_action_value:
                        best_action = action
                        _best_action_objective_components = comps
                        best_action_value = value

                except ValueError:
                    # Action is invalid
                    continue

    # Choose the action with the best (lowest) cost
    if not best_action:
        raise ValueError("No valid action found.")

    # print("Best action found:")
    # if info.new_customers:
    #     print(f"{best_action=}, {best_action_objective_components=}, {adjusted_weights=}, {best_action_value=}")

    return best_action


def run_experiment(weights, seed):
    t_max = 31
    grid_size = 20
    initial_customers = 2
    future_customers_cluster_1 = 13
    future_customers_cluster_2 = 10
    future_customers_cluster_3 = 5

    # future_customers_cluster_1 = 18
    # future_customers_cluster_2 = 10
    # future_customers_cluster_3 = 10

    customer_generator = ClusteredCustomerGenerator(
        clusters=[
            Cluster(Node(x=5, y=5), 1.5, grid_size, initial_customers, initial=True),
            Cluster(Node(x=5, y=5), 1.5, grid_size, future_customers_cluster_1, t_min=0, t_max=t_max // 2),
            Cluster(Node(x=15, y=15), 1.5, grid_size, future_customers_cluster_2, t_min=0, t_max=t_max // 2),
            Cluster(Node(x=5, y=15), 1.5, grid_size, future_customers_cluster_3, t_min=0, t_max=t_max),
        ]
    )

    env = TimeBudgetingEnv(
        customer_generator=customer_generator,
        t_max=t_max,
        grid_size=20,
    )
    greedy_env = TimeBudgetingEnv(
        customer_generator=customer_generator,
        t_max=t_max,
        grid_size=20,
    )

    observation, info = env.reset(seed=seed)

    terminated = False
    truncated = False
    total_reward = 0
    while not terminated and not truncated:
        action = modified_objective_function_policy(observation, info, env, weights)
        next_observation, reward, terminated, truncated, info = env.step(action)
        observation = next_observation
        total_reward += reward
        # print(f"Location: {info.vehicle_position}, ")

    # Greedy policy
    observation, info = greedy_env.reset(seed=seed)

    terminated = False
    truncated = False
    greedy_total_reward = 0
    while not terminated and not truncated:
        action = greedy_policy(observation, info, greedy_env)
        next_observation, reward, terminated, truncated, info = greedy_env.step(action)
        observation = next_observation
        greedy_total_reward += reward

    print(f"Episode reward: {total_reward}")
    print(f"Greedy episode reward: {greedy_total_reward}")

    return (total_reward, greedy_total_reward)


def worker(params, seed):
    a, b, c = params["a"], params["b"], params["c"]
    weights = np.array([a, b, c])
    reward, greedy_reward = run_experiment(weights, seed=seed)
    return (seed, a, b, c, reward, greedy_reward)


if __name__ == "__main__":
    grid = ParameterGrid({
        "a": [1],
        "b": [0, 1 / 4, 1 / 2, 1, 2],
        "c": [0, 1 / 4, 1 / 2, 1, 2],
    })
    runs_per_param_comb = 10

    seed = 0
    tasks = []
    for params in grid:
        for _ in range(runs_per_param_comb):
            tasks.append((params, seed))
            seed += 1

    results = []
    with ProcessPoolExecutor(max_workers=None) as executor:
        futures = [executor.submit(worker, p, s) for p, s in tasks]
        for f in tqdm(as_completed(futures), total=len(futures)):
            results.append(f.result())

    results_dir = "./results/5/"
    os.makedirs(results_dir, exist_ok=True)
    # Write results to csv using pandas
    df = pd.DataFrame(results, columns=["seed", "a", "b", "c", "reward", "greedy_reward"])
    df.to_csv(f"{results_dir}experiment_results.csv", index=False)

    # For each param comb calc the number of times the reward is larger than the greedy reward
    df["better_than_greedy"] = df["reward"] > df["greedy_reward"]
    better_counts = df.groupby(["a", "b", "c"])["better_than_greedy"].sum().reset_index()
    better_counts.rename(columns={"better_than_greedy": "count"}, inplace=True)
    # Save the better counts to a csv
    better_counts.to_csv(f"{results_dir}better_counts.csv", index=False)

    # Also add a column for AT LEAST as good:
    df["at_least_as_good"] = df["reward"] >= df["greedy_reward"]
    at_least_counts = df.groupby(["a", "b", "c"])["at_least_as_good"].sum().reset_index()
    at_least_counts.rename(columns={"at_least_as_good": "count"}, inplace=True)
    at_least_counts.to_csv(f"{results_dir}at_least_counts.csv", index=False)
