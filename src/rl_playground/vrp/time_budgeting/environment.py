"""
Time Budgeting

Applying heuristic inspired by the one described in the following paper:

Marlin W. Ulmer, Dirk C. Mattfeld, Felix Köster (2017) Budgeting Time for Dynamic Vehicle Routing with Stochastic Customer Requests. Transportation Science 52(1):20-37.
https://doi.org/10.1287/trsc.2016.0719
"""

from itertools import pairwise
from math import ceil

import gymnasium as gym
import numpy as np

from rl_playground.vrp.time_budgeting.types import Customer, Info, Node, Observation, ResetOptions


class TimeBudgetingEnv(gym.Env):
    def __init__(
        self,
        t_max: int = 100,
        vehicle_speed: float = 1.0,
        number_of_initial_requests: int = 5,
        number_of_future_requests: int = 20,
        grid_size: int = 20,
        depot: Node | None = None,
    ) -> None:
        super().__init__()

        self._t_max = t_max
        self._vehicle_speed = vehicle_speed
        self._number_of_initial_customers = number_of_initial_requests
        self._number_of_future_customers = number_of_future_requests
        self._grid_size = grid_size
        center: int = grid_size // 2
        self._depot = depot if depot else Node(center, center)

        # TODO: Define action and observation spaces

        self.reset()

    def _travel_time(self, from_node: Node, to_node: Node) -> int:
        return ceil(from_node.distance_to(to_node) / self._vehicle_speed)

    def _free_time_budget(self) -> int:
        route_time = sum(self._travel_time(u, v) for u, v in pairwise(self._route))
        return self._t_max - self._point_of_time - route_time

    def _get_obs(self) -> Observation:
        vehicle_position = self._route[-1] if self._route else self._depot  # FIXME
        return Observation(
            current_time=self._point_of_time,
            vehicle_position=vehicle_position,
            remaining_route=self._route,
            new_customers=[req.node for req in self._new_customers],
        )

    def _get_info(self) -> Info:
        return Info(
            point_of_time=self._point_of_time,
            free_time_budget=self._free_time_budget(),
        )

    def reset(self, seed=None, options: ResetOptions | None = None):  # type: ignore
        super().reset(seed=seed)

        self._point_of_time: int = 0
        self._route: list[Node] = []

        if options:
            assert len(options.initial_customers) == self._number_of_initial_customers
            assert len(options.future_customers) == self._number_of_future_customers

            self._insert_nodes_into_route([req.node for req in options.initial_customers])
            self._future_customers = options.future_customers
        else:
            initial_customers: list[Customer] = self._generate_customers(self._number_of_initial_customers, t=0)
            self._insert_nodes_into_route([req.node for req in initial_customers])
            self._future_customers = self._generate_customers(self._number_of_future_customers)

        self._last_step_time = 0
        self._point_of_time = self._travel_time(self._depot, self._route[0]) if self._route else 0
        self._new_customers = list(
            filter(lambda req: self._last_step_time <= req.request_time < self._point_of_time, self._future_customers)
        )

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):  # type: ignore
        accepted_requests = []

        # TODO

        observation = self._get_obs()
        reward = len(accepted_requests)  # Immediate reward is the number of newly accepted requests
        terminated = self._point_of_time >= self._t_max
        truncated = False
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self, mode="human"): ...  # TODO

    def close(self): ...  # TODO?

    def _generate_customers(self, number_of_requests, t: int | None = None) -> list[Customer]:
        customers = [
            Customer(
                node=Node(x=np.random.randint(0, self._grid_size + 1), y=np.random.randint(0, self._grid_size + 1)),
                request_time=t if t else np.random.randint(0, self._t_max),
            )
            for _ in range(number_of_requests)
        ]

        return sorted(
            customers,
            key=lambda x: x.request_time,
        )

    def _insert_nodes_into_route(self, nodes: list[Node]) -> None:
        # Insert accepted requests into the current route using cheapest insertion heuristic
        for node in nodes:
            best_position = None
            min_increase = float("inf")
            for i in range(len(self._route) + 1):
                new_route = self._route[:i] + [node] + self._route[i:]
                total_time = sum(self._travel_time(u, v) for u, v in pairwise(new_route))
                if total_time < min_increase:
                    min_increase = total_time
                    best_position = i
            if best_position is not None:
                self._route.insert(best_position, node)
