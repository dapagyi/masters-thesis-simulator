"""
Time Budgeting

Applying heuristic inspired by the one described in the following paper:

Marlin W. Ulmer, Dirk C. Mattfeld, Felix Köster (2017) Budgeting Time for Dynamic Vehicle Routing with Stochastic Customer Requests. Transportation Science 52(1):20-37.
https://doi.org/10.1287/trsc.2016.0719
"""

from itertools import dropwhile, pairwise, takewhile
from math import ceil

import gymnasium as gym
import numpy as np

from rl_playground.vrp.time_budgeting.custom_types import Action, Customer, Info, Node, Observation, ResetOptions


class TimeBudgetingEnv(gym.Env):
    def __init__(
        self,
        t_max: int = 100,
        vehicle_speed: float = 1.0,
        number_of_initial_customers: int | None = None,
        number_of_future_customers: int | None = None,
        grid_size: int = 20,
        depot: Node | None = None,
    ) -> None:
        super().__init__()

        self._t_max = t_max
        self._vehicle_speed = vehicle_speed
        self._number_of_initial_customers = number_of_initial_customers
        self._number_of_future_customers = number_of_future_customers
        self._grid_size = grid_size
        center: int = grid_size // 2
        self._depot = depot if depot else Node(center, center)

        # TODO: Define action and observation spaces

    def _travel_time(self, from_node: Node, to_node: Node) -> int:
        return ceil(from_node.distance_to(to_node) / self._vehicle_speed)

    def _free_time_budget(self) -> int:
        route_time = sum(self._travel_time(u, v) for u, v in pairwise(self._route))
        return self._t_max - self._point_of_time - route_time

    def _get_obs(self) -> Observation:
        # Remove customers that are already processed (either accepted or rejected)
        self._future_customers = list(
            dropwhile(
                lambda customer: customer.request_time <= self._last_step_time,
                self._future_customers,
            )
        )

        # Take only the customers that are new in this step
        self._new_customers_in_current_step = list(
            takewhile(
                lambda customer: customer.request_time <= self._point_of_time,
                self._future_customers,
            )
        )

        return Observation(
            current_time=self._point_of_time,
            vehicle_position=self._route[0],
            remaining_route=self._route[1:],  # Exclude current position
            new_customers=self._new_customers_in_current_step,
        )

    def _get_info(self) -> Info:
        # FIXME: Maybe this method should return a numpy array instead?
        return Info(
            point_of_time=self._point_of_time,
            free_time_budget=self._free_time_budget(),
        )

    def reset(self, seed: int | None = None, options: ResetOptions | None = None) -> tuple[Observation, Info]:  # type: ignore
        super().reset(seed=seed)

        self._point_of_time: int = 0
        self._route: list[Node] = [self._depot, self._depot]  # Start and end at depot

        if options:
            self._number_of_initial_customers = len(options.initial_customers)
            self._number_of_future_customers = len(options.future_customers)
            self._initial_customers = options.initial_customers
            self._future_customers = options.future_customers
        else:
            if self._number_of_initial_customers is None or self._number_of_future_customers is None:
                raise ValueError("Number of initial and future customers must be provided")
            self._initial_customers = self._generate_customers(self._number_of_initial_customers, t=0)
            self._future_customers = self._generate_customers(self._number_of_future_customers)

        # We use allow_insert_at_beginning=False to ensure that the depot stays the first node in the route.
        self._insert_nodes_into_route(
            [customer.node for customer in self._initial_customers], allow_insert_at_beginning=False
        )

        # Travel to the first customer, update the route
        self._route = self._route[1:]  # Remove depot from the route
        self._last_step_time = 0
        self._point_of_time = self._travel_time(self._depot, self._route[0]) if self._route else 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action: Action):  # type: ignore
        self._last_step_time = self._point_of_time
        accepted_customers_nodes = [customer.node for customer in action.accepted_customers]

        if action.wait_at_current_location:
            self._insert_nodes_into_route(accepted_customers_nodes, allow_insert_at_beginning=False)
            self._point_of_time += 1
        else:
            current_position, self._route = self._route[0], self._route[1:]
            self._insert_nodes_into_route(accepted_customers_nodes)
            self._point_of_time += self._travel_time(current_position, self._route[0])

        observation = self._get_obs()
        reward = len(action.accepted_customers)  # Immediate reward is the number of newly accepted customers
        terminated = len(self._route) == 1 and self._route[0] == self._depot and not self._future_customers
        truncated = False
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self, mode="human"): ...  # TODO

    def close(self): ...  # TODO?

    def _generate_customers(self, number_of_customers, t: int | None = None) -> list[Customer]:
        customers = [
            Customer(
                node=Node(x=np.random.randint(0, self._grid_size + 1), y=np.random.randint(0, self._grid_size + 1)),
                request_time=t if t else np.random.randint(0, self._t_max),
            )
            for _ in range(number_of_customers)
        ]

        return sorted(
            customers,
            key=lambda x: x.request_time,
        )

    def _insert_nodes_into_route(self, nodes: list[Node], allow_insert_at_beginning: bool = True) -> None:
        # Insert accepted customers' nodes into the current route using cheapest insertion heuristic
        for node in nodes:
            best_position = None
            min_route_time = float("inf")

            # Usually, we can insert nodes at the beginning of the route,
            # but at the first step and when we want to stay at our current location,
            # we only allow insertions after the first node.
            #
            # The last position must remain the depot,
            # so we don't allow inserting a node at index len(self._route).
            for i in range(0 if allow_insert_at_beginning else 1, len(self._route)):
                new_route = self._route[:i] + [node] + self._route[i:]
                total_time = sum(self._travel_time(u, v) for u, v in pairwise(new_route))
                if total_time < min_route_time:
                    min_route_time = total_time
                    best_position = i
            if best_position is not None:
                self._route.insert(best_position, node)

            if self._point_of_time + min_route_time > self._t_max:
                raise ValueError("Route exceeds time budget")
