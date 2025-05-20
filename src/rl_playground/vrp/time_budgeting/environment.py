"""
Time Budgeting

Applying a heuristic inspired by the one described in the following paper:

Marlin W. Ulmer, Dirk C. Mattfeld, Felix Köster (2017) Budgeting Time for Dynamic Vehicle Routing
with Stochastic Customer Requests. Transportation Science 52(1):20-37.
https://doi.org/10.1287/trsc.2016.0719
"""

from itertools import dropwhile, pairwise, takewhile
from math import ceil

import gymnasium as gym
import numpy as np

from rl_playground.vrp.time_budgeting.custom_types import Action, Customer, Info, Node, Observation, ResetOptions
from rl_playground.vrp.time_budgeting.routing import min_insert_heuristic


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

    def free_time_budget(self, route: list[Node] | None = None, point_of_time: int | None = None) -> int:
        if route is None:
            route = self._route
        if point_of_time is None:
            point_of_time = self._point_of_time

        route_time = sum(self._travel_time(u, v) for u, v in pairwise(route))
        return self._t_max - point_of_time - route_time

    def _validate_route(self, route: list[Node] | None = None, point_of_time: int | None = None):
        if self.free_time_budget(route, point_of_time) < 0:
            raise ValueError("Route exceeds maximum travel time")

    def _generate_customers(self, number_of_customers, initial: bool) -> list[Customer]:
        customers = [
            Customer(
                node=Node(x=np.random.randint(0, self._grid_size), y=np.random.randint(0, self._grid_size)),
                request_time=0 if initial else np.random.randint(1, self._t_max),
            )
            for _ in range(number_of_customers)
        ]

        return sorted(
            customers,
            key=lambda x: x.request_time,
        )

    def _remove_processed_customers(self) -> None:
        # Remove customers that have already been processed (either accepted or rejected)
        self._future_customers = list(
            dropwhile(
                lambda customer: customer.request_time <= self._last_step_time,
                self._future_customers,
            )
        )

    def reset(self, seed: int | None = None, options: ResetOptions | None = None) -> tuple[Observation, Info]:  # type: ignore
        super().reset(seed=seed)
        np.random.seed(seed)

        self._point_of_time: int = 0
        self._route: list[Node] = [self._depot, self._depot]  # Start and end at the depot

        if options:
            self._number_of_initial_customers = len(options.initial_customers)
            self._number_of_future_customers = len(options.future_customers)
            initial_customers = options.initial_customers
            self._future_customers = options.future_customers
        else:
            if self._number_of_initial_customers is None or self._number_of_future_customers is None:
                raise ValueError("Number of initial and future customers must be provided")
            initial_customers = self._generate_customers(self._number_of_initial_customers, initial=True)
            self._future_customers = self._generate_customers(self._number_of_future_customers, initial=False)

        self._route = min_insert_heuristic(
            route=self._route,
            nodes=[customer.node for customer in initial_customers],
            travel_time=self._travel_time,
        )
        self._validate_route()

        # Travel to the first customer and update the route.
        self._route = self._route[1:]  # Remove depot from the route.
        self._last_step_time = 0
        self._point_of_time = self._travel_time(self._depot, self._route[0]) if self._route else 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action: Action):  # type: ignore
        self._last_step_time = self._point_of_time
        self._route, self._point_of_time = self.calculate_post_decison_state(action)
        self._remove_processed_customers()
        observation = self._get_obs()
        reward = len(action.accepted_customers)
        terminated = len(self._route) == 1 and not self._future_customers
        truncated = False
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def calculate_post_decison_state(self, action: Action) -> tuple[list[Node], int]:
        point_of_time = self._point_of_time
        route = self._route.copy()

        accepted_customers_nodes = [customer.node for customer in action.accepted_customers]
        should_reset = len(route) == 1 and accepted_customers_nodes is not None
        if should_reset:
            route = [self._depot, self._depot]

        route = min_insert_heuristic(
            route=route,
            nodes=accepted_customers_nodes,
            travel_time=self._travel_time,
        )

        if action.wait_at_current_location:
            point_of_time += 1
        else:
            current_position, route = route[0], route[1:]
            point_of_time += self._travel_time(current_position, route[0])

        self._validate_route(route, point_of_time)
        return route, point_of_time

    def _get_obs(self) -> Observation:
        return (self._point_of_time, self.free_time_budget())

    def _get_info(self) -> Info:
        # Take only the customers that are new in this step.
        new_customers_in_current_step = list(
            takewhile(
                lambda customer: customer.request_time <= self._point_of_time,
                self._future_customers,
            )
        )
        return Info(
            current_time=self._point_of_time,
            vehicle_position=self._route[0],
            remaining_route=self._route[1:],  # Exclude current position
            new_customers=new_customers_in_current_step,
        )
