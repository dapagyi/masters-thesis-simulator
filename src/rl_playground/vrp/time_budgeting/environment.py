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

from rl_playground.vrp.time_budgeting.custom_types import Action, Info, Node, Observation
from rl_playground.vrp.time_budgeting.customers.generators import CustomerGenerator
from rl_playground.vrp.time_budgeting.routing import min_insert_heuristic


class TimeBudgetingEnv(gym.Env):
    def __init__(
        self,
        customer_generator: CustomerGenerator,
        t_max: int = 100,
        grid_size: int = 20,
        vehicle_speed: float = 1.0,
        depot: Node | None = None,
    ) -> None:
        super().__init__()

        self._customer_generator = customer_generator
        self._t_max = t_max
        self._vehicle_speed = vehicle_speed
        self._grid_size = grid_size
        center = grid_size // 2
        self._depot = depot if depot else Node(center, center)

    def _travel_time(self, from_node: Node, to_node: Node) -> int:
        return max(ceil(from_node.distance_to(to_node) / self._vehicle_speed), 1)

    def free_time_budget(self, route: list[Node], point_of_time: int) -> int:
        route_time = sum(self._travel_time(u, v) for u, v in pairwise(route))
        return self._t_max - point_of_time - route_time

    def _validate_route(self, route: list[Node], point_of_time: int):
        if self.free_time_budget(route, point_of_time) < 0:
            raise ValueError("Route exceeds maximum travel time")

    def _remove_processed_customers(self) -> None:
        # Remove customers that have already been processed (either accepted or rejected)
        self._future_customers = list(
            dropwhile(
                lambda customer: customer.request_time <= self._last_step_time,
                self._future_customers,
            )
        )

    @property
    def t_max(self) -> int:
        return self._t_max

    @property
    def final_route(self) -> list[Node]:
        """Returns the final route after all customers have been processed."""
        assert self.is_done
        return [*self._final_route, self._depot]

    @property
    def is_done(self) -> bool:
        return (len(self._route) == 1 and not self._future_customers) or self._point_of_time == self._t_max

    def reset(self, seed: int | None = None) -> tuple[Observation, Info]:  # type: ignore
        super().reset(seed=seed)
        np.random.seed(seed)

        self._point_of_time: int = 0
        self._route: list[Node] = [self._depot, self._depot]  # Start and end at the depot
        self._final_route: list[Node] = [self._depot]

        self._customer_generator.reset()
        initial_customers = sorted(self._customer_generator.initial_customers, key=lambda c: c.request_time)
        self._future_customers = sorted(self._customer_generator.future_customers, key=lambda c: c.request_time)
        self._route = min_insert_heuristic(
            route=self._route,
            nodes=[customer.node for customer in initial_customers],
            travel_time=self._travel_time,
        )
        self._validate_route(self._route, self._point_of_time)
        self._all_arrived_customer_nodes_so_far = [customer.node for customer in initial_customers]

        # Travel to the first customer and update the route.
        self._route = self._route[1:]  # Remove depot from the route.
        self._last_step_time = 0
        self._point_of_time = self._travel_time(self._depot, self._route[0]) if self._route else 1

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action: Action):  # type: ignore
        self._last_step_time = self._point_of_time

        self._route, self._point_of_time, left_position = self.calculate_post_decison_state(action)
        if left_position:
            self._final_route.append(left_position)
        self._remove_processed_customers()

        observation = self._get_obs()
        reward = len(action.accepted_customers)
        terminated = self.is_done
        truncated = False
        info = self._get_info()

        self._all_arrived_customer_nodes_so_far.extend([customer.node for customer in info.new_customers])

        return observation, reward, terminated, truncated, info

    def calculate_post_decison_state(self, action: Action) -> tuple[list[Node], int, Node | None]:
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

        left_position = None
        if action.wait_at_current_location:
            point_of_time += 1
        else:
            current_position, route = route[0], route[1:]
            next_position = route[0]
            point_of_time += self._travel_time(current_position, next_position)
            left_position = current_position

        self._validate_route(route, point_of_time)
        return route, point_of_time, left_position

    def _get_obs(self) -> Observation:
        return (self._point_of_time, self.free_time_budget(self._route, self._point_of_time))

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
            all_arrived_customer_nodes_so_far=self._all_arrived_customer_nodes_so_far,
        )
