from dataclasses import dataclass
from math import sqrt


@dataclass
class Node:
    x: int
    y: int

    def distance_to(self, other: "Node") -> float:
        return sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


@dataclass
class Customer:
    node: Node
    request_time: int = 0


@dataclass
class Info:
    point_of_time: int
    free_time_budget: int


@dataclass
class Observation:
    current_time: int
    vehicle_position: Node
    remaining_route: list[Node]
    new_customers: list[Customer]


@dataclass
class Action:
    accepted_customers: list[Customer]
    # next_customer: Customer | None = None  # None if the vehicle should stay where it is
    #
    # The paper describes the action as a tuple of accepted customers and the next customer to visit.
    # Since we maintain the route, it is more natural to consider if the vehicle should stay at the current location.
    wait_at_current_location: bool


@dataclass
class ResetOptions:
    initial_customers: list[Customer]
    future_customers: list[Customer]
