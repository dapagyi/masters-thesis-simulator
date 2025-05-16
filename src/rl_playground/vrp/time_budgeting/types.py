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
class ResetOptions:
    initial_customers: list[Customer]
    future_customers: list[Customer]
