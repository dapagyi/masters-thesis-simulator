from dataclasses import dataclass
from math import sqrt


@dataclass(frozen=True)
class Node:
    x: float
    y: float

    def __hash__(self):
        return id(self)

    def distance_to(self, other: "Node") -> float:
        return sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


@dataclass(frozen=True)
class Customer:
    node: Node
    request_time: int = 0


type Observation = tuple[int, int]


@dataclass
class Info:
    current_time: int
    vehicle_position: Node
    remaining_route: list[Node]
    new_customers: list[Customer]
    all_arrived_customer_nodes_so_far: list[Node]


@dataclass
class Action:
    accepted_customers: list[Customer]
    # next_customer: Customer | None = None  # None if the vehicle should stay where it is
    #
    # The paper describes the action as a tuple of accepted customers and the next customer to visit.
    # Since we maintain the route, it is more natural to consider if the vehicle should stay at the current location.
    wait_at_current_location: bool
