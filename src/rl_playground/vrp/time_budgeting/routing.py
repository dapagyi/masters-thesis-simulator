from collections.abc import Callable
from itertools import pairwise

from rl_playground.vrp.time_budgeting.custom_types import Node


def min_insert_heuristic(
    route: list[Node],
    nodes: list[Node],
    allow_insert_at_beginning: bool,
    travel_time: Callable[[Node, Node], int],
    max_travel_time: int,
) -> list[Node]:
    route = route.copy()
    # Insert accepted customers' nodes into the current route using cheapest insertion heuristic
    for node in nodes:
        best_position = None
        min_route_time = float("inf")

        # Usually, we can insert nodes at the beginning of the route,
        # but at the first step and when we want to stay at our current location,
        # we only allow insertions after the first node.
        #
        # The last position must remain the depot,
        # so we don't allow inserting a node at index len(route).
        for i in range(0 if allow_insert_at_beginning else 1, len(route)):
            new_route = route[:i] + [node] + route[i:]
            total_time = sum(travel_time(u, v) for u, v in pairwise(new_route))
            if total_time < min_route_time:
                min_route_time = total_time
                best_position = i
        if best_position is not None:
            route.insert(best_position, node)

        if min_route_time > max_travel_time:  # type: ignore
            raise ValueError("Route exceeds maximum travel time")

    return route
