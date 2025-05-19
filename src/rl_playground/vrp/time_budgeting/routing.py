from collections.abc import Callable
from itertools import pairwise

from rl_playground.vrp.time_budgeting.custom_types import Node


def min_insert_heuristic(route: list[Node], nodes: list[Node], travel_time: Callable[[Node, Node], int]) -> list[Node]:
    """
    Inserts nodes into the inner part of the given route using the cheapest insertion heuristic.

    The first and last nodes of the route remain unchanged.
    """
    route = route.copy()

    for node in nodes:
        best_position = None
        min_total_time = float("inf")

        for i in range(1, len(route)):
            new_route = route[:i] + [node] + route[i:]
            total_time = sum(travel_time(u, v) for u, v in pairwise(new_route))
            if total_time < min_total_time:
                min_total_time = total_time
                best_position = i

        if best_position is not None:
            route.insert(best_position, node)

    return route
