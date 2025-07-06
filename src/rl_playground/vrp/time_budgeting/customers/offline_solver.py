import logging
import random
from math import ceil

import fire
import mlflow
import numpy as np
from dotenv import load_dotenv
from pulp import PULP_CBC_CMD, LpBinary, LpInteger, LpMaximize, LpProblem, LpVariable, lpSum

from rl_playground.vrp.time_budgeting.custom_types import Customer, Node
from rl_playground.vrp.time_budgeting.customers.customer_logging import log_customers
from rl_playground.vrp.time_budgeting.customers.generators import CustomerGenerator, UniformCustomerGenerator
from rl_playground.vrp.time_budgeting.mlflow_logger import setup_mlflow_logger, upload_log_to_mlflow

logger = logging.getLogger(__name__)

VEHICLE_SPEED = 1.0


def get_test_generator() -> CustomerGenerator:
    class TestGenerator(CustomerGenerator):
        def reset(self):
            self._initial_customers = [
                Customer(Node(9, 5), request_time=0),
            ]
            self._future_customers = [
                Customer(Node(8, 8), request_time=2),
                Customer(Node(7, 5), request_time=9),
                Customer(Node(3, 3), request_time=16),
                Customer(Node(3, 5), request_time=17),
            ]

    generator = TestGenerator()
    return generator


def get_generator(
    t_max: int, grid_size: int, number_of_initial_customers: int, number_of_future_customers: int, seed: int
) -> CustomerGenerator:
    random.seed(seed)
    np.random.seed(seed)

    generator = UniformCustomerGenerator(
        number_of_initial_customers=number_of_initial_customers,
        number_of_future_customers=number_of_future_customers,
        grid_size=grid_size,
        t_max=t_max,
    )
    return generator


def print_solution_summary(
    v_0: Customer, v_0_: Customer, customers: list[Customer], V: list[Customer], model, s, t, x, y
):
    logger.info(f"Status: {model.status}, Objective = {model.objective.value()}")  # type: ignore

    for i, u in enumerate(V):
        for j, v in enumerate(V):
            if u != v and x[u][v].value() == 1:
                logger.info(f"{i}->{j} x: {x[u][v].value()} y: {y[u][v].value()}")

    accepted_customers = [(i, v) for i, v in enumerate(customers) if s[v].value() == 1]
    route = [v_0]
    for i, v in sorted(accepted_customers, key=lambda c: t[c[1]].value()):
        logger.info(f"{i}: at time {t[v].value()}")

        if len(route) >= 2 and y[route[-1]][v].value() == 1:
            route.append(v_0_)
        route.append(v)

    route.append(v_0_)

    route = [node.node for node in route]
    logger.info(f"Route: {', '.join([f'({node.x}, {node.y})' for node in route])}")
    log_customers(customers, v_0.node, route)

    logger.info(f"Objective value: {model.objective.value()}")
    mlflow.log_metric("objective_value", model.objective.value())


def main(t_max=32, grid_size=6, number_of_initial_customers=2, number_of_future_customers=6, seed=42, test=False):
    load_dotenv()

    with mlflow.start_run():
        log_file_path = setup_mlflow_logger()

        mlflow.set_tags({
            "topic": "offline_solver",
            "mlflow.note.content": "Offline solver for the time budgeting environment.",
        })

        generator = (
            get_generator(
                t_max=t_max,
                grid_size=grid_size,
                number_of_initial_customers=number_of_initial_customers,
                number_of_future_customers=number_of_future_customers,
                seed=seed,
            )
            if not test
            else get_test_generator()
        )
        generator.reset()

        mlflow.log_params({
            "t_max": t_max,
            "grid_size": grid_size,
            "number_of_initial_customers": len(generator.initial_customers),
            "number_of_future_customers": len(generator.future_customers),
            "seed": seed,
        })

        for i, customer in enumerate(generator.all_customers):
            logger.info(
                f"{i}: {customer}, distance from depot: "
                f"{ceil(customer.node.distance_to(Node(grid_size // 2, grid_size // 2)))}"
            )

        v_0, v_0_, customers, V, model, s, t, x, y = solver(t_max, grid_size, generator)

        print_solution_summary(v_0, v_0_, customers, V, model, s, t, x, y)
        upload_log_to_mlflow(log_file_path)


def solver(t_max: int, grid_size: int, generator: CustomerGenerator) -> tuple:
    v_0 = Customer(Node(grid_size // 2, grid_size // 2), request_time=0)
    v_0_ = Customer(Node(grid_size // 2, grid_size // 2), request_time=0)

    V_0 = generator.initial_customers
    V_1 = generator.future_customers
    customers = V_0 + V_1
    V = [*customers, v_0, v_0_]

    model = LpProblem("dvrp-budgeting-time", LpMaximize)

    s = {v: LpVariable(f"s_{i}", cat=LpBinary) for i, v in enumerate(customers)}
    t = {v: LpVariable(f"t_{i}", lowBound=0, upBound=t_max, cat=LpInteger) for i, v in enumerate(V)}
    x = {u: {v: LpVariable(f"x_{i}_{j}", cat=LpBinary) for j, v in enumerate(V) if u != v} for i, u in enumerate(V)}
    y = {u: {v: LpVariable(f"y_{i}_{j}", cat=LpBinary) for j, v in enumerate(V) if u != v} for i, u in enumerate(V)}

    for v in V_0:
        model += s[v] == 1

    for v in customers:
        model += lpSum([(1 - s[v]), *[x[u][v] for u in V if u != v]]) == 1
        model += lpSum([(1 - s[v]), *[x[v][u] for u in V if u != v]]) == 1

    model += lpSum([x[v_0][u] for u in customers]) == 1
    model += lpSum([x[u][v_0_] for u in customers]) == 1

    # Optional:
    model += lpSum([x[u][v_0] for u in customers]) == 0
    model += lpSum([x[v_0_][u] for u in customers]) == 0

    for u in V:
        for v in V:
            if u != v:
                model += y[u][v] <= x[u][v]

    for u in V:
        for v in V:
            if u != v:
                d_uv = max(ceil(u.node.distance_to(v.node) / VEHICLE_SPEED), 1)
                d_uv0 = max(ceil(u.node.distance_to(v_0.node) / VEHICLE_SPEED), 1)
                d_v0u = max(ceil(v_0.node.distance_to(u.node) / VEHICLE_SPEED), 1)
                d_v0v = max(ceil(v_0.node.distance_to(v.node) / VEHICLE_SPEED), 1)

                model += (t_max + d_uv) * (1 - x[u][v] + y[u][v]) + t[v] >= t[u] + d_uv
                model += (t_max + d_uv) * (1 - x[u][v] + y[u][v]) + t[v] >= v.request_time + d_uv

                model += (t_max + d_uv0 + d_v0u) * (1 - y[u][v]) + t[v] >= t[u] + d_uv0 + d_v0v
                model += (t_max + d_uv0 + d_v0u) * (1 - y[u][v]) + t[v] >= v.request_time + d_v0v

    model += t[v_0_] <= t_max

    # Objective function
    model += lpSum(s[v] for v in customers)

    model_file = "tmp/dvrp_budgeting_time.lp.txt"
    model.writeLP(model_file)
    mlflow.log_artifact(model_file, "")
    status = model.solve(PULP_CBC_CMD(msg=False))

    if status != 1:
        raise RuntimeError(f"Solver failed with status {status}")
    return v_0, v_0_, customers, V, model, s, t, x, y


if __name__ == "__main__":
    fire.Fire(main)
