import random
from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from dotenv import load_dotenv

from rl_playground.vrp.time_budgeting.custom_types import Customer, Node
from rl_playground.vrp.time_budgeting.customers.generators import UniformCustomerGenerator


def log_customers_csv(
    customers: list[Customer],
    artifact_dir_path: Path = Path(),
) -> None:
    tmp_dir = Path("tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    file_path = tmp_dir / "customers.csv"

    with open(file_path, "w") as f:
        f.write("node_x,node_y,request_time\n")
        for customer in customers:
            f.write(f"{customer.node.x},{customer.node.y},{customer.request_time}\n")

    artifact_dir_path = artifact_dir_path / "customers"
    mlflow.log_artifact(str(file_path), str(artifact_dir_path))


def plot_2d_histogram(
    customers: list[Customer],
    bins: int = 16,
    artifact_dir_path: Path = Path(),
) -> None:
    xs = [customer.node.x for customer in customers]
    ys = [customer.node.y for customer in customers]

    fig, ax = plt.subplots(figsize=(7, 6))
    h = ax.hist2d(xs, ys, bins=bins, cmap="Blues")
    fig.colorbar(h[3], ax=ax, label="Count")

    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_title("2D Histogram of Customer Locations")
    plt.close(fig)

    artifact_path = artifact_dir_path / "customers" / "2d_histogram.svg"
    mlflow.log_figure(fig, str(artifact_path))


def plot_locations(
    customers: list[Customer],
    route: list[Node] | None = None,
    artifact_dir_path: Path = Path(),
) -> None:
    xs = [customer.node.x for customer in customers]
    ys = [customer.node.y for customer in customers]
    request_times = [customer.request_time for customer in customers]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(xs, ys, c=request_times, edgecolors="black", linewidths=0.8, alpha=0.8, cmap="Blues")
    for x, y, t in zip(xs, ys, request_times, strict=False):
        ax.annotate(str(t), (x + 0.2, y + 0.2), fontsize=14, alpha=0.8)

    if route is not None and len(route) > 1:
        for start, end in pairwise(route):
            ax.annotate(
                "",
                xy=(end.x, end.y),
                xytext=(start.x, start.y),
                arrowprops={
                    "arrowstyle": "->",
                    "color": "black",
                    "lw": 1.6,
                    "alpha": 0.8,
                },
            )

    ax.set_title("Customer Locations with Request Times")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    plt.close(fig)

    artifact_path = artifact_dir_path / "customers" / "locations_with_request_times.svg"
    mlflow.log_figure(fig, str(artifact_path))


if __name__ == "__main__":
    load_dotenv()

    with mlflow.start_run():
        random.seed(42)
        np.random.seed(42)
        mlflow.set_tag("dev", True)
        mlflow.set_tag("run_type", "customer_generation")

        customer_generator = UniformCustomerGenerator(
            number_of_initial_customers=10,
            number_of_future_customers=50,
            grid_size=20,
            t_max=63,
        )
        customer_generator.reset()
        log_customers_csv(customer_generator.all_customers)
        plot_2d_histogram(customer_generator.all_customers)
        route = random.sample(customer_generator.all_customers, 10)
        route = [customer.node for customer in route]
        plot_locations(customer_generator.all_customers, route=route)
