import random
from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from dotenv import load_dotenv

from rl_playground.vrp.time_budgeting.custom_types import Customer, Node
from rl_playground.vrp.time_budgeting.customers.generators import (
    Cluster,
    ClusteredCustomerGenerator,
    CustomerGenerator,
    UniformCustomerGenerator,
)

# REL_PATH = "customers"
REL_PATH = "."


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

    artifact_dir_path = artifact_dir_path / REL_PATH
    mlflow.log_artifact(str(file_path), str(artifact_dir_path) if str(artifact_dir_path) != "." else "")


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

    artifact_path = artifact_dir_path / REL_PATH / "customer_2d_histogram.svg"
    mlflow.log_figure(fig, str(artifact_path))


def plot_locations(
    customers: list[Customer],
    depot: Node,
    route: list[Node] | None = None,
    artifact_dir_path: Path = Path(),
) -> None:
    xs = [customer.node.x for customer in customers]
    ys = [customer.node.y for customer in customers]
    request_times = [customer.request_time for customer in customers]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(xs, ys, c=request_times, label="Customers", edgecolors="black", linewidths=0.8, alpha=0.8, cmap="Blues")
    for x, y, t in zip(xs, ys, request_times, strict=False):
        ax.annotate(str(t), (x + 0.1, y + 0.1), fontsize=14, alpha=0.8)

    if route is not None and len(route) > 1:
        for start, end in pairwise(route):
            ax.annotate(
                "",
                xy=(end.x, end.y),
                xytext=(start.x, start.y),
                arrowprops={
                    "arrowstyle": "->",
                    "color": "gray",
                    "lw": 1.6,
                    "alpha": 0.6,
                },
            )

    ax.scatter(depot.x, depot.y, color="k", s=50, label="Depot", edgecolors="black", linewidths=1.5, marker="s")

    ax.set_title("Customer Locations with Request Times")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    plt.close(fig)

    artifact_path = artifact_dir_path / REL_PATH / "customer_locations.svg"
    mlflow.log_figure(fig, str(artifact_path))


def plot_customer_clusters(
    customer_clusters: list[list[Customer]],
    clusters: list[Cluster],
    artifact_dir_path: Path = Path(),
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, cluster in enumerate(customer_clusters):
        xs = [customer.request_time for customer in cluster]
        ys = [i] * len(xs)
        ax.scatter(xs, ys, label=f"Cluster {i + 1} ({len(cluster)} customers)", alpha=0.6)

    ax.set_yticks(range(len(customer_clusters)))
    ax.set_yticklabels(
        [
            f"Cluster {i + 1}\n({clusters[i].center.x:.1f}, {clusters[i].center.y:.1f})"
            for i in range(len(customer_clusters))
        ],
        ha="center",
    )
    ax.tick_params(axis="y", pad=35)
    ax.set_ylim(bottom=-0.5, top=len(customer_clusters) - 0.5)
    ax.set_xticks(
        np.arange(0, max(customer.request_time for cluster in customer_clusters for customer in cluster) + 1, step=5)
    )
    ax.set_xticklabels(
        np.arange(0, max(customer.request_time for cluster in customer_clusters for customer in cluster) + 1, step=5),
    )
    ax.set_xlabel("Request Time")
    ax.set_title("Customer Clusters by Request Time")
    ax.grid(True, linestyle="--", alpha=0.5)

    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    plt.close(fig)

    artifact_path = artifact_dir_path / REL_PATH / "customer_clusters.svg"
    mlflow.log_figure(fig, str(artifact_path))


def log_customers(
    customer_generator: CustomerGenerator,
    depot: Node,
    route: list[Node] | None = None,
    artifact_dir_path: Path = Path(),
    bins: int = 16,
) -> None:
    log_customers_csv(customer_generator.all_customers, artifact_dir_path)
    plot_2d_histogram(customer_generator.all_customers, bins, artifact_dir_path)
    plot_locations(customer_generator.all_customers, depot, route, artifact_dir_path)
    if isinstance(customer_generator, ClusteredCustomerGenerator):
        plot_customer_clusters(customer_generator.customer_clusters, customer_generator.clusters, artifact_dir_path)


if __name__ == "__main__":
    load_dotenv()

    with mlflow.start_run():
        random.seed(42)
        np.random.seed(42)
        mlflow.set_tag("dev", True)
        mlflow.set_tag("run_type", "customer_generation")

        grid_size = 20
        customer_generator = UniformCustomerGenerator(
            number_of_initial_customers=10,
            number_of_future_customers=50,
            grid_size=grid_size,
            t_max=63,
        )

        t_max = 30
        customer_generator = ClusteredCustomerGenerator(
            clusters=[
                Cluster(Node(x=5, y=15), 5, grid_size, 5, initial=True),
                Cluster(Node(x=5, y=15), 4, grid_size, 25, t_min=0, t_max=t_max),
                Cluster(Node(x=16, y=12), 2, grid_size, 10, t_min=t_max // 2, t_max=t_max),
            ]
        )
        depot = Node(x=grid_size // 4, y=grid_size // 4)
        customer_generator.reset()
        route = random.sample(customer_generator.all_customers, 10)
        route = [customer.node for customer in route]
        log_customers(customer_generator, depot, route)
