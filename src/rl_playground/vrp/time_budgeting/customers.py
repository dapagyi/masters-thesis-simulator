import random
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from rl_playground.vrp.time_budgeting.custom_types import Customer, Node


def generate_clustered_customers(
    number_of_customers: int,
    grid_size: int,
    t_max: int,
    num_clusters: int = 3,
    cluster_std: float = 2,
) -> list[Customer]:
    # Randomly place cluster centers
    cluster_centers = [(np.random.uniform(0, grid_size), np.random.uniform(0, grid_size)) for _ in range(num_clusters)]

    customers = []
    for _ in range(number_of_customers):
        # Randomly pick a cluster center
        cx, cy = cluster_centers[np.random.randint(num_clusters)]

        # Sample customer position around the cluster center
        x = np.clip(np.random.normal(loc=cx, scale=cluster_std), 0, grid_size)
        y = np.clip(np.random.normal(loc=cy, scale=cluster_std), 0, grid_size)

        customers.append(
            Customer(
                node=Node(x=x, y=y),
                request_time=np.random.randint(1, t_max),
            )
        )

    return customers


def generate_customers_uniform(number_of_customers: int, grid_size: int, t_max: int) -> list[Customer]:
    return [
        Customer(
            node=Node(x=np.random.uniform(0, grid_size), y=np.random.uniform(0, grid_size)),
            request_time=np.random.randint(1, t_max),
        )
        for _ in range(number_of_customers)
    ]


def visualize_customers(
    customers: list[Customer], route: list[Customer] | None = None, results_dir: Path = Path("results")
) -> None:
    xs = [customer.node.x for customer in customers]
    ys = [customer.node.y for customer in customers]
    request_times = [customer.request_time for customer in customers]
    interarrival_times = [t2 - t1 for t1, t2 in zip(request_times[:-1], request_times[1:])]

    fig, axs = plt.subplots(1, 3, figsize=(18, 6), dpi=144)
    fig.suptitle("Customer Visualization")

    ax = axs[0]
    ax.scatter(xs, ys, c=request_times, edgecolors="black", linewidths=0.3, alpha=1)
    for x, y, t in zip(xs, ys, request_times):
        ax.annotate(str(t), (x, y), fontsize=7, alpha=0.6)

    if route is not None and len(route) > 1:
        for i in range(len(route) - 1):
            start = route[i].node
            end = route[i + 1].node
            ax.annotate(
                "",
                xy=(end.x, end.y),
                xytext=(start.x, start.y),
                arrowprops={"arrowstyle": "->", "color": "black", "lw": 0.8, "alpha": 0.8},
            )

    ax.set_title("Customer Locations with Request Times (Density)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    ax = axs[1]
    ax.hist(request_times, bins=20, color="skyblue", edgecolor="black")
    ax.set_title("Arrival Times")
    ax.set_xlabel("Time")
    ax.set_ylabel("Count")

    ax = axs[2]
    ax.hist(interarrival_times, bins=20, color="lightgreen", edgecolor="black")
    ax.set_title("Interarrival Times")
    ax.set_xlabel("Δ Time")
    ax.set_ylabel("Count")

    results_dir.mkdir(exist_ok=True)
    save_path = results_dir / "customers.svg"
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(save_path)
    plt.close(fig)


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    customers = generate_clustered_customers(40, grid_size=20, t_max=120)
    route = random.sample(customers, 10)

    visualize_customers(customers, route)
    print(f"Generated {len(customers)} customers and saved visualization to 'results/customer_visualization.png'.")
