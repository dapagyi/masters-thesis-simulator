import random
from itertools import pairwise
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from rl_playground.vrp.time_budgeting.custom_types import Customer, Node


def visualize_customers(
    customers: list[Customer],
    route: list[Node] | None = None,
    results_dir: Path = Path("results/tmp"),
    filename: str = "customers.png",
) -> None:
    xs = [customer.node.x for customer in customers]
    ys = [customer.node.y for customer in customers]
    request_times = [customer.request_time for customer in customers]
    interarrival_times = [t2 - t1 for t1, t2 in pairwise(request_times)]

    fig, axs = plt.subplots(1, 3, figsize=(18, 6), dpi=144)
    fig.suptitle("Customer Visualization")

    ax = axs[0]
    ax.scatter(xs, ys, c=request_times, edgecolors="black", linewidths=0.3, alpha=1)
    for x, y, t in zip(xs, ys, request_times, strict=False):
        ax.annotate(str(t), (x, y), fontsize=7, alpha=0.6)

    if route is not None and len(route) > 1:
        for start, end in pairwise(route):
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
    save_path = results_dir / filename
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(save_path)
    plt.close(fig)


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    from rl_playground.vrp.time_budgeting.customers.generators import UniformCustomerGenerator

    customer_generator = UniformCustomerGenerator(
        number_of_initial_customers=15,
        number_of_future_customers=5,
        grid_size=20,
        t_max=63,
    )
    customer_generator.reset()
    customers = customer_generator.initial_customers + customer_generator.future_customers

    route = [x.node for x in random.sample(customers, 10)]

    visualize_customers(customers, route)
    print(f"Generated {len(customers)} customers.")
