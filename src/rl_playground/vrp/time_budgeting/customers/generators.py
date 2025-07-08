from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from rl_playground.vrp.time_budgeting.custom_types import Customer, Node


@dataclass
class CustomerGenerator(ABC):
    """
    Abstract base class for generating customers.
    """

    @property
    def initial_customers(self) -> list[Customer]:
        return self._initial_customers[:]

    @property
    def future_customers(self) -> list[Customer]:
        return self._future_customers[:]

    @property
    def all_customers(self) -> list[Customer]:
        return self.initial_customers + self.future_customers

    def __post_init__(self):
        self._initial_customers: list[Customer] = []
        self._future_customers: list[Customer] = []

    @abstractmethod
    def reset(self) -> None:
        pass


@dataclass
class UniformCustomerGenerator(CustomerGenerator):
    number_of_initial_customers: int
    number_of_future_customers: int
    grid_size: int
    t_max: int

    def _generate_uniform_customers(self, number_of_customers, initial: bool = False) -> list[Customer]:
        return [
            Customer(
                node=Node(x=np.random.uniform(0, self.grid_size), y=np.random.uniform(0, self.grid_size)),
                request_time=0 if initial else np.random.randint(1, self.t_max),
            )
            for _ in range(number_of_customers)
        ]

    def reset(self) -> None:
        self._initial_customers = self._generate_uniform_customers(self.number_of_initial_customers, True)
        self._future_customers = self._generate_uniform_customers(self.number_of_future_customers, False)


@dataclass
class Cluster:
    center: Node
    radius: float
    grid_size: int
    number_of_customers: int
    t_min: int | None = None
    t_max: int | None = None
    initial: bool = False

    def __post_init__(self):
        assert self.initial or (self.t_min is not None and self.t_max is not None)

    def generate_customers(self) -> list[Customer]:
        customers = []
        for _ in range(self.number_of_customers):
            x = np.clip(
                np.random.uniform(self.center.x - self.radius, self.center.x + self.radius),
                0,
                self.grid_size,
            )
            y = np.clip(
                np.random.uniform(self.center.y - self.radius, self.center.y + self.radius),
                0,
                self.grid_size,
            )
            customers.append(
                Customer(
                    node=Node(x=x, y=y),
                    request_time=0 if self.initial else np.random.randint(self.t_min, self.t_max),  # type: ignore
                )
            )
        return customers


@dataclass
class ClusteredCustomerGenerator(CustomerGenerator):
    clusters: list[Cluster]

    @property
    def customer_clusters(self) -> list[list[Customer]]:
        return self._customer_clusters

    def reset(self) -> None:
        self._customer_clusters = [cluster.generate_customers() for cluster in self.clusters]
        self._initial_customers = []
        self._future_customers = []
        for i, cluster in enumerate(self.clusters):
            if cluster.initial:
                self._initial_customers.extend(self._customer_clusters[i])
            else:
                self._future_customers.extend(self._customer_clusters[i])
