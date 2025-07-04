from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from rl_playground.vrp.time_budgeting.custom_types import Customer, Node


@dataclass()
class CustomerGenerator(ABC):
    """
    Abstract base class for generating customers.
    """

    @property
    def initial_customers(self) -> list[Customer]:
        return self._initial_customers

    @property
    def future_customers(self) -> list[Customer]:
        return self._future_customers

    @property
    def all_customers(self) -> list[Customer]:
        return self.initial_customers + self.future_customers

    def __post_init__(self):
        self._initial_customers: list[Customer] = []
        self._future_customers: list[Customer] = []

    @abstractmethod
    def reset(self):
        pass


@dataclass()
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
