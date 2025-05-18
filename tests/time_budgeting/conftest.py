import random

import pytest


@pytest.fixture(autouse=True)
def fixed_seed():
    random.seed(42)
