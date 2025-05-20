import random

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def fixed_seed():
    random.seed(42)
    np.random.seed(42)
