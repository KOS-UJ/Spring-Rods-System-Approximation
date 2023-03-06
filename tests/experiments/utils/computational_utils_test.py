from typing import Callable, Tuple

import numpy as np
import pytest
from experiments.utils.computational_utils import l2_norm


@pytest.mark.parametrize(
    'function, interval, expected', [
        (lambda x: np.full_like(x, 1), ((-2, -1), (1, 2)), np.sqrt(2)),
        (lambda x: np.where(x > 0, x, 0), ((-1, 0), (0, 1)), np.sqrt(1/3))
    ])
def test_l2_norm(function: Callable, interval: Tuple[Tuple[float, float], Tuple[float, float]], expected: float):
    domain = np.linspace(*(interval[0])), np.linspace(*(interval[1]))
    values = function(domain[0]), function(domain[1])

    result = l2_norm(values, domain)

    assert result == pytest.approx(expected)
