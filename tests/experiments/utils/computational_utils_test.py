from typing import Callable, Tuple

import numpy as np
import pytest
from experiments.utils.computational_utils import l2_norm, approximate_in_positions


@pytest.mark.parametrize(
    'function, interval, expected', [
        (lambda x: np.full_like(x, 1), ((-2, -1), (1, 2)), np.sqrt(2)),
        (lambda x: np.where(x > 0, x, 0), ((-1, 0), (0, 1)), np.sqrt(1/3)),
        (lambda x: np.sin(x), ((-np.pi, np.pi), (-np.pi, np.pi)), np.sqrt(2 * np.pi))
    ])
def test_l2_norm(function: Callable, interval: Tuple[Tuple[float, float], Tuple[float, float]], expected: float):
    domain = np.linspace(*(interval[0]), 10000), np.linspace(*(interval[1]), 10000)
    values = function(domain[0]), function(domain[1])

    result = l2_norm(values, domain)

    assert result == pytest.approx(expected)


@pytest.mark.parametrize('function, interval, ref_step_size, tar_step_size', [
    (lambda x: np.full_like(x, 1), (-1, 1), 0.0003, 0.1),
    (np.sin, (-np.pi, np.pi), 0.001, 0.1),
    (lambda x: x**3, (0, 100), 0.0001, 0.1),
    (lambda x: x, (0, 100), 1, 0.01)
])
def test_approximate_in_positions(function, interval, ref_step_size, tar_step_size):

    interval_size = interval[1] - interval[0]
    ref_domain = np.linspace(*interval, int(interval_size / ref_step_size))
    tar_domain = np.linspace(*interval, int(interval_size / tar_step_size))

    ref_values = function(ref_domain)

    result = approximate_in_positions(ref_values, ref_domain, tar_domain)

    expected = function(tar_domain)
    print(result - expected)

    assert result == pytest.approx(expected)
