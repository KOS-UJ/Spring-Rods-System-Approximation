from typing import Tuple
import numpy as np


def l2_norm(values: Tuple[np.ndarray, np.ndarray], domain: Tuple[np.ndarray, np.ndarray]) -> float:

    base = np.sum((
        np.sum(np.diff(val) ** 2 * np.diff(dom) ** 2)
        for val, dom in zip(values, domain)
    ))
    return np.sqrt(base / 3)
