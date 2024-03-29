from typing import Tuple
import numpy as np


def l2_norm(values: Tuple[np.ndarray, np.ndarray], domain: Tuple[np.ndarray, np.ndarray]) -> float:
    base = np.sum(np.fromiter((
        np.sum((val[:-1]**2 + val[1:]**2 + val[:-1]*val[1:]) * np.diff(dom))
        for val, dom in zip(values, domain)
    ), dtype=float))
    return np.sqrt(base / 3)


def approximate_in_positions(
        values: np.ndarray,
        reference_nodes: np.ndarray,
        target_nodes: np.ndarray
):
    assert reference_nodes.size == values.size
    assert reference_nodes[0] == target_nodes[0]
    assert reference_nodes[-1] == target_nodes[-1]
    result = np.empty_like(target_nodes)
    result[0] = values[0]
    ref_idx = 0
    for idx, curr_node in enumerate(target_nodes[1:], 1):
        while reference_nodes[ref_idx] <= curr_node and ref_idx < reference_nodes.size - 1:
            ref_idx += 1
        assert reference_nodes[ref_idx - 1] <= curr_node <= reference_nodes[ref_idx]

        delta = reference_nodes[ref_idx] - reference_nodes[ref_idx - 1]
        shift = curr_node - reference_nodes[ref_idx - 1]
        lin_diff = (values[ref_idx] - values[ref_idx - 1]) * shift / delta
        result[idx] = values[ref_idx - 1] + lin_diff
    return result
