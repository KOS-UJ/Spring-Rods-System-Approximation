import numpy as np

def l2_norm(
        values: np.ndarray,
        nodes: np.ndarray
):
    half = nodes.size // 2

    left_val_diff = np.diff(values[:half])
    left_pos_diff = np.diff(nodes[:half])

    right_val_diff = np.diff(values[half:])
    right_pos_diff = np.diff(values[half:])

    left_norm = np.sum(left_val_diff**2 * left_pos_diff**2)
    right_norm = np.sum(right_val_diff**2 * right_pos_diff**2)

    return np.sqrt((left_norm + right_norm) / 3)
