from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
import seaborn as sns

from spring_rods_system_setup import SpringRodsSystemSetup


def plot_displacements_and_stress(
        model: SpringRodsSystemSetup,
        results: List[Tuple[np.ndarray, np.ndarray]],
        parameters_space: list,
        parameter_name: str,
        limits: Tuple[float, float],
        path: str
):
    normalize = colors.Normalize(vmin=limits[0], vmax=limits[1])
    cmap = sns.diverging_palette(255, 15, s=100, l=60, sep=1, center="dark", as_cmap=True)

    fig = plt.gcf()
    fig.set_size_inches((6, 3))

    plt.yticks(
        list(range(-1, -len(parameters_space) - 1, -1)),
        parameters_space
    )
    plt.ylim((-len(parameters_space) - 1, 0))

    for idx, (displacements, stresses) in enumerate(results, 1):

        for side in (0, 1):
            new_positions = model.domain[side] + displacements[side]
            for p_idx, _ in enumerate(new_positions[:-1]):
                plt.plot(
                    [new_positions[p_idx], new_positions[p_idx + 1]],
                    [-idx, -idx],
                    color=cmap(normalize(stresses[side][p_idx])),
                    linewidth=4
                )

    plt.axvline(x=0, color='gray', linestyle='--')
    plt.axvline(x=-0.5, color='gray', linestyle=':')
    plt.axvline(x=0.5, color='gray', linestyle=':')

    plt.ylabel(parameter_name)
    plt.xlabel('Position')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=normalize)
    fig.colorbar(sm)

    plt.savefig(path, bbox_inches='tight')
