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
    cmap = sns.dark_palette((.7, .7, .7), n_colors=15, as_cmap=True)
    # width = 20

    plt.yticks(
        list(range(-1, -len(parameters_space) - 1, -1)),
        parameters_space
    )
    plt.ylim((-len(parameters_space) - 1, 0))

    for idx, (displacements, stresses) in enumerate(results, 1):

        for side in (0, 1):
            new_positions = model.domain[side] + displacements[side]
            mids = (new_positions[1:] + new_positions[:-1]) / 2
            plt.scatter(
                x=mids,
                y=np.full_like(mids, fill_value=-idx),
                # s=width * normalize(stresses[side]),
                c=cmap(normalize(stresses[side])),
                # marker='s'
                marker="|"
            )

    plt.axvline(x=0, color='gray', linestyle='--')
    plt.axvline(x=-model.spring_len / 2, color='gray', linestyle=':')
    plt.axvline(x=model.spring_len / 2, color='gray', linestyle=':')

    plt.figtext(0.15, 0.01, 'a', horizontalalignment='center', verticalalignment='top')
    plt.figtext(0.285, 0.01, '-l', horizontalalignment='center', verticalalignment='top')
    plt.figtext(0.58, 0.01, 'l', horizontalalignment='center', verticalalignment='top')
    plt.figtext(0.72, 0.01, 'b', horizontalalignment='center', verticalalignment='top')

    plt.ylabel(parameter_name)
    plt.xlabel('Position')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=normalize)
    plt.gcf().colorbar(sm, aspect=50)

    plt.savefig(path, bbox_inches='tight')
