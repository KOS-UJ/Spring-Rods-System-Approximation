from typing import Tuple, Callable, Union
import numpy as np


class SpringRodsSystemSetup:
    """
    Provides the functional which minimum is the solution for the spring rods system
    and encapsulates the mechanics of this system
    """

    def __init__(
            self,
            interval: Tuple[float, float],
            spring_len: float,
            step_size: float,
            material_const: Tuple[float, float],
            spring_const: Tuple[float, float],
            body_forces: Callable[[np.ndarray], Union[np.ndarray, float]]
    ):
        self.spring_len = spring_len
        half_spring_len = spring_len / 2

        left_end, right_end = interval
        left_rod = np.arange(left_end, -half_spring_len + step_size / 2, step_size)
        right_rod = np.arange(half_spring_len, right_end + step_size / 2, step_size)
        self.domain = (left_rod, right_rod)

        assert np.isclose(left_rod[0], left_end)
        assert np.isclose(left_rod[-1], -half_spring_len)
        assert np.isclose(right_rod[0], half_spring_len)
        assert np.isclose(right_rod[-1], right_end)

        self.alphas = material_const
        self.spring_const = spring_const

        self.body_forces_in_elements = self.compute_body_forces(body_forces)

    def __call__(self, displacement_field: np.ndarray):
        """
        Compute F(u) = (1/2) <Au, u> + j(u) - <f, u>

        :param displacement_field: array of displacements in both left and right rod
        WITH NO VALUES OF NODES UNDER DIRICHLET CONDITION
        :return: value of the functional F(u) defined in (6.3)
        """
        # add the boundary nodes with zero dirichlet condition
        displacement_field = np.pad(displacement_field, (1, 1))
        # divide displacement field to corresponding left and right rods
        right_rod_beg = self.domain[0].size
        rods_displacements = (displacement_field[:right_rod_beg], displacement_field[right_rod_beg:])
        return self.stress_displacement_prod(rods_displacements) / 2 \
            + self.effect_of_spring(rods_displacements) \
            - self.effect_of_body_forces(rods_displacements)

    def set_material_const(self, material_const: Tuple[float, float]):
        self.alphas = material_const

    def set_spring_const(self, spring_const: Tuple[float, float]):
        self.spring_const = spring_const

    def set_body_forces(self, body_forces: Callable[[np.ndarray], Union[np.ndarray, float]]):
        self.body_forces_in_elements = self.compute_body_forces(body_forces)

    def stress_displacement_prod(self, rods_displacements: Tuple[np.ndarray, np.ndarray]):
        """
        :param rods_displacements: pair of displacements in left and right rod
        :return: value of the dot product <Au, u> defined in (4.11)
        """
        return np.sum(np.concatenate([
            self.alphas[side] * np.diff(rods_displacements[side]) ** 2 / np.diff(self.domain[side])
            for side in (0, 1)
        ]))

    def effect_of_spring(self, rods_displacements: Tuple[np.ndarray, np.ndarray]):
        """
        :param rods_displacements: pair of displacements in left and right rod
        :return: value of the function hat{j}(u) defined in (6.1)
        """
        left_end = rods_displacements[0][-1]
        right_end = rods_displacements[1][0]
        const = self.spring_const[0 if right_end - left_end < 0 else 1]
        return const * (right_end - left_end) ** 2 / 2

    def effect_of_body_forces(self, rods_displacements: Tuple[np.ndarray, np.ndarray]):
        """
        :param rods_displacements: pair of displacements in left and right rod
        :return: value of the dot product <f, u> defined in (4.13)
        """
        average_displacement = np.concatenate([
            np.diff(self.domain[side]) * (rods_displacements[side][1:] + rods_displacements[side][:-1]) / 2
            for side in (0, 1)
        ])
        return np.sum(self.body_forces_in_elements * average_displacement)

    def compute_body_forces(self, force_function: Callable[[np.ndarray], Union[np.ndarray, float]]):
        """
        :param force_function: function that takes positions and returns corresponding body forces
        :return: value of the body forces in the centers of finite elements
        """
        centers = np.concatenate([
            (self.domain[side][1:] + self.domain[side][:-1]) / 2 for side in (0, 1)
        ])
        return force_function(centers)
