from typing import Tuple

import numpy as np
from scipy import optimize

from spring_rods_system_setup import SpringRodsSystemSetup


class SpringRodsSystemSolver:
    """
    Solver for the spring-rod system.
    Encapsulates the computation of equilibrium state of the system.
    """
    
    def __init__(self, model: SpringRodsSystemSetup):
        self.model = model
        self.free_nodes_num = self.model.domain[0].size + self.model.domain[1].size - 2
        self.right_rod_beg = self.model.domain[0].size - 1

        self.penetration_constraint = self.create_penetration_constraint()

    def __call__(self, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        :return: pair of displacement vectors corresponding to left and right rod in the equilibrium.
        """

        result = optimize.minimize(
            fun=self.model,
            x0=np.zeros(self.free_nodes_num),
            constraints=self.penetration_constraint,
            options={'maxiter': 1000}
        )
        assert result.success
        # add the boundary nodes (under homogenous dirichlet constraint)
        displacement = np.pad(result.x, (1, 1))
        # check non-penetrating body constraint
        assert np.all(np.diff(np.concatenate((self.model.domain[0], self.model.domain[1])) + displacement) > 0)
        return displacement[:self.model.domain[0].size], displacement[self.model.domain[0].size:]

    def create_penetration_constraint(self) -> optimize.LinearConstraint:
        constraint = np.zeros(self.free_nodes_num)
        constraint[self.right_rod_beg - 1] = 1
        constraint[self.right_rod_beg] = -1

        constraint = optimize.LinearConstraint(A=constraint, lb=-np.inf, ub=self.model.spring_len)
        return constraint

    def compute_stresses(self, displacements: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        stress = tuple()
        for side in (0, 1):
            position_diff = np.diff(self.model.domain[side])
            displacement_diff = np.diff(displacements[side])
            stress += (self.model.alphas[side] * displacement_diff / position_diff,)

        return stress
