from typing import Tuple

import numpy as np
from scipy import optimize

from spring_rods_system_setup import SpringRodsSystemSetup


class SpringRodsSystemSolver:
    """
    Solver for the spring-rod system.
    Encapsulates the computation of equilibrium state of the system.
    """
    
    def __init__(self, model: SpringRodsSystemSetup, spring_len_bounds: tuple = None):
        self.model = model
        self.free_nodes_num = self.model.domain[0].size + self.model.domain[1].size - 2
        self.right_rod_beg = self.model.domain[0].size - 1

        bounds = spring_len_bounds or (-np.inf, model.spring_len)
        self.penetration_constraint = self.create_spring_len_constraint(bounds)

    def __call__(self, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        :return: pair of displacement vectors corresponding to left and right rod in the equilibrium.
        """
        result = optimize.minimize(
            fun=self.model,
            x0=np.zeros(self.free_nodes_num),
            method='SLSQP',
            constraints=self.penetration_constraint,
            tol=1e-13,
            options={'maxiter': 10000}
        )
        assert result.success
        # add the boundary nodes (under homogenous dirichlet constraint)
        displacement = np.pad(result.x, (1, 1))

        assert np.all(np.diff(self.model.domain[0] + displacement[:self.model.domain[0].size]) >= 0)
        assert np.all(np.diff(self.model.domain[1] + displacement[self.model.domain[0].size:]) >= 0)

        rods_div_idx = self.model.domain[0].size
        return displacement[:rods_div_idx], displacement[rods_div_idx:]

    def create_spring_len_constraint(self, bounds: tuple) -> optimize.LinearConstraint:
        constraint = np.zeros(self.free_nodes_num)
        constraint[self.right_rod_beg - 1] = 1
        constraint[self.right_rod_beg] = -1

        constraint = optimize.LinearConstraint(A=constraint, lb=bounds[0], ub=bounds[1])
        return constraint

    def compute_stresses(self, displacements: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        stress = tuple()
        for side in (0, 1):
            position_diff = np.diff(self.model.domain[side])
            displacement_diff = np.diff(displacements[side])
            stress += (self.model.alphas[side] * displacement_diff / position_diff,)

        return stress
