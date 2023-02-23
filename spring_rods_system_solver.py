import numpy as np
from scipy import optimize

from spring_rods_system_setup import SpringRodsSystemSetup


class SpringRodsSystemSolver:

    def __init__(self, model: SpringRodsSystemSetup):
        self.model = model
        self.penetration_constraint = self.create_penetration_constraint()

    def __call__(self, *args, **kwargs) -> np.ndarray:
        result = optimize.minimize(
            fun=self.model,
            x0=np.zeros(2 * self.model.nodes_num - 2),
            constraints=self.penetration_constraint,
            options={'maxiter': 1000}
        )
        assert result.success
        # add the boundary nodes (under homogenous dirichlet constraint)
        displacement = np.pad(result.x, (1, 1))
        # check non-penetrating body constraint
        assert np.all(np.diff(np.concatenate((self.model.domain[0], self.model.domain[1])) + displacement) > 0)
        # TODO this change of output type have to be included in experiments
        return displacement

    def create_penetration_constraint(self):
        free_nodes_num = self.model.nodes_num - 1
        constraint = np.zeros(2 * free_nodes_num)
        constraint[free_nodes_num - 1] = 1
        constraint[free_nodes_num] = -1

        constraint = optimize.LinearConstraint(A=constraint, lb=-np.inf, ub=self.model.spring_len)
        return constraint

    def compute_stresses(self, displacements):
        rods_displacement = (displacements[:self.model.nodes_num], displacements[self.model.nodes_num:])

        side_stresses = [[], []]
        for side in (0, 1):
            position_diff = np.diff(self.model.domain[side])
            displacement_diff = np.diff(rods_displacement[side])
            side_stresses[side] = self.model.alphas[side] * displacement_diff / position_diff

        stresses = np.concatenate((side_stresses[0], side_stresses[1]))
        return stresses
