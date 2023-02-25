import numpy as np
from scipy import optimize

from spring_rods_system_setup import SpringRodsSystemSetup


class SpringRodsSystemSolver:

    def __init__(self, model: SpringRodsSystemSetup):
        self.model = model
        self.free_nodes_num = self.model.domain[0].size + self.model.domain[1].size - 2
        self.right_rod_beg = self.model.domain[0].size - 1

        self.penetration_constraint = self.create_penetration_constraint()

    def __call__(self, *args, **kwargs) -> np.ndarray:
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
        # TODO this change of output type have to be included in experiments
        return displacement

    def create_penetration_constraint(self):
        constraint = np.zeros(self.free_nodes_num)
        constraint[self.right_rod_beg - 1] = 1
        constraint[self.right_rod_beg] = -1

        constraint = optimize.LinearConstraint(A=constraint, lb=-np.inf, ub=self.model.spring_len)
        return constraint

    def compute_stresses(self, displacements):
        right_rod_beg = self.model.domain[0].size
        rods_displacement = (displacements[:right_rod_beg], displacements[right_rod_beg:])

        side_stresses = [[], []]
        for side in (0, 1):
            position_diff = np.diff(self.model.domain[side])
            displacement_diff = np.diff(rods_displacement[side])
            side_stresses[side] = self.model.alphas[side] * displacement_diff / position_diff

        stresses = np.concatenate((side_stresses[0], side_stresses[1]))
        return stresses
