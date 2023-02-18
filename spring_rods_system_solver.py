import numpy as np
from scipy import optimize

from spring_rods_system_setup import SpringRodsSystemSetup

class SpringRodsSystemSolver:

    def __init__(self, model: SpringRodsSystemSetup):
        self.model = model
        self.penetration_constraint = self.create_penetration_constraint()
        self.dirichlet_boundaries = self.create_boundary_constraint()

    def __call__(self, *args, **kwargs):
        result = optimize.minimize(
            fun=self.model,
            x0=np.zeros(2 * self.model.nodes_num),
            constraints=self.penetration_constraint,
            bounds=self.dirichlet_boundaries)
        # No penetration constraint
        assert np.all(np.diff(np.concatenate((self.model.domain[0], self.model.domain[1])) + result.x) > 0)
        return result

    def create_penetration_constraint(self):
        constraint = np.zeros(2 * self.model.nodes_num)
        constraint[self.model.nodes_num - 1] = 1
        constraint[self.model.nodes_num] = -1

        constraint = optimize.LinearConstraint(A=constraint, lb=-np.inf, ub=self.model.spring_len)
        return constraint

    def create_boundary_constraint(self):
        lower_bounds = np.full(2 * self.model.nodes_num, fill_value=-np.inf)
        lower_bounds[0] = 0
        lower_bounds[-1] = 0

        upper_bounds = np.full(2 * self.model.nodes_num, fill_value=np.inf)
        upper_bounds[0] = 0
        upper_bounds[-1] = 0

        constraint = optimize.Bounds(lower_bounds, upper_bounds)
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
