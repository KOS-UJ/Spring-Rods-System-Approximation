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
