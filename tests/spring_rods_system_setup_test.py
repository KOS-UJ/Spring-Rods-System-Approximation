import pytest
import numpy as np
from scipy.integrate import quad

from spring_rods_system_setup import SpringRodsSystemSetup


class TestSpringRodsSystemSetup:

    def setup(self):
        self.system = SpringRodsSystemSetup(
            interval=(-10, 10),
            spring_len=3,
            step_size=0.001,
            material_const=(1, 1),
            spring_const=(1, 0.75),
            body_forces=lambda x: np.where(x < 0, 1, -1)
        )

    @pytest.mark.parametrize('function, derivative, spring_behaviour', [
        (lambda x: np.where(x < 0, (x + 10), (x - 10)), lambda x: np.full_like(x, 1), 'compression'),
        (lambda x: (x/10)**2 - 1, lambda x: x/50, 'compression')
    ])
    def test_stress_displacement_prod(self, function, derivative, spring_behaviour):
        domain = self.system.domain
        fun_values = (function(domain[0]), function(domain[1]))

        result = self.system.stress_displacement_prod(fun_values)

        spring_coef = self.system.spring_const[0 if spring_behaviour == 'compression' else 1]
        left_rod_int = quad(lambda x: spring_coef * derivative(x)**2, domain[0][0], domain[0][-1])[0]
        right_rod_int = quad(lambda x: spring_coef * derivative(x)**2, domain[1][0], domain[1][-1])[0]
        expected = left_rod_int + right_rod_int
        assert result == pytest.approx(expected)

    @pytest.mark.parametrize('displacements, spring_reaction', [
        ((np.array([1]), np.array([1])), 0),
        ((np.array([1]), np.array([-1])), 2),
        ((np.array([-1]), np.array([1])), 1.5),
        ((np.array([-2]), np.array([2])), 6),
    ])
    def test_effect_of_spring(self, displacements, spring_reaction):
        result = self.system.effect_of_spring(displacements)

        assert result == pytest.approx(spring_reaction)

    @pytest.mark.parametrize('displacement_func, body_func', [
        (lambda x: np.where(x < 0, (x + 10), (x - 10)), lambda x: np.full_like(x, 1)),
        (lambda x: np.where(x < 0, (x + 10), (x-10)), lambda x: np.where(x < 0, 1, 2)),
        (lambda x: np.where(x < 0, (x + 10)**2, (x-10)**2), lambda x: np.full_like(x, 1)),
        (lambda x: np.where(x < 0, (x + 10)**2, (x-10)**2), lambda x: np.where(x < 0, -1, 1))
    ])
    def test_effect_of_body_forces(self, displacement_func, body_func):
        self.system.set_body_forces(body_func)
        domain = self.system.domain
        displacements = (displacement_func(domain[0]), displacement_func(domain[1]))

        result = self.system.effect_of_body_forces(displacements)

        left_rod_int = quad(lambda x: displacement_func(x) * body_func(x), domain[0][0], domain[0][-1])[0]
        right_rod_int = quad(lambda x: displacement_func(x) * body_func(x), domain[1][0], domain[1][-1])[0]
        expected = left_rod_int + right_rod_int
        assert result == pytest.approx(expected)
