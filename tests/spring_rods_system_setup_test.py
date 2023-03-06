import pytest
import numpy as np
from scipy.integrate import quad

from spring_rods_system_setup import SpringRodsSystemSetup


class TestSpringRodsSystemSetup:

    def setup(self):
        self.system = SpringRodsSystemSetup(
            interval=(-10, 10),
            spring_len=3,
            step_size=0.1,
            material_const=(1, 1),
            spring_const=(1, 0.75),
            body_forces=lambda x: np.where(x < 0, 1, -1)
        )

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
        (lambda x: np.where(x < 0, (x + 10), (x-10)), lambda x: np.full_like(x, 1)),
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
