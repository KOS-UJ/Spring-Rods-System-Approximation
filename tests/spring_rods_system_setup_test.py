import pytest
import numpy as np
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
        ((np.array([-2]), np.array([2])), 8),
    ])
    def test_effect_of_spring(self, displacements, spring_reaction):
        result = self.system.effect_of_spring(displacements)

        assert result == pytest.approx(spring_reaction)
