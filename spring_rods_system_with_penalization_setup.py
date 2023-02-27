from typing import Tuple, Callable, Union

import numpy as np

from spring_rods_system_setup import SpringRodsSystemSetup


class SpringRodSystemWithPenalizationSetup(SpringRodsSystemSetup):

    def __init__(
            self,
            interval: Tuple[float, float],
            spring_len: float,
            step_size: float,
            material_const: Tuple[float, float],
            spring_const: Tuple[float, float],
            body_forces: Callable[[np.ndarray], Union[np.ndarray, float]],
            penalization_function: Callable[[Tuple[np.ndarray, np.ndarray]], float],
            penalization_const: float
    ):
        super().__init__(interval, spring_len, step_size, material_const, spring_const, body_forces)

        self.penalization_const = penalization_const
        self.penalization_func = penalization_function

    def stress_displacement_prod(self, rods_displacements: Tuple[np.ndarray, np.ndarray]):
        base_value = super().stress_displacement_prod(rods_displacements)
        penalization = self.penalization_func(rods_displacements) / self.penalization_const
        return base_value + penalization
