from typing import Tuple, Callable, Union
import numpy as np

class SpringRodsSystemSetup:

    def __init__(
            self,
            interval: Tuple[float, float],
            spring_len: float,
            nodes_num: int,
            material_const: Tuple[float, float],
            spring_const: Tuple[float, float],
            body_forces: Callable[[np.ndarray], Union[np.ndarray, float]]
    ):
        self.left_end, self.right_end = interval
        self.spring_len = spring_len
        self.half_spring_len = spring_len / 2
        self.nodes_num = nodes_num
        self.body_forces = body_forces

        self.left_domain = np.linspace(self.left_end, -self.half_spring_len, nodes_num)
        self.right_domain = np.linspace(self.half_spring_len, self.right_end, nodes_num)
        self.alphas = material_const
        self.spring_const = spring_const

    def __call__(self, displacements: np.ndarray):
        return self.A(displacements) / 2 + self.j(displacements) - self.f(displacements)

    def set_material_const(self, material_const: Tuple[float, float]):
        self.alphas = material_const

    def set_spring_const(self, spring_const: Tuple[float, float]):
        self.spring_const = spring_const

    def set_body_forces(self, body_forces: Callable[[np.ndarray], Union[np.ndarray, float]]):
        self.body_forces = body_forces

    def A(self, displacements: np.ndarray):
        left_displ = displacements[:self.nodes_num]
        right_displ = displacements[self.nodes_num:]

        left = np.diff(left_displ) ** 2 / np.diff(self.left_domain)
        right = np.diff(right_displ) ** 2 / np.diff(self.right_domain)

        return self.alphas[0] * np.sum(left) + self.alphas[1] * np.sum(right)

    def j(self, displacements: np.ndarray):
        left_end_displ = displacements[self.nodes_num - 1]
        right_end_displ = displacements[self.nodes_num]
        const = self.spring_const[0 if right_end_displ - left_end_displ < 0 else 1]
        return const * (right_end_displ - left_end_displ) ** 2 / 2

    def f(self, displacements: np.ndarray):
        fun_vals = self.compute_body_forces()

        left_displ = displacements[:self.nodes_num]
        right_displ = displacements[self.nodes_num:]

        left = np.diff(self.left_domain) * (left_displ[1:] + left_displ[:-1]) / 2
        right = np.diff(self.right_domain) * (right_displ[1:] + right_displ[:-1]) / 2

        return np.sum(fun_vals * np.concatenate((left, right)))

    def compute_body_forces(self):
        left_centers = (self.left_domain[1:] + self.left_domain[:-1]) / 2
        right_centers = (self.right_domain[1:] + self.right_domain[:-1]) / 2
        centers = np.concatenate((left_centers, right_centers))
        return self.body_forces(centers)
