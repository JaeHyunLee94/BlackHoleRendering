import numpy as np


class Solver:
    def __init__(self, scene):
        self._scene = scene
        pass

    def set_scene(self, black_holes):
        self._scene = black_holes

    def solve(self, ray):
        """
        TODO: Determine the color of the ray.
        Temporal implementation ^^b
        """
        for bh in self._scene:
            origin_to_center = ray.origin - bh.pos
            a = np.dot(ray.direction, ray.direction)
            b = 2 * np.dot(ray.direction, origin_to_center)
            c = np.dot(origin_to_center, origin_to_center) - bh.radius ** 2
            discriminant = b ** 2 - 4 * a * c

            if discriminant > 0:
                ray.color = np.array([0, 0, 0])
            else:
                pass




