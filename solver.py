import numpy as np


class Solver:
    def __init__(self, blackhole, skymap=None):
        self.blackhole = blackhole
        self.skymap = skymap

    def set_scene(self, blackhole):
        self.blackhole = blackhole

    def solve(self, ray):
        """
        TODO: Determine the color of the ray.
        Temporal implementation ^^b
        """
        intersect_with_bh = False
        origin_to_center = ray.position - self.blackhole.pos
        a = np.dot(ray.direction, ray.direction)
        b = 2 * np.dot(ray.direction, origin_to_center)
        c = np.dot(origin_to_center, origin_to_center) - self.blackhole.radius ** 2
        discriminant = b ** 2 - 4 * a * c

        if discriminant > 0:
            intersect_with_bh = True

        if intersect_with_bh:
            ray.color = np.array([0, 0, 0])
        else:
            ray.color = self.skymap.get_color_from_ray(ray)
