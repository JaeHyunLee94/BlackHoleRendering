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

    def solve_forward_euler(self, ray):
        L = np.linalg.norm(np.cross(ray.position, ray.direction))
        delta_lambda = 0.05
        max_iter = 1000
        pos = ray.position.copy()
        dir_ = ray.direction.copy()
        event_horizon_hit = False
        for i in range(max_iter):
            new_pos = pos + delta_lambda * dir_
            r_square = np.dot(new_pos, new_pos)
            r_fourth = r_square ** 2
            r = np.sqrt(r_square)
            constant = (L ** 2 / r_fourth) * (1 - 1.5 / r)
            new_dir = dir_ + delta_lambda * constant * pos  # Added negative sign

            pos = new_pos.copy()
            dir_ = new_dir.copy()
            if r < self.blackhole.radius:
                event_horizon_hit = True
                break

        ray.position = pos.copy()  # Update ray's position
        ray.direction = dir_.copy()  # Update ray's direction
        if event_horizon_hit:
            ray.color = np.array([0, 0, 0])
        else:
            ray.color = self.skymap.get_color_from_ray(ray)
