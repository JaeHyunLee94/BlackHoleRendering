import numpy as np
import taichi as ti

@ti.data_oriented
class Solver:
    def __init__(self, blackhole, skymap=None):
        self.blackhole = blackhole
        self.skymap = skymap

    @ti.kernel
    def solve_forward_euler(self, positions: ti.template(), directions: ti.template(), colors: ti.template()):
        blackhole_pos = ti.Vector([self.blackhole.pos[0], self.blackhole.pos[1], self.blackhole.pos[2]])
        blackhole_radius = self.blackhole.radius
        max_iter = 1000
        delta_lambda = ti.cast(0.05, ti.f32)  # Ensure delta_lambda is float32

        for i, j in positions:
            pos = positions[i, j]
            dir_ = directions[i, j]
            L = dir_.cross(pos).norm()
            event_horizon_hit = False
            for iter in range(max_iter):
                new_pos = pos + delta_lambda * dir_
                r_square = new_pos.dot(new_pos)
                r_fourth = r_square * r_square
                r = ti.sqrt(r_square)
                # Ensure constants are float32
                one_point_five = ti.cast(1.5, ti.f32)
                constant = (L * L / r_fourth) * (1 - one_point_five / r)
                new_dir = dir_ + delta_lambda * constant * pos

                pos = new_pos
                dir_ = new_dir
                if r < blackhole_radius:
                    event_horizon_hit = True
                    break
            positions[i, j] = pos
            directions[i, j] = dir_
            if event_horizon_hit:
                colors[i, j] = ti.Vector([0.0, 0.0, 0.0])
            else:
                D = dir_.normalized()
                colors[i, j] = self.skymap.get_color_from_ray_ti(D)