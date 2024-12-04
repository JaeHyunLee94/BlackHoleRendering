import numpy as np
import taichi as ti

@ti.data_oriented
class Solver:
    def __init__(self, blackhole_radius, skymap=None):
        self.blackhole_radius = blackhole_radius
        self.skymap = skymap

    @ti.kernel
    def solve_forward_euler(self, positions: ti.template(), directions: ti.template(), colors: ti.template()):
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
                if r < self.blackhole_radius:
                    event_horizon_hit = True
                    break
            positions[i, j] = pos
            directions[i, j] = dir_
            if event_horizon_hit:
                colors[i, j] = ti.Vector([0.0, 0.0, 0.0])
            else:
                D = dir_.normalized()
                colors[i, j] = self.skymap.get_color_from_ray_ti(D)

    @ti.kernel
    def solve_rk4(self, positions: ti.template(), directions: ti.template(), colors: ti.template()):
        max_iter = 1000
        delta_lambda = ti.cast(0.05, ti.f32)  # Ensure delta_lambda is float32

        for i, j in positions:
            pos = positions[i, j]
            dir_ = directions[i, j]
            L = dir_.cross(pos).norm()
            event_horizon_hit = False

            for iter in range(max_iter):
                # RK4 integration for position
                k1_pos = delta_lambda * dir_
                k1_dir = delta_lambda * rk4_f(pos, dir_, L)

                k2_pos = delta_lambda * (dir_ + 0.5 * k1_dir)
                k2_dir = delta_lambda * rk4_f(pos + 0.5 * k1_pos, dir_ + 0.5 * k1_dir, L)

                k3_pos = delta_lambda * (dir_ + 0.5 * k2_dir)
                k3_dir = delta_lambda * rk4_f(pos + 0.5 * k2_pos, dir_ + 0.5 * k2_dir, L)
                
                k4_pos = delta_lambda * (dir_ + k3_dir)
                k4_dir = delta_lambda * rk4_f(pos + k3_pos, dir_ + k3_dir, L)

                pos = pos + (k1_pos + 2 * k2_pos + 2 * k3_pos + k4_pos) / 6
                dir_ = dir_ + (k1_dir + 2 * k2_dir + 2 * k3_dir + k4_dir) / 6

                # Check if the ray hits the event horizon
                r = ti.sqrt(pos.dot(pos))
                if r < self.blackhole_radius:
                    event_horizon_hit = True
                    break

            positions[i, j] = pos
            directions[i, j] = dir_
            if event_horizon_hit:
                colors[i, j] = ti.Vector([0.0, 0.0, 0.0])
            else:
                D = dir_.normalized()
                colors[i, j] = self.skymap.get_color_from_ray_ti(D)