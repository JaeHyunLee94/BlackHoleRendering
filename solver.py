import numpy as np
import taichi as ti

from scene import Scene

@ti.func
def rk4_f(pos, dir_, L_square):
    # function for RK4
    r = pos.norm()
    r_fourth = r ** 4
    one_point_five = ti.cast(1.5, ti.f32)
    return (L_square * pos / r_fourth) * (1 - one_point_five / r)

@ti.data_oriented
class Solver:
    def __init__(self, scene: Scene):
        self.scene = scene

    @ti.kernel
    def solve_forward_euler(self, positions: ti.template(), directions: ti.template(), colors: ti.template()):
        max_iter = 1000
        delta_lambda = ti.cast(0.05, ti.f32)  # Ensure delta_lambda is float32

        for i, j in positions:
            pos = positions[i, j]
            dir_ = directions[i, j]
            L_square = dir_.cross(pos).norm() ** 2
            event_horizon_hit = False
            accretion_disk_hit = False # test
            for iter in range(max_iter):
                new_pos = pos + delta_lambda * dir_
                r = new_pos.norm()
                r_fourth = r ** 4
                # Ensure constants are float32
                one_point_five = ti.cast(1.5, ti.f32)
                constant = (L_square / r_fourth) * (1 - one_point_five / r)
                new_dir = dir_ + delta_lambda * constant * pos

                pos = new_pos
                dir_ = new_dir

                # Accretion disk test
                if ti.abs(pos[2])<=0.05 and pos[:2].norm() >= self.scene.accretion_r1 and pos[:2].norm() <= self.scene.accretion_r2:
                    accretion_disk_hit = True
                    break

                if r < self.scene.blackhole_r:
                    event_horizon_hit = True
                    break
            positions[i, j] = pos
            directions[i, j] = dir_
            # if event_horizon_hit:
            #     colors[i, j] = ti.Vector([0.0, 0.0, 0.0])
            # else:
            #     D = dir_.normalized()
            #     colors[i, j] = self.scene.skymap.get_color_from_ray_ti(D)
            if event_horizon_hit:
                colors[i, j] = ti.Vector([0.0, 0.0, 0.0])
            elif accretion_disk_hit:
                colors[i, j] = ti.Vector([1.0, 1.0, 1.0])
            else:
                D = dir_.normalized()
                colors[i, j] = self.scene.skymap.get_color_from_ray_ti(D)

    @ti.kernel
    def solve_rk4(self, positions: ti.template(), directions: ti.template(), colors: ti.template()):
        max_iter = 1000
        delta_lambda = ti.cast(0.05, ti.f32)  # Ensure delta_lambda is float32

        for i, j in positions:
            pos = positions[i, j]
            dir_ = directions[i, j]
            L_square = dir_.cross(pos).norm() ** 2
            event_horizon_hit = False

            for iter in range(max_iter):
                # RK4 integration for position
                k1_pos = delta_lambda * dir_
                k1_dir = delta_lambda * rk4_f(pos, dir_, L_square)

                k2_pos = delta_lambda * (dir_ + 0.5 * k1_dir)
                k2_dir = delta_lambda * rk4_f(pos + 0.5 * k1_pos, dir_ + 0.5 * k1_dir, L_square)

                k3_pos = delta_lambda * (dir_ + 0.5 * k2_dir)
                k3_dir = delta_lambda * rk4_f(pos + 0.5 * k2_pos, dir_ + 0.5 * k2_dir, L_square)
                
                k4_pos = delta_lambda * (dir_ + k3_dir)
                k4_dir = delta_lambda * rk4_f(pos + k3_pos, dir_ + k3_dir, L_square)

                pos = pos + (k1_pos + 2 * k2_pos + 2 * k3_pos + k4_pos) / 6
                dir_ = dir_ + (k1_dir + 2 * k2_dir + 2 * k3_dir + k4_dir) / 6

                # Check if the ray hits the event horizon
                r = ti.sqrt(pos.dot(pos))
                if r < self.scene.blackhole_r:
                    event_horizon_hit = True
                    break

            positions[i, j] = pos
            directions[i, j] = dir_
            if event_horizon_hit:
                colors[i, j] = ti.Vector([0.0, 0.0, 0.0])
            else:
                D = dir_.normalized()
                colors[i, j] = self.scene.skymap.get_color_from_ray_ti(D)

    @ti.kernel
    def solve_leapfrog(self, positions: ti.template(), directions: ti.template(), colors: ti.template()):
        max_iter = 1000
        delta_lambda = ti.cast(0.05, ti.f32)  # Ensure delta_lambda is float32

        for i, j in positions:
            pos = positions[i, j]
            dir_ = directions[i, j]
            L_square = dir_.cross(pos).norm() ** 2
            event_horizon_hit = False

            # Half-step velocity update
            r = pos.norm()
            r_fourth = r ** 4
            one_point_five = ti.cast(1.5, ti.f32)
            constant = (L_square / r_fourth) * (1 - one_point_five / r)
            dir_ = dir_ + 0.5 * delta_lambda * constant * pos

            for iter in range(max_iter):
                # Full-step position update
                pos = pos + delta_lambda * dir_

                # Recalculate constants with new position
                r = pos.norm()
                r_fourth = r ** 4
                constant = (L_square / r_fourth) * (1 - one_point_five / r)

                # Full-step velocity update
                dir_ = dir_ + delta_lambda * constant * pos

                # Check if it hits the event horizon
                if r < self.scene.blackhole_r:
                    event_horizon_hit = True
                    break

            # Store results
            positions[i, j] = pos
            directions[i, j] = dir_
            if event_horizon_hit:
                colors[i, j] = ti.Vector([0.0, 0.0, 0.0])
            else:
                D = dir_.normalized()
                colors[i, j] = self.scene.skymap.get_color_from_ray_ti(D)
