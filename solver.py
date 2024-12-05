import numpy as np
import taichi as ti

from scene import Scene

<<<<<<< HEAD

=======
# function for RK4
>>>>>>> d44b278aa6fb9e44c95d35dafdac9ddea1769a77
@ti.func
def rk4_f(pos, L_square):
    r = pos.norm()
    r_fourth = r ** 4
    one_point_five = ti.cast(1.5, ti.f32)
    return (L_square * pos / r_fourth) * (1 - one_point_five / r)


@ti.data_oriented
class Solver:
    def __init__(self, scene: Scene, h):
        self.scene = scene
        self.h = h

    # Forward Euler method
    @ti.kernel
    def solve_forward_euler(self, positions: ti.template(), directions: ti.template(), colors: ti.template()):
<<<<<<< HEAD
        max_iter = 10000
        delta_lambda = ti.cast(0.05, ti.f32)  # Ensure delta_lambda is float32
=======

        one_point_five = ti.cast(1.5, ti.f32)
>>>>>>> d44b278aa6fb9e44c95d35dafdac9ddea1769a77

        for i, j in positions:
            pos = positions[i, j]
            dir_ = directions[i, j]
            L_square = dir_.cross(pos).norm() ** 2

            event_horizon_hit = False
<<<<<<< HEAD
            accretion_disk_hit = False  # test
            for iter in range(max_iter):
                new_pos = pos + delta_lambda * dir_
=======
            accretion_disk_hit = False

            while True:
                new_pos = pos + self.h * dir_
>>>>>>> d44b278aa6fb9e44c95d35dafdac9ddea1769a77
                r = new_pos.norm()
                r_fourth = r ** 4
                # Ensure constants are float32
                constant = (L_square / r_fourth) * (1 - one_point_five / r)
                new_dir = dir_ + self.h * constant * pos

                pos = new_pos
                dir_ = new_dir

<<<<<<< HEAD
                # Accretion disk test
                if ti.abs(pos[2]) <= 0.05 and pos[:2].norm() >= self.scene.accretion_r1 and pos[
                                                                                            :2].norm() <= self.scene.accretion_r2:
=======
                # Check for event horizon or accretion disk hit
                if ti.abs(pos[2])<=0.1 and pos[:2].norm() >= self.scene.accretion_r1 and pos[:2].norm() <= self.scene.accretion_r2:
>>>>>>> d44b278aa6fb9e44c95d35dafdac9ddea1769a77
                    accretion_disk_hit = True
                    break
            
                if r < self.scene.blackhole_r:
                    event_horizon_hit = True
                    colors[i, j] = 0.5 * ti.Vector([1.0, 1.0, 1.0])
                    break
                elif r > self.scene.skymap.r_max:
                    break

            positions[i, j] = pos
            directions[i, j] = dir_

            if event_horizon_hit:
                colors[i, j] = ti.Vector([0.0, 0.0, 0.0])
            elif accretion_disk_hit:
                D = dir_.normalized()

                colors[i, j] += 0.5 * self.scene.skymap.get_color_from_ray_ti(D)
            else:
                 colors[i, j] = self.scene.skymap.get_color_from_ray_ti(dir_)

    # Runge-Kutta 4-step method
    @ti.kernel
    def solve_rk4(self, positions: ti.template(), directions: ti.template(), colors: ti.template()):

        for i, j in positions:
            pos = positions[i, j]
            dir_ = directions[i, j]
            L_square = dir_.cross(pos).norm() ** 2

            event_horizon_hit = False
            accretion_disk_hit = False
            while True:
                # RK4 integration for position
                k1_pos = self.h * dir_
                k1_dir = self.h * rk4_f(pos, L_square)

                k2_pos = self.h * (dir_ + 0.5 * k1_dir)
                k2_dir = self.h * rk4_f(pos + 0.5 * k1_pos, L_square)

<<<<<<< HEAD
                k3_pos = delta_lambda * (dir_ + 0.5 * k2_dir)
                k3_dir = delta_lambda * rk4_f(pos + 0.5 * k2_pos, dir_ + 0.5 * k2_dir, L_square)

                k4_pos = delta_lambda * (dir_ + k3_dir)
                k4_dir = delta_lambda * rk4_f(pos + k3_pos, dir_ + k3_dir, L_square)
=======
                k3_pos = self.h * (dir_ + 0.5 * k2_dir)
                k3_dir = self.h * rk4_f(pos + 0.5 * k2_pos, L_square)
                
                k4_pos = self.h * (dir_ + k3_dir)
                k4_dir = self.h * rk4_f(pos + k3_pos, L_square)
>>>>>>> d44b278aa6fb9e44c95d35dafdac9ddea1769a77

                pos = pos + (k1_pos + 2 * k2_pos + 2 * k3_pos + k4_pos) / 6
                dir_ = dir_ + (k1_dir + 2 * k2_dir + 2 * k3_dir + k4_dir) / 6

                # Check if the ray hits the event horizon or the skymap
                r = ti.sqrt(pos.dot(pos))
                if r < self.scene.blackhole_r:
                    event_horizon_hit = True
                    break
                elif r > self.scene.skymap.r_max:
                    break

            positions[i, j] = pos
            directions[i, j] = dir_

            if event_horizon_hit:
                colors[i, j] = ti.Vector([0.0, 0.0, 0.0])
            else:
                 colors[i, j] = self.scene.skymap.get_color_from_ray_ti(dir_)

    # Leapfrog method
    @ti.kernel
    def solve_leapfrog(self, positions: ti.template(), directions: ti.template(), colors: ti.template()):
        
        one_point_five = ti.cast(1.5, ti.f32)

        for i, j in positions:
            pos = positions[i, j]
            dir_ = directions[i, j]
            L_square = dir_.cross(pos).norm() ** 2

            # Half-step velocity update
            r = pos.norm()
            r_fourth = r ** 4
            constant = (L_square / r_fourth) * (1 - one_point_five / r)
            dir_ = dir_ + 0.5 * self.h * constant * pos

            event_horizon_hit = False
            accretion_disk_hit = False
            while True:
                # Full-step position update
                pos = pos + self.h * dir_

                # Recalculate constants with new position
                r = pos.norm()
                r_fourth = r ** 4
                constant = (L_square / r_fourth) * (1 - one_point_five / r)

                # Full-step velocity update
                dir_ = dir_ + self.h * constant * pos

                # Check if the ray hits the event horizon or the skymap
                if r < self.scene.blackhole_r:
                    event_horizon_hit = True
                    break
                elif r > self.scene.skymap.r_max:
                    break

            # Store results
            positions[i, j] = pos
            directions[i, j] = dir_

            if event_horizon_hit:
                colors[i, j] = ti.Vector([0.0, 0.0, 0.0])
            else:
                 colors[i, j] = self.scene.skymap.get_color_from_ray_ti(dir_)

    # Adams-Bashforth 2-step method
    @ti.kernel
    def solve_ab2(self, positions: ti.template(), directions: ti.template(), colors: ti.template()):

        three_over_two = ti.cast(1.5, ti.f32)
        one_over_two = ti.cast(0.5, ti.f32)
        one_point_five = ti.cast(1.5, ti.f32)

        for i, j in positions:
            pos = positions[i, j]
            dir_ = directions[i, j]
            L_square = dir_.cross(pos).norm() ** 2

            # Initialize f_{n-1}
            f_pos_prev = dir_
            r = pos.norm()
            r_fourth = r ** 4
            constant = (L_square / r_fourth) * (1 - one_point_five / r)
            f_dir_prev = constant * pos

            event_horizon_hit = False
            accretion_disk_hit = False
            while True:
                # Compute f_n
                f_pos_n = dir_
                r = pos.norm()
                r_fourth = r ** 4
                constant = (L_square / r_fourth) * (1 - one_point_five / r)
                f_dir_n = constant * pos

                pos = pos + self.h * (three_over_two * f_pos_n - one_over_two * f_pos_prev)
                dir_ = dir_ + self.h * (three_over_two * f_dir_n - one_over_two * f_dir_prev)

                # Update previous function evaluations
                f_pos_prev = f_pos_n
                f_dir_prev = f_dir_n

                # Check if the ray hits the event horizon or the skymap
                if r < self.scene.blackhole_r:
                    event_horizon_hit = True
                    break
                elif r > self.scene.skymap.r_max:
                    break

            # Store results
            positions[i, j] = pos
            directions[i, j] = dir_
            if event_horizon_hit:
                colors[i, j] = ti.Vector([0.0, 0.0, 0.0])
            else:
                 colors[i, j] = self.scene.skymap.get_color_from_ray_ti(dir_)