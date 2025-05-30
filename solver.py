import numpy as np
import taichi as ti

from scene import Scene


@ti.data_oriented
class Solver:
    def __init__(self, scene: Scene, h):
        self.scene = scene
        self.h = h

    # function for RK4
    @ti.func
    def rk4_f(self, pos, L_square):
        r = pos.norm()
        one_point_five = ti.cast(1.5, ti.f32)
        return - (L_square * pos * one_point_five) / (r ** 5)

    @ti.func
    def determine_color(self, event_horizon_hit, accretion_disk_hit, pos, accretion_disk_hit_x, accretion_disk_hit_y):
        color = ti.Vector([0.0, 0.0, 0.0])
        # Check if the ray hits the event horizon
        if event_horizon_hit:
            color = ti.Vector([0.0, 0.0, 0.0])  # Black for event horizon
        else:
            # Get the skymap color based on the ray's position
            color = self.scene.skymap.get_color_from_ray_ti(pos)

        # Check if the ray hits the accretion disk
        if accretion_disk_hit:
            # Blend the accretion disk color with the skymap color
            accretion_color = self.scene.get_accretion_disk_color_ti(
                accretion_disk_hit_x, accretion_disk_hit_y
            )
            color = self.scene.accretion_alpha * accretion_color + \
                    (1 - self.scene.accretion_alpha) * color

        return color

    # Forward Euler method
    @ti.kernel
    def solve_forward_euler(self, positions: ti.template(), directions: ti.template(), colors: ti.template()):
        one_point_five = ti.cast(1.5, ti.f32)
        ad_hit_coord = ti.Vector([0.0, 0.0])
        for i, j in positions:
            pos = positions[i, j]
            dir_ = directions[i, j]
            dir_ = dir_.normalized()
            L_square = dir_.cross(pos).norm() ** 2

            event_horizon_hit = False
            while True:
                new_pos = pos + self.h * dir_
                r = new_pos.norm()
                constant = - (L_square * one_point_five) / (r ** 5)
                new_dir = dir_ + self.h * constant * pos

                # Check for event horizon or accretion disk hit

                if pos[2] * new_pos[2] < 0:
                    t = -pos[2] / (new_pos[2] - pos[2])  # remove the +1e-7
                    ad_hit_coord = pos[:2] + t * (new_pos[:2] - pos[:2])

                    if self.scene.accretion_r2 >= ad_hit_coord.norm() >= self.scene.accretion_r1:
                        colors[i, j] = self.scene.get_accretion_disk_color_ti(
                            ad_hit_coord[0], ad_hit_coord[1]) + colors[i, j]

                pos = new_pos
                dir_ = new_dir

                if r < self.scene.blackhole_r:
                    event_horizon_hit = True
                    break
                elif r > self.scene.skymap.r_max:
                    break

            if event_horizon_hit:
                colors[i, j] = ti.Vector(
                    [0.0, 0.0, 0.0]) + self.scene.accretion_alpha * colors[i, j]  # Black for event horizon
            else:
                # Get the skymap color based on the ray's position
                colors[i, j] = self.scene.skymap.get_color_from_ray_ti(
                    pos) + self.scene.accretion_alpha * colors[i, j]

            colors[i, j] = ti.math.clamp(colors[i, j], 0.0, 1.0)

    # Runge-Kutta 4-step method
    @ti.kernel
    def solve_rk4(self, positions: ti.template(), directions: ti.template(), colors: ti.template()):

        for i, j in positions:
            pos = positions[i, j]
            dir_ = directions[i, j]
            L_square = dir_.cross(pos).norm() ** 2

            event_horizon_hit = False
            while True:
                # RK4 integration for position
                k1_pos = self.h * dir_
                k1_dir = self.h * self.rk4_f(pos, L_square)

                k2_pos = self.h * (dir_ + 0.5 * k1_dir)
                k2_dir = self.h * self.rk4_f(pos + 0.5 * k1_pos, L_square)

                k3_pos = self.h * (dir_ + 0.5 * k2_dir)
                k3_dir = self.h * self.rk4_f(pos + 0.5 * k2_pos, L_square)

                k4_pos = self.h * (dir_ + k3_dir)
                k4_dir = self.h * self.rk4_f(pos + k3_pos, L_square)

                new_pos = pos + (k1_pos + 2 * k2_pos + 2 * k3_pos + k4_pos) / 6
                new_dir_ = dir_ + (k1_dir + 2 * k2_dir + 2 * k3_dir + k4_dir) / 6

                # Check for event horizon or accretion disk hit
                if pos[2] * new_pos[2] < 0:
                    t = -pos[2] / (new_pos[2] - pos[2])  # remove the +1e-7
                    ad_hit_coord = pos[:2] + t * (new_pos[:2] - pos[:2])

                    if self.scene.accretion_r2 >= ad_hit_coord.norm() >= self.scene.accretion_r1:
                        colors[i, j] = self.scene.get_accretion_disk_color_ti(
                            ad_hit_coord[0], ad_hit_coord[1]) + colors[i, j]

                # Check if the ray hits the event horizon or the skymap
                r = pos.norm()
                if r < self.scene.blackhole_r:
                    event_horizon_hit = True
                    break
                elif r > self.scene.skymap.r_max:
                    break

                pos = new_pos
                dir_ = new_dir_

            if event_horizon_hit:
                colors[i, j] = ti.Vector(
                    [0.0, 0.0, 0.0]) + self.scene.accretion_alpha * colors[i, j]  # Black for event horizon
            else:
                # Get the skymap color based on the ray's position
                colors[i, j] = self.scene.skymap.get_color_from_ray_ti(
                    pos) + self.scene.accretion_alpha * colors[i, j]

            colors[i, j] = ti.math.clamp(colors[i, j], 0.0, 1.0)

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
            constant = - (L_square * one_point_five) / (r ** 5)
            dir_ = dir_ + 0.5 * self.h * constant * pos

            event_horizon_hit = False
            accretion_disk_hit = False
            while True:
                # Full-step position update
                new_pos = pos + self.h * dir_

                # Recalculate constants with new position
                r = new_pos.norm()
                constant = - (L_square * one_point_five) / (r ** 5)

                # Full-step velocity update
                new_dir_ = dir_ + self.h * constant * pos

                # Check for event horizon or accretion disk hit
                if pos[2] * new_pos[2] < 0:
                    t = -pos[2] / (new_pos[2] - pos[2])  # remove the +1e-7
                    ad_hit_coord = pos[:2] + t * (new_pos[:2] - pos[:2])

                    if self.scene.accretion_r2 >= ad_hit_coord.norm() >= self.scene.accretion_r1:
                        colors[i, j] = self.scene.get_accretion_disk_color_ti(
                            ad_hit_coord[0], ad_hit_coord[1]) + colors[i, j]

                pos = new_pos
                dir_ = new_dir_

                # Check if the ray hits the event horizon or the skymap
                r = ti.sqrt(pos.dot(pos))
                if r < self.scene.blackhole_r:
                    event_horizon_hit = True
                    break
                elif r > self.scene.skymap.r_max:
                    break

            if event_horizon_hit:
                colors[i, j] = ti.Vector(
                    [0.0, 0.0, 0.0]) + self.scene.accretion_alpha * colors[i, j]  # Black for event horizon
            else:
                # Get the skymap color based on the ray's position
                colors[i, j] = self.scene.skymap.get_color_from_ray_ti(
                    pos) + self.scene.accretion_alpha * colors[i, j]

            colors[i, j] = ti.math.clamp(colors[i, j], 0.0, 1.0)

    # Adams-Bashforth 2-step method
    @ti.kernel
    def solve_ab2(self, positions: ti.template(), directions: ti.template(), colors: ti.template()):
        three_over_two = ti.cast(1.5, ti.f32)
        one_over_two = ti.cast(0.5, ti.f32)
        one_point_five = ti.cast(1.5, ti.f32)
        ad_hit_coord = ti.Vector([0.0, 0.0])

        for i, j in positions:
            pos = positions[i, j]
            dir_ = directions[i, j]
            L_square = dir_.cross(pos).norm() ** 2

            # Initialize f_{n-1}
            f_pos_prev = dir_
            r = pos.norm()
            constant = - (L_square * one_point_five) / (r ** 5)
            f_dir_prev = constant * pos

            event_horizon_hit = False
            accretion_disk_hit = False
            while True:
                # Compute f_n
                f_pos_n = dir_
                r = pos.norm()
                constant = - (L_square * one_point_five) / (r ** 5)
                f_dir_n = constant * pos

                new_pos = pos + self.h * (three_over_two * f_pos_n - one_over_two * f_pos_prev)
                new_dir_ = dir_ + self.h * (three_over_two * f_dir_n - one_over_two * f_dir_prev)

                # Check for event horizon or accretion disk hit
                if pos[2] * new_pos[2] < 0:
                    t = -pos[2] / (new_pos[2] - pos[2])  # remove the +1e-7
                    ad_hit_coord = pos[:2] + t * (new_pos[:2] - pos[:2])

                    if self.scene.accretion_r2 >= ad_hit_coord.norm() >= self.scene.accretion_r1:
                        colors[i, j] = self.scene.get_accretion_disk_color_ti(
                            ad_hit_coord[0], ad_hit_coord[1]) + colors[i, j]

                pos = new_pos
                dir_ = new_dir_

                # Update previous function evaluations
                f_pos_prev = f_pos_n
                f_dir_prev = f_dir_n

                # Check if the ray hits the event horizon or the skymap
                if r < self.scene.blackhole_r:
                    event_horizon_hit = True
                    break
                elif r > self.scene.skymap.r_max:
                    break

            if event_horizon_hit:
                colors[i, j] = ti.Vector(
                    [0.0, 0.0, 0.0]) + self.scene.accretion_alpha * colors[i, j]  # Black for event horizon
            else:
                # Get the skymap color based on the ray's position
                colors[i, j] = self.scene.skymap.get_color_from_ray_ti(
                    pos) + self.scene.accretion_alpha * colors[i, j]

            colors[i, j] = ti.math.clamp(colors[i, j], 0.0, 1.0)

    @ti.kernel
    def solve_am4(self, positions: ti.template(), directions: ti.template(), colors: ti.template()):

        # Store coefficients as constants (arrays are now Taichi-compatible)
        coefficients_ab2 = ti.static([3 / 2.0, -1 / 2.0])  # Adams-Bashforth 2-step
        coefficients_ab3 = ti.static([23 / 12.0, -16 / 12.0, 5 / 12.0])  # Adams-Bashforth 3-step
        coefficients_ab = ti.static([9 / 24.0, -19 / 24.0, 5 / 24.0, 1 / 24.0])  # Adams-Bashforth 4-step
        coefficients_am = ti.static([9 / 24.0, 19 / 24.0, -5 / 24.0, -1 / 24.0])  # Adams-Moulton

        for i, j in positions:
            pos = positions[i, j]
            dir_ = directions[i, j]
            L_square = dir_.cross(pos).norm() ** 2

            # Initialize function evaluations
            f_pos_prev = [ti.Vector([0.0, 0.0, 0.0]) for _ in range(4)]
            f_dir_prev = [ti.Vector([0.0, 0.0, 0.0]) for _ in range(4)]

            # Step 1: Use Euler's method for the first step
            f_pos_prev[0] = dir_
            f_dir_prev[0] = self.rk4_f(pos, L_square)
            pos = pos + self.h * f_pos_prev[0]
            dir_ = dir_ + self.h * f_dir_prev[0]

            # Step 2: Use Adams-Bashforth 2-step
            f_pos_prev[1] = dir_
            f_dir_prev[1] = self.rk4_f(pos, L_square)
            f_pos_update = ti.Vector([0.0, 0.0, 0.0])
            f_dir_update = ti.Vector([0.0, 0.0, 0.0])
            for k in ti.static(range(2)):  # Use ti.static for Python-style loops
                f_pos_update += coefficients_ab2[k] * f_pos_prev[1 - k]
                f_dir_update += coefficients_ab2[k] * f_dir_prev[1 - k]
            pos = pos + self.h * f_pos_update
            dir_ = dir_ + self.h * f_dir_update

            # Step 3: Use Adams-Bashforth 3-step
            f_pos_prev[2] = dir_
            f_dir_prev[2] = self.rk4_f(pos, L_square)
            f_pos_update = ti.Vector([0.0, 0.0, 0.0])
            f_dir_update = ti.Vector([0.0, 0.0, 0.0])
            for k in ti.static(range(3)):
                f_pos_update += coefficients_ab3[k] * f_pos_prev[2 - k]
                f_dir_update += coefficients_ab3[k] * f_dir_prev[2 - k]
            pos = pos + self.h * f_pos_update
            dir_ = dir_ + self.h * f_dir_update

            # Start Adams-Moulton 4-step method
            event_horizon_hit = False

            while True:
                # Predictor step: Adams-Bashforth 4-step
                f_pos_predictor = ti.Vector([0.0, 0.0, 0.0])
                f_dir_predictor = ti.Vector([0.0, 0.0, 0.0])
                for k in ti.static(range(4)):
                    f_pos_predictor += coefficients_ab[k] * f_pos_prev[3 - k]
                    f_dir_predictor += coefficients_ab[k] * f_dir_prev[3 - k]

                pos_predictor = pos + self.h * f_pos_predictor
                dir_predictor = dir_ + self.h * f_dir_predictor

                # Corrector step: Adams-Moulton
                f_pos_corrector = dir_predictor
                f_dir_corrector = self.rk4_f(pos_predictor, L_square)

                f_pos_update = ti.Vector([0.0, 0.0, 0.0])
                f_dir_update = ti.Vector([0.0, 0.0, 0.0])
                for k in ti.static(range(4)):
                    if k < 3:
                        f_pos_update += coefficients_am[k] * f_pos_prev[3 - k]
                        f_dir_update += coefficients_am[k] * f_dir_prev[3 - k]
                    else:
                        f_pos_update += coefficients_am[k] * f_pos_corrector
                        f_dir_update += coefficients_am[k] * f_dir_corrector

                new_pos = pos + self.h * f_pos_update
                new_dir_ = dir_ + self.h * f_dir_update

                # Check for event horizon or accretion disk hit
                if pos[2] * new_pos[2] < 0:
                    t = -pos[2] / (new_pos[2] - pos[2])  # remove the +1e-7
                    ad_hit_coord = pos[:2] + t * (new_pos[:2] - pos[:2])

                    if self.scene.accretion_r2 >= ad_hit_coord.norm() >= self.scene.accretion_r1:
                        colors[i, j] = self.scene.get_accretion_disk_color_ti(
                            ad_hit_coord[0], ad_hit_coord[1]) + colors[i, j]

                pos = new_pos
                dir_ = new_dir_

                # Shift previous function evaluations
                for k in ti.static(range(3)):  # Reverse logic manually
                    f_pos_prev[3 - k] = f_pos_prev[2 - k]
                    f_dir_prev[3 - k] = f_dir_prev[2 - k]

                f_pos_prev[0] = dir_
                f_dir_prev[0] = self.rk4_f(pos, L_square)

                # Check if the ray hits the event horizon or the skymap
                r = pos.norm()
                if r < self.scene.blackhole_r:
                    event_horizon_hit = True
                    break
                elif r > self.scene.skymap.r_max:
                    break

            if event_horizon_hit:
                colors[i, j] = ti.Vector(
                    [0.0, 0.0, 0.0]) + self.scene.accretion_alpha * colors[i, j]  # Black for event horizon
            else:
                # Get the skymap color based on the ray's position
                colors[i, j] = self.scene.skymap.get_color_from_ray_ti(
                    pos) + self.scene.accretion_alpha * colors[i, j]

            colors[i, j] = ti.math.clamp(colors[i, j], 0.0, 1.0)
