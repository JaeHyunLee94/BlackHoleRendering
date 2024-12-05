import numpy as np
import taichi as ti

from scene import Scene

# function for RK4
@ti.func
def rk4_f(pos, dir_, L_square):
    r = pos.norm()
    r_fourth = r ** 4
    one_point_five = ti.cast(1.5, ti.f32)
    return (L_square * pos / r_fourth) * (1 - one_point_five / r)

@ti.data_oriented
class Solver:
    def __init__(self, scene: Scene, delta_lambda):
        self.scene = scene
        self.delta_lambda = delta_lambda

    @ti.kernel
    def solve_forward_euler(self, positions: ti.template(), directions: ti.template(), colors: ti.template()):

        for i, j in positions:
            pos = positions[i, j]
            dir_ = directions[i, j]
            L_square = dir_.cross(pos).norm() ** 2

            event_horizon_hit = False
            accretion_disk_hit = False
            while True:
                new_pos = pos + self.delta_lambda * dir_
                r = new_pos.norm()
                r_fourth = r ** 4
                # Ensure constants are float32
                one_point_five = ti.cast(1.5, ti.f32)
                constant = (L_square / r_fourth) * (1 - one_point_five / r)
                new_dir = dir_ + self.delta_lambda * constant * pos

                pos = new_pos
                dir_ = new_dir

                # Check for event horizon or accretion disk hit
                # if ti.abs(pos[2])<=0.05 and pos[:2].norm() >= self.scene.accretion_r1 and pos[:2].norm() <= self.scene.accretion_r2:
                #     accretion_disk_hit = True
                #     break
                if r < self.scene.blackhole_r:
                    event_horizon_hit = True
                    break
                elif r > self.scene.skymap.r_max:
                    break

            positions[i, j] = pos
            directions[i, j] = dir_

            if event_horizon_hit:
                colors[i, j] = ti.Vector([0.0, 0.0, 0.0])
            # elif self.accretion_disk_hit:
            #     colors[i, j] = ti.Vector([1.0, 1.0, 1.0])
            else:
                 colors[i, j] = self.scene.skymap.get_color_from_ray_ti(dir_)

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
                k1_pos = self.delta_lambda * dir_
                k1_dir = self.delta_lambda * rk4_f(pos, dir_, L_square)

                k2_pos = self.delta_lambda * (dir_ + 0.5 * k1_dir)
                k2_dir = self.delta_lambda * rk4_f(pos + 0.5 * k1_pos, dir_ + 0.5 * k1_dir, L_square)

                k3_pos = self.delta_lambda * (dir_ + 0.5 * k2_dir)
                k3_dir = self.delta_lambda * rk4_f(pos + 0.5 * k2_pos, dir_ + 0.5 * k2_dir, L_square)
                
                k4_pos = self.delta_lambda * (dir_ + k3_dir)
                k4_dir = self.delta_lambda * rk4_f(pos + k3_pos, dir_ + k3_dir, L_square)

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

    @ti.kernel
    def solve_leapfrog(self, positions: ti.template(), directions: ti.template(), colors: ti.template()):

        for i, j in positions:
            pos = positions[i, j]
            dir_ = directions[i, j]
            L_square = dir_.cross(pos).norm() ** 2

            # Half-step velocity update
            r = pos.norm()
            r_fourth = r ** 4
            one_point_five = ti.cast(1.5, ti.f32)
            constant = (L_square / r_fourth) * (1 - one_point_five / r)
            dir_ = dir_ + 0.5 * self.delta_lambda * constant * pos

            event_horizon_hit = False
            accretion_disk_hit = False
            while True:
                # Full-step position update
                pos = pos + self.delta_lambda * dir_

                # Recalculate constants with new position
                r = pos.norm()
                r_fourth = r ** 4
                constant = (L_square / r_fourth) * (1 - one_point_five / r)

                # Full-step velocity update
                dir_ = dir_ + self.delta_lambda * constant * pos

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

                # Adams-Bashforth 2-step method
                pos = pos + self.delta_lambda * (three_over_two * f_pos_n - one_over_two * f_pos_prev)
                dir_ = dir_ + self.delta_lambda * (three_over_two * f_dir_n - one_over_two * f_dir_prev)

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
                
                
    @ti.kernel
    def solve_am4(self, positions: ti.template(), directions: ti.template(), colors: ti.template()):

        nine_over_forty = ti.cast(9 / 40.0, ti.f32)
        thirtytwo_over_forty = ti.cast(32 / 40.0, ti.f32)
        twelve_over_forty = ti.cast(12 / 40.0, ti.f32)
        seven_over_forty = ti.cast(7 / 40.0, ti.f32)

        for i, j in positions:
            pos = positions[i, j]
            dir_ = directions[i, j]
            L_square = dir_.cross(pos).norm() ** 2

            # Initialize f_{n-3}, f_{n-2}, f_{n-1}
            f_pos_prev3 = dir_
            r = pos.norm()
            r_fourth = r ** 4
            constant = (L_square / r_fourth) * (1 - nine_over_forty / r)
            f_dir_prev3 = constant * pos

            pos_prev2 = pos
            dir_prev2 = dir_
            r = pos_prev2.norm()
            r_fourth = r ** 4
            constant = (L_square / r_fourth) * (1 - nine_over_forty / r)
            f_pos_prev2 = dir_prev2
            f_dir_prev2 = constant * pos_prev2

            pos_prev1 = pos
            dir_prev1 = dir_
            r = pos_prev1.norm()
            r_fourth = r ** 4
            constant = (L_square / r_fourth) * (1 - nine_over_forty / r)
            f_pos_prev1 = dir_prev1
            f_dir_prev1 = constant * pos_prev1

            event_horizon_hit = False
            accretion_disk_hit = False
            while True:
                # Predictor step (Adams-Bashforth as predictor)
                f_pos_predictor = dir_
                r = pos.norm()
                r_fourth = r ** 4
                constant = (L_square / r_fourth) * (1 - nine_over_forty / r)
                f_dir_predictor = constant * pos

                pos_predictor = pos + self.delta_lambda * (
                    thirtytwo_over_forty * f_pos_prev1
                    - twelve_over_forty * f_pos_prev2
                    + seven_over_forty * f_pos_prev3
                )
                dir_predictor = dir_ + self.delta_lambda * (
                    thirtytwo_over_forty * f_dir_prev1
                    - twelve_over_forty * f_dir_prev2
                    + seven_over_forty * f_dir_prev3
                )

                # Corrector step (Adams-Moulton)
                r = pos_predictor.norm()
                r_fourth = r ** 4
                constant = (L_square / r_fourth) * (1 - nine_over_forty / r)
                f_pos_corrector = dir_predictor
                f_dir_corrector = constant * pos_predictor

                pos_new = pos + self.delta_lambda * (
                    nine_over_forty * f_pos_corrector
                    + thirtytwo_over_forty * f_pos_prev1
                    - twelve_over_forty * f_pos_prev2
                    + seven_over_forty * f_pos_prev3
                )
                dir_new = dir_ + self.delta_lambda * (
                    nine_over_forty * f_dir_corrector
                    + thirtytwo_over_forty * f_dir_prev1
                    - twelve_over_forty * f_dir_prev2
                    + seven_over_forty * f_dir_prev3
                )

                pos, dir_ = pos_new, dir_new

                # Update previous function evaluations
                f_pos_prev3 = f_pos_prev2
                f_dir_prev3 = f_dir_prev2
                f_pos_prev2 = f_pos_prev1
                f_dir_prev2 = f_dir_prev1
                f_pos_prev1 = f_pos_predictor
                f_dir_prev1 = f_dir_predictor

                r = pos.norm()
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