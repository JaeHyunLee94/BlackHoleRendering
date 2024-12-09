# main python script for running and solving Blackhole ODE

# Written by
#   - JaeHyun Lee: jaehyun.lee@wisc.edu
#   - Joon Suk Huh: jhuh23@wisc.edu
#   - Suenggwan Jo: sjo32@wisc.edu
#   - Hyeong Kyu Choi: hyeongkyu.choi@wisc.edu
import argparse
import numpy as np
import matplotlib.pyplot as plt

from camera import Camera
from solver import Solver
from skymap import Skymap
from scene import Scene

import taichi as ti


def main():
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Parse rendering parameters.")

    # Camera position
    parser.add_argument(
        "-pov", "-p", nargs=3, metavar=('x', 'y', 'z'),
        help="Camera position in cartesian coordinate (default: [0,3,0.2])",
        type=float,
        default=[0, 3.0, 0.2]
    )

    # Focal length (float)
    parser.add_argument("-focal", "-f", type=float,
                        default=1.5,
                        help="Focal length (default: 1.5)")

    # Field of View (FoV) (float between 0 and 180)
    parser.add_argument(
        "-fov",
        type=float,
        default=90,
        help="Field of View (FoV) in degrees (float between 0 and 180) (default: 90)"
    )

    # Resolution (string: '4k' or 'fhd')
    parser.add_argument(
        "-resolution", "-r",
        type=str,
        default='4k',
        choices=["4k", "fhd"],
        help="Resolution: '4k' or 'fhd' (default: 4k)"
    )

    # Texture file path (string)
    parser.add_argument("-texture", "-t", type=str,
                        default='texture/high_res/space_texture_high1.jpg',
                        help="Texture file path (string)")

    # Accretion disk texture file path (string)
    parser.add_argument("-at", type=str,
                        default='texture/ad/adisk.jpg',
                        help="Accretion disk texture file path (string)")

    # GPU or CPU flag (use '--gpu' for GPU, default is GPU)
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU for rendering (default: use GPU)"
    )

    # Output file name base
    parser.add_argument("-output", "-o", type=str,
                        default='result',
                        help="Base output file name. (default: result)")

    # accretion r1 r2
    parser.add_argument("-ar1", type=float,
                        default=2,
                        help="inner radius of accretion disk (default: 2)")
    # accretion r1 r2
    parser.add_argument("-ar2", type=float,
                        default=6,
                        help="outer radius of accretion disk (default: 6)")

    args = parser.parse_args()

    # Initialize Taichi
    if args.cpu:
        ti.init(arch=ti.cpu)
    else:
        # On Apple Silicon, use Metal explicitly
        ti.init(arch=ti.metal)

        # Determine resolution
    if args.resolution == '4k':
        resol = np.array([3840, 2160])
        img_width, img_height = 3840, 2160
    else:
        resol = np.array([1920, 1080])
        img_width, img_height = 1920, 1080

    print('Welcome to Math/CS714 Project')

    # Setup camera and get rays as numpy arrays
    my_camera = Camera(np.array(args.pov, dtype=np.float32),
                       np.float32(args.focal),
                       np.array([0, 0, 0], dtype=np.float32),
                       resol, fov=np.float32(args.fov % 180))
    print('Generating rays...')

    # Generate and convert rays into numpy arrays
    original_positions_taichi, original_directions_taichi = my_camera.get_all_rays()

    # Convert Taichi fields to numpy arrays
    original_positions = original_positions_taichi.to_numpy()
    original_directions = original_directions_taichi.to_numpy()

    # Initialize positions and directions Taichi fields
    positions = ti.Vector.field(3, dtype=ti.f32, shape=(img_width, img_height))
    directions = ti.Vector.field(3, dtype=ti.f32, shape=(img_width, img_height))
    colors = ti.Vector.field(3, dtype=ti.f32, shape=(img_width, img_height))

    # Use the numpy arrays to reset Taichi fields before each experiment

    # Initialize scene
    scene = Scene(
        blackhole_r=ti.cast(1.0, ti.f32),
        accretion_r1=ti.cast(args.ar1, ti.f32),
        accretion_r2=ti.cast(args.ar2, ti.f32),
        accretion_temp=ti.cast(400., ti.f32),
        accretion_alpha=ti.cast(1, ti.f32),
        skymap=Skymap(args.texture, r_max=10)
    )
    scene.set_accretion_disk_texture(args.at)

    # Lists of integrators and lambdas to experiment with
    # integrators = ["euler", "rk4", "leapfrog", "ab2", "am4"]
    # lamb_values = [0.1, 0.05, 0.01, 0.001]
    my_solver = Solver(scene, h=ti.cast(0.1, ti.f32))
    integrator = "am4"
    h = 0.001
    print(f"Running integrator: {integrator}, lambda: {h}")

    # Reinitialize solver

    # Reset fields before each run
    colors.fill(0.0)  # start from a clean color field
    positions.from_numpy(original_positions)  # Reload original ray positions
    directions.from_numpy(original_directions)  # Reload original ray directions

    # Solve ODE
    print('Solving ODE...')
    if integrator == "euler":
        my_solver.solve_forward_euler(positions, directions, colors)
    elif integrator == 'rk4':
        my_solver.solve_rk4(positions, directions, colors)
    elif integrator == 'leapfrog':
        my_solver.solve_leapfrog(positions, directions, colors)
    elif integrator == 'ab2':
        my_solver.solve_ab2(positions, directions, colors)
    elif integrator == 'am4':
        my_solver.solve_am4(positions, directions, colors)

    # Render and save
    print('Rendering...')
    img = my_camera.render(colors)
    output_filename = f"experiment_lambda_size/result_{integrator}_lambda_{h}.png"
    plt.figure(figsize=(img_width / 100, img_height / 100), dpi=100)
    plt.imshow(np.transpose(img, (1, 0, 2)))
    plt.axis('off')
    plt.savefig(output_filename, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved: {output_filename}")

    # Create an argument parser


if __name__ == '__main__':
    main()
