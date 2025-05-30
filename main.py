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
        help="Camera position in cartesian coordinate (default: [1,1,1] )",
        type=float,
        default=[6, 0, 0.5]
    )

    # Focal length (float)
    parser.add_argument("-focal", "-f", type=float,
                        default=1.8,
                        help="Focal length (default: 1.8)")

    # Field of View (FoV) (float between 0 and 180)
    parser.add_argument(
        "-fov",
        type=float,
        default=60,
        help="Field of View (FoV) in degrees (float between 0 and 180) (default: 60)"
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

    # Integrator (string: 'euler' or 'rk4')
    parser.add_argument(
        "-integrator", "-i",
        type=str,
        default='am4',
        choices=["euler", "rk4", "leapfrog", "ab2", "am4"],
        help="Integrators: 'euler', 'rk4', 'leapfrog'. (default: am4)"
    )

    # GPU or CPU flag (use '--gpu' for GPU, default is CPU)
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU for rendering (default: use GPU)"
    )

    # Texture file path (string)
    parser.add_argument("-output", "-o", type=str,
                        default='result.png',
                        help="Output file name. (default: result.png)")

    # time step size
    parser.add_argument("-step_size", "-s", type=float,
                        default=0.011,
                        help="time step size. (default: 0.01)")

    # accretion r1 r2
    parser.add_argument("-ar1", type=float,
                        default=2,
                        help="inner radius of accretion disk (default: 2)")
    # accretion r1 r2
    parser.add_argument("-ar2", type=float,
                        default=3.5,
                        help="outer radius of accretion disk (default: 6)")

    args = parser.parse_args()
    if args.cpu:
        ti.init(arch=ti.cpu)  # Use CPU for acceleration.
    else:
        ti.init(arch=ti.gpu)  # Use GPU for acceleration.

    if args.resolution == '4k':
        resol = np.array([3840, 2160])
    else:
        resol = np.array([1920, 1080])

    print('Welcome to Math/CS714 Project')

    # Ensure that position and look_at are float32
    my_camera = Camera(np.array(args.pov, dtype=np.float32), np.float32(args.focal),
                       np.array([0, 0, 0], dtype=np.float32), resol, fov=np.float32(args.fov % 180))
    print('Generating rays...')
    positions, directions = my_camera.get_all_rays()

    # Initialize the Scene
    scene = Scene(blackhole_r=ti.cast(1.0, ti.f32), accretion_r1=ti.cast(args.ar1, ti.f32),
                  accretion_r2=ti.cast(args.ar2, ti.f32), accretion_temp=ti.cast(400., ti.f32),
                  accretion_alpha=ti.cast(1, ti.f32),
                  skymap=Skymap(args.texture, r_max=10))
    scene.set_accretion_disk_texture(args.at)
    my_solver = Solver(scene, h=ti.cast(args.step_size, ti.f32))

    # Initialize Taichi fields
    image_width = my_camera._image_width
    image_height = my_camera._image_height

    colors = ti.Vector.field(3, dtype=ti.f32, shape=(image_width, image_height))
    colors.fill(0.0)

    print('Solving ODE...')
    if args.integrator == "euler":
        my_solver.solve_forward_euler(positions, directions, colors)
    elif args.integrator == 'rk4':
        my_solver.solve_rk4(positions, directions, colors)
    elif args.integrator == 'leapfrog':
        my_solver.solve_leapfrog(positions, directions, colors)
    elif args.integrator == 'ab2':
        my_solver.solve_ab2(positions, directions, colors)
    elif args.integrator == 'am4':
        my_solver.solve_am4(positions, directions, colors)

    # Rendering the image from the rays
    print('Rendering...')
    img = my_camera.render(colors)
    print('Image resolution: ', img.shape)
    if args.resolution == '4k':
        img_width, img_height = 3840, 2160  # 4K resolution
    else:  # 'fhd'
        img_width, img_height = 1920, 1080  # Full HD resolution

    # Plot and save the figure
    plt.figure(figsize=(img_width / 100, img_height / 100), dpi=100)
    plt.imshow(np.transpose(img, (1, 0, 2)))
    plt.axis('off')

    # Save the figure with the appropriate resolution
    plt.savefig(args.output, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close()




if __name__ == '__main__':
    main()
