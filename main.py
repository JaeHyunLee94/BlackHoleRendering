# main python script for running and solving Blackhole ODE

# Written by
#   - JaeHyun Lee: jaehyun.lee@wisc.edu
#   - Joon Suk Huh: jhuh23@wisc.edu
#   - Suenggwan Jo: sjo32@wisc.edu
#   - Hyeong Kyu Choi: hyeongkyu.choi@wisc.edu
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from camera import Camera
from solver import Solver
from skymap import Skymap

import taichi as ti

ti.init(arch=ti.gpu)  # Use GPU for acceleration; use ti.cpu if GPU is not available

if __name__ == '__main__':
    print('Hello CS714')

    # Ensure that position and look_at are float32
    my_camera = Camera(np.array([4, 4, 4], dtype=np.float32), np.float32(2.0),
                       np.array([0, 0, 0], dtype=np.float32), np.array([1920, 1080]))
    print('Generating rays...')
    positions_np, directions_np = my_camera.get_all_rays()

    image_path = 'texture/high_res/space_texture_high2.jpg'

    # Initialize the Skymap
    skymap = Skymap(image_path)

    my_solver = Solver(blackhole_radius = 1.0, skymap = skymap)

    # Initialize Taichi fields
    image_width = my_camera._image_width
    image_height = my_camera._image_height

    positions = ti.Vector.field(3, dtype=ti.f32, shape=(image_width, image_height))
    directions = ti.Vector.field(3, dtype=ti.f32, shape=(image_width, image_height))
    colors = ti.Vector.field(3, dtype=ti.f32, shape=(image_width, image_height))

    positions.from_numpy(positions_np.astype(np.float32))
    directions.from_numpy(directions_np.astype(np.float32))
    colors.fill(0.0)

    print('Solving ODE...')
    my_solver.solve_forward_euler(positions, directions, colors)

    # Rendering the image from the rays
    print('Rendering...')
    img = my_camera.render(colors)
    print('Image resolution: ', img.shape)

    plt.imshow(np.transpose(img, (1, 0, 2)))
    plt.axis('off')
    plt.savefig('result.png')
    plt.show()