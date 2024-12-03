# main python script for running and solving Blackhole ODE

# Written by
#   - JaeHyun Lee: jaehyun.lee@wisc.edu
#   - Joon Suk Huh: jhuh23@wisc.edu
#   - Suenggwan Jo: sjo32@wisc.edu
#   - Hyeong Kyu Choi: hyeongkyu.choi@wisc.edu
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from blackhole import BlackHole
from camera import Camera
from solver import Solver
from skymap import Skymap

if __name__ == '__main__':
    print('Hello CS714')

    # Rendering Three black hole radius is fixed to 1
    Schwarzschild_blackhole = BlackHole()

    my_camera = Camera(np.array([5, 5, 5]), 1, np.array([0, 0, 0]), np.array([640, 480]))
    print('Generating rays...')
    my_rays = my_camera.get_all_rays()

    image_path = 'texture/high_res/space_texture_high2.jpg'

    # Initialize the Skymap
    skymap = Skymap(image_path)

    my_solver = Solver(Schwarzschild_blackhole, skymap)

    rays_flat = my_rays.flatten()
    for ray in tqdm(rays_flat, desc='Solving ODE', total=my_rays.size):
        # Solving ODE for each ray
        my_solver.solve(ray)

    # Rendering the image from the rays
    print('Rendering...')
    img = my_camera.render(my_rays)
    print('Image resolution: ', img.shape)

    plt.imshow(np.transpose(img, (1, 0, 2)))
    plt.axis('off')
    plt.savefig('result.png')
    plt.show()
