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

if __name__ == '__main__':
    print('Hello CS714')

    # Rendering Three black holes (For now these are just spheres)
    bh1 = BlackHole(np.array([0, 2, 3]), 1.4)
    bh2 = BlackHole(np.array([1, 1, 1]), 1)
    bh3 = BlackHole(np.array([4, 0, 1]), 2)
    scene = [bh1, bh2, bh3]

    my_camera = Camera(np.array([5, 5, 5]), 1, np.array([0, 0, 0]), np.array([720, 480]))
    print('Generating rays...')
    my_rays = my_camera.get_all_rays()

    my_solver = Solver(scene)

    rays_flat = my_rays.flatten()
    for ray in tqdm(rays_flat, desc='Solving ODE', total=my_rays.size):
        # Solving ODE for each ray
        my_solver.solve(ray)

    # Rendering the image from the rays
    print('Rendering...')
    img = my_camera.render(my_rays)
    print('Image resolution: ', img.shape)

    plt.imshow(img)
    plt.axis('off')
    plt.show()
