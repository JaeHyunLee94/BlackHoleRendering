# main python script for running and solving Blackhole ODE
# Written by
#   - JaeHyun Lee: jaehyun.lee@wisc.edu
#   - Joon Suk Huh: jhuh23@wisc.edu
#   - Suenggwan Jo: sjo32@wisc.edu
#   - Hyeong Kyu Choi: hyeongkyu.choi@wisc.edu
import numpy as np
import matplotlib.pyplot as plt

from blackhole import BlackHole
from camera import Camera
from solver import Solver

if __name__ == '__main__':
    print('Hello CS714')

    # Rendering Three black holes
    bh1 = BlackHole(np.array([0, 2, 3]), 1.4)
    bh2 = BlackHole(np.array([1, 1, 1]), 1)
    bh3 = BlackHole(np.array([4, 0, 1]), 2)
    scene = [bh1, bh2, bh3]

    cmr = Camera(np.array([5, 5, 5]), 1, np.array([0, 0, 0]), np.array([720, 480]))
    print('Generating rays...')
    rays = cmr.get_all_rays()

    print('Solving ODE...')
    my_solver = Solver(scene)
    for i in range(len(rays)):
        for j in range(len(rays[i])):
            my_solver.solve(rays[i, j])

    print('Rendering...')
    img = cmr.render(rays)
    print('image res: ', img.shape)

    plt.imshow(img)
    plt.axis('off')  # Hide the axis
    plt.show()



