import numpy as np
class Ray:
    def __init__(self, origin, direction):
        self.position = origin  # current ray position
        self.direction = direction  # current direction

        self.color = np.array([0, 0, 0])  # initialize color to black
        # current position in spherical coordinates
        # self.r, self.theta, self.phi = self.cartesian_to_spherical(self.position)

    def at(self, t):
        return self.position + t * self.direction