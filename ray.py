import numpy as np


class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction
        self.color = np.array([255, 255, 255])

    def at(self, t):
        return self.origin + t * self.direction
