import numpy as np


class BlackHole:
    """
    Just a simple sphere for now.
    """

    def __init__(self, pos=np.array([0, 0, 0]), radius=1):
        self.pos = pos
        self.radius = radius
