class BlackHole:
    def __init__(self, pos, radius):
        self._pos = pos
        self._radius = radius
        pass

    def get_radius(self):
        return self._radius

    def get_pos(self):
        return self._pos
