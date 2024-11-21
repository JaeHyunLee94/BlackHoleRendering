class Camera:
    def __init__(self, pos, look_at):
        self._pos = pos
        self._look_at = look_at

    def set_pos(self, pos):
        self._pos = pos

    def set_image_plane(self, width, height):
        pass

    def render(self):
        pass
