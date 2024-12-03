import numpy as np
from ray import Ray

class Camera:
    """
    Simple Pinhole camera model for generating rays and rendering images.
    """
    def __init__(self, pos, focal_length, look_at, img_res, up=np.array([0, 1, 0], dtype=np.float32), fov=90):
        self._pos = pos.astype(np.float32)
        self._focal_length = np.float32(focal_length)
        self._look_at = look_at.astype(np.float32)

        self._up = up.astype(np.float32)
        self._forward = self._look_at - self._pos
        self._forward = self._forward / np.linalg.norm(self._forward)

        # Compute the right vector
        self._right = np.cross(self._forward, self._up)
        self._right = self._right / np.linalg.norm(self._right)

        # Recompute the true up vector to ensure orthogonality
        self._up = np.cross(self._right, self._forward)
        self._up = self._up / np.linalg.norm(self._up)

        self._fov = fov

        self._image_width = int(img_res[0])
        self._image_height = int(img_res[1])
        self._aspect_ratio = self._image_width / self._image_height

    def get_all_rays(self):
        """
        Generates rays from the camera position through each pixel in the image plane.
        Returns:
            positions_np: numpy array of shape (image_width, image_height, 3)
            directions_np: numpy array of shape (image_width, image_height, 3)
        """

        # Image plane dimensions
        image_plane_height = 2 * self._focal_length * np.tan(np.radians(self._fov / 2)).astype(np.float32)
        image_plane_width = image_plane_height * self._aspect_ratio

        # Pixel size in world units
        pixel_width = image_plane_width / self._image_width
        pixel_height = image_plane_height / self._image_height

        # Starting point (top-left corner) of the image plane in world coordinates
        image_plane_center = self._pos + self._forward * self._focal_length
        top_left = (image_plane_center
                    - (image_plane_width / 2) * self._right
                    + (image_plane_height / 2) * self._up)

        # Initialize arrays to store the positions and directions
        positions_np = np.zeros((self._image_width, self._image_height, 3), dtype=np.float32)
        directions_np = np.zeros((self._image_width, self._image_height, 3), dtype=np.float32)

        for i in range(self._image_width):
            for j in range(self._image_height):
                # Compute the position of the current pixel on the image plane
                pixel_pos = (top_left
                             + (i + 0.5) * pixel_width * self._right
                             - (j + 0.5) * pixel_height * self._up)
                # Direction from the camera position to the pixel position
                direction = pixel_pos - self._pos
                direction = direction / np.linalg.norm(direction)
                positions_np[i, j] = self._pos
                directions_np[i, j] = direction.astype(np.float32)

        return positions_np, directions_np

    def render(self, colors):
        """
        Render the scene by assigning colors based on ray directions.
        Args:
            colors: a Taichi field of shape (image_width, image_height, 3)
        Returns:
            image: a numpy array representing the rendered image
        """
        image_np = colors.to_numpy()
        image = np.clip(image_np, 0, 1)
        return image