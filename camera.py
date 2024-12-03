import numpy as np
from ray import Ray


class Camera:
    """
    Simple Pinhole camera model for generating rays and rendering images.
    """
    def __init__(self, pos, focal_length, look_at, img_res, up=np.array([0, 1, 0]), fov=90):
        self._pos = pos
        self._focal_length = focal_length
        self._look_at = look_at

        self._up = up
        self._forward = self._look_at - self._pos
        self._forward = self._forward / np.linalg.norm(self._forward)

        # Compute the right vector
        self._right = np.cross(self._forward, self._up)
        self._right = self._right / np.linalg.norm(self._right)

        # Recompute the true up vector to ensure orthogonality
        self._up = np.cross(self._right, self._forward)
        self._up = self._up / np.linalg.norm(self._up)

        self._fov = fov

        self._image_width = img_res[0]
        self._image_height = img_res[1]
        self._image = np.zeros((img_res[0], img_res[1], 3))
        self._aspect_ratio = self._image_width / self._image_height

    # def set_pos(self, pos):
    #     self._pos = pos

    # def set_image_res(self, img_res):
    #     self._image = np.zeros(img_res)
    #
    # def set_focal_length(self, focal_length):
    #     self._focal_length = focal_length
    #
    # def get_image_width(self):
    #     return self._image.shape[0]
    #
    # def get_image_height(self):
    #     return self._image.shape[1]

    def get_all_rays(self):
        """
        Generates rays from the camera position through each pixel in the image plane.
        Returns:
            rays: a 2D array (height x width) of Ray objects
        """

        # Image plane dimensions (assuming a field of view of 90 degrees)
        image_plane_height = 2 * self._focal_length * np.tan(np.radians(self._fov / 2))
        image_plane_width = image_plane_height * self._aspect_ratio

        # Pixel size in world units
        pixel_width = image_plane_width / self._image_width
        pixel_height = image_plane_height / self._image_height

        # Starting point (top-left corner) of the image plane in world coordinates
        image_plane_center = self._pos + self._forward * self._focal_length
        top_left = (image_plane_center
                    - (image_plane_width / 2) * self._right
                    + (image_plane_height / 2) * self._up)

        # Initialize a 2D array to store the rays
        rays = np.empty((self._image_width, self._image_height), dtype=object)

        for i in range(self._image_width):
            for j in range(self._image_height):
                # Compute the position of the current pixel on the image plane
                pixel_pos = (top_left
                             + (i + 0.5) * pixel_width * self._right
                             - (j + 0.5) * pixel_height * self._up)
                # Direction from the camera position to the pixel position
                direction = pixel_pos - self._pos
                direction = direction / np.linalg.norm(direction)
                rays[i, j] = Ray(self._pos, direction)

        return rays

    def render(self, rays):
        """
        Render the scene by assigning colors based on ray directions.
        Args:
            rays: a 2D array of Ray objects
        Returns:
            image: a numpy array representing the rendered image
        """

        for i in range(self._image_width):
            for j in range(self._image_height):
                self._image[i, j] = rays[i, j].color/255.0

        # Ensure pixel values are within [0, 1]
        image = np.clip(self._image, 0, 1)
        return image
