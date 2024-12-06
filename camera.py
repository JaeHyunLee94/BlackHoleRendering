import numpy as np
import taichi as ti


@ti.data_oriented
class Camera:
    def __init__(self, pos, focal_length, look_at, img_res, up=np.array([0, 0, 1], dtype=np.float32), fov=90):
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

        # Allocate Taichi fields for positions, directions, and rendered image
        self.positions = ti.Vector.field(3, dtype=ti.f32, shape=(self._image_width, self._image_height))
        self.directions = ti.Vector.field(3, dtype=ti.f32, shape=(self._image_width, self._image_height))
        self.image = ti.Vector.field(3, dtype=ti.f32, shape=(self._image_width, self._image_height))  # RGB image

    @ti.kernel
    def generate_rays(self):
        # Image plane dimensions
        fov_radians = (self._fov / 2.0) * (3.141592653589793 / 180.0)  # Convert degrees to radians
        image_plane_height = 2.0 * self._focal_length * ti.tan(fov_radians)
        image_plane_width = image_plane_height * self._aspect_ratio

        # Pixel size in world units
        pixel_width = image_plane_width / self._image_width
        pixel_height = image_plane_height / self._image_height

        # Starting point (top-left corner) of the image plane in world coordinates
        image_plane_center = ti.Vector(self._pos) + ti.Vector(self._forward) * self._focal_length
        top_left = (
                image_plane_center
                - (image_plane_width / 2.0) * ti.Vector(self._right)
                + (image_plane_height / 2.0) * ti.Vector(self._up)
        )

        for i, j in ti.ndrange(self._image_width, self._image_height):
            # Compute the position of the current pixel on the image plane
            pixel_pos = (
                    top_left
                    + (i + 0.5) * pixel_width * ti.Vector(self._right)
                    - (j + 0.5) * pixel_height * ti.Vector(self._up)
            )
            # Direction from the camera position to the pixel position
            direction = pixel_pos - ti.Vector(self._pos)
            direction = direction.normalized()

            # Assign values to Taichi fields
            self.positions[i, j] = ti.Vector(self._pos)
            self.directions[i, j] = direction

    @ti.kernel
    def generate_rays_perpendicular(self):
        # Image plane dimensions based on the field of view
        fov_radians = (self._fov / 2.0) * (3.141592653589793 / 180.0)  # Convert degrees to radians
        image_plane_height = 2.0 * self._focal_length * ti.tan(fov_radians)
        image_plane_width = image_plane_height * self._aspect_ratio

        # Pixel size in world units
        pixel_width = image_plane_width / self._image_width
        pixel_height = image_plane_height / self._image_height

        # Compute the center and top-left corner of the image plane
        image_plane_center = ti.Vector(self._pos) + ti.Vector(self._forward) * self._focal_length
        top_left = (
                image_plane_center
                - (image_plane_width / 2.0) * ti.Vector(self._right)
                + (image_plane_height / 2.0) * ti.Vector(self._up)
        )

        # All rays will share the same direction, which is the forward direction.
        ray_direction = ti.Vector(self._forward).normalized()

        for i, j in ti.ndrange(self._image_width, self._image_height):
            # Compute the position of the current pixel on the image plane
            pixel_pos = (
                    top_left
                    + (i + 0.5) * pixel_width * ti.Vector(self._right)
                    - (j + 0.5) * pixel_height * ti.Vector(self._up)
            )

            # For an orthographic (parallel) projection, the ray's origin is at pixel_pos,
            # and the direction is simply the camera's forward direction.
            self.positions[i, j] = pixel_pos
            self.directions[i, j] = ray_direction

    @ti.kernel
    def render_scene(self, colors: ti.template()):
        # Assign the colors from the Taichi field `colors` to the image
        for i, j in self.image:
            self.image[i, j] = colors[i, j]

    def get_all_rays(self):
        # Call the Taichi kernel to generate rays
        self.generate_rays()
        # self.generate_rays_perpendicular()
        return self.positions, self.directions

    def render(self, colors):
        """
        Render the scene by assigning colors based on ray directions.

        Args:
            colors: a Taichi field of shape (image_width, image_height, 3) containing RGB values.

        Returns:
            image: a numpy array representing the rendered image.
        """
        # Call the Taichi kernel to render the scene
        self.render_scene(colors)

        # Convert the Taichi field to a NumPy array
        image_np = self.image.to_numpy()
        return np.clip(image_np, 0, 1)  # Ensure values are in the range [0, 1]
