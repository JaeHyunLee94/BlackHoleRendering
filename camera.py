import numpy as np
import taichi as ti

@ti.data_oriented
class Camera:
    def __init__(self, pos, focal_length, look_at, img_res, up=np.array([0, 0, 1], dtype=np.float32), fov=90):
        # Initialize camera parameters
        pos = pos.astype(np.float32)
        look_at = look_at.astype(np.float32)
        up = up.astype(np.float32)

        self._image_width = int(img_res[0])
        self._image_height = int(img_res[1])
        self._aspect_ratio = self._image_width / self._image_height

        # Define Taichi fields for camera parameters
        self.pos = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.look_at = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.up = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.forward = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.right = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.focal_length = ti.field(dtype=ti.f32, shape=())
        self.fov = ti.field(dtype=ti.f32, shape=())
        self.aspect_ratio = ti.field(dtype=ti.f32, shape=())

        # Assign initial values to Taichi fields
        self.pos[None] = pos
        self.look_at[None] = look_at
        self.up[None] = up
        self.focal_length[None] = np.float32(focal_length)
        self.fov[None] = np.float32(fov)
        self.aspect_ratio[None] = self._image_width / self._image_height

        # Allocate Taichi fields for positions, directions, and rendered image
        self.positions = ti.Vector.field(3, dtype=ti.f32, shape=(self._image_width, self._image_height))
        self.directions = ti.Vector.field(3, dtype=ti.f32, shape=(self._image_width, self._image_height))
        self.image = ti.Vector.field(3, dtype=ti.f32, shape=(self._image_width, self._image_height))  # RGB image

        # Initialize forward, right vectors
        self.update_camera_vectors()

    @ti.kernel
    def update_camera_vectors(self):
        # Compute forward, right, up vectors based on pos and look_at
        direction = (self.look_at[None] - self.pos[None]).normalized()
        self.forward[None] = direction
        self.right[None] = ti.math.cross(direction, self.up[None]).normalized()
        self.up[None] = ti.math.cross(self.right[None], self.forward[None]).normalized()

    @ti.kernel
    def generate_rays(self):
        # Image plane dimensions
        fov_radians = (self.fov[None] / 2.0) * (3.141592653589793 / 180.0)  # Convert degrees to radians
        image_plane_height = 2.0 * self.focal_length[None] * ti.tan(fov_radians)
        image_plane_width = image_plane_height * self.aspect_ratio[None]

        # Pixel size in world units
        pixel_width = image_plane_width / self._image_width
        pixel_height = image_plane_height / self._image_height

        # Starting point (top-left corner) of the image plane in world coordinates
        image_plane_center = self.pos[None] + self.forward[None] * self.focal_length[None]
        top_left = (
                image_plane_center
                - (image_plane_width / 2.0) * self.right[None]
                + (image_plane_height / 2.0) * self.up[None]
        )

        for i, j in ti.ndrange(self._image_width, self._image_height):
            # Compute the position of the current pixel on the image plane
            pixel_pos = (
                    top_left
                    + (i + 0.5) * pixel_width * self.right[None]
                    - (j + 0.5) * pixel_height * self.up[None]
            )
            # Direction from the camera position to the pixel position
            direction = (pixel_pos - self.pos[None]).normalized()

            # Assign values to Taichi fields
            self.positions[i, j] = self.pos[None]
            self.directions[i, j] = direction

    @ti.kernel
    def generate_rays_perpendicular(self):
        # Image plane dimensions based on the field of view
        fov_radians = (self.fov[None] / 2.0) * (3.141592653589793 / 180.0)  # Convert degrees to radians
        image_plane_height = 2.0 * self.focal_length[None] * ti.tan(fov_radians)
        image_plane_width = image_plane_height * self.aspect_ratio[None]

        # Pixel size in world units
        pixel_width = image_plane_width / self._image_width
        pixel_height = image_plane_height / self._image_height

        # Compute the center and top-left corner of the image plane
        image_plane_center = self.pos[None] + self.forward[None] * self.focal_length[None]
        top_left = (
                image_plane_center
                - (image_plane_width / 2.0) * self.right[None]
                + (image_plane_height / 2.0) * self.up[None]
        )

        # All rays will share the same direction, which is the forward direction.
        ray_direction = self.forward[None].normalized()

        for i, j in ti.ndrange(self._image_width, self._image_height):
            # Compute the position of the current pixel on the image plane
            pixel_pos = (
                    top_left
                    + (i + 0.5) * pixel_width * self.right[None]
                    - (j + 0.5) * pixel_height * self.up[None]
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
        # Render the scene by assigning colors
        self.render_scene(colors)
        image_np = self.image.to_numpy()
        return np.clip(image_np, 0, 1)

    def update_camera(self):
        # Update the camera's orientation vectors after position or look_at changes
        self.update_camera_vectors()