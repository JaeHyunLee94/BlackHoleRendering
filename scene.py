import numpy as np
import taichi as ti

from PIL import Image
from skymap import Skymap


@ti.data_oriented
class Scene:
    def __init__(self, blackhole_r: ti.f32, accretion_r1: ti.f32,
                 accretion_r2: ti.f32, accretion_temp: ti.f32, accretion_alpha: ti.f32, skymap: Skymap):
        self.blackhole_r = blackhole_r
        self.accretion_r1 = accretion_r1
        self.accretion_r2 = accretion_r2
        self.accretion_temp = accretion_temp
        self.accretion_alpha = accretion_alpha
        self.skymap = skymap
        self.has_accretion_disk_texture = False
        self.accretion_image = None
        self.img_height = None
        self.img_width = None
        self.texture_field = None

    def set_accretion_disk_texture(self, image_path):
        self.has_accretion_disk_texture = True
        self.accretion_image = self.load_texture(image_path)
        self.img_height, self.img_width, _ = self.accretion_image.shape
        self.texture_field = ti.Vector.field(3, dtype=ti.f32, shape=(self.img_height, self.img_width))
        self.texture_field.from_numpy(self.accretion_image)

    def load_texture(self, image_path):
        """
        Loads the image and converts it into a numpy array.

        Parameters:
        - image_path: str, path to the image file.

        Returns:
        - texture: numpy.ndarray, the image as a numpy array.
        """
        image = Image.open(image_path)
        image = image.convert('RGB')  # Ensure the image is in RGB format
        # Normalize and convert to float32
        texture = (np.array(image).astype(np.float32)) / 255.0
        print(f"Loaded accretion disk texture with shape: {texture.shape}")
        return texture

    @ti.func
    def get_accretion_disk_color_ti(self, x, y):

        # Compute squared distance from the origin in the x-y plane
        r_squared = x ** 2 + y ** 2
        r = ti.sqrt(r_squared)
        color = ti.Vector([0.0, 0.0, 0.0])

        # Check if the ray falls within the specified disk
        if self.accretion_r1 < r < self.accretion_r2:
            # Compute spherical coordinates

            phi = ti.atan2(y, x)
            if phi < 0:
                phi += 2 * ti.math.pi  # Use Taichi's pi

            # Compute texture coordinates (u, v)
            u = phi / (2 * ti.math.pi)
            v = (r - self.accretion_r1) / (self.accretion_r2 - self.accretion_r1)

            # Map (u, v) to texture pixel coordinates
            tex_u = ti.cast(u * (self.img_width - 1), ti.i32)
            tex_v = ti.cast(v * (self.img_height - 1), ti.i32)

            # Ensure indices are within bounds
            tex_u = ti.min(ti.max(tex_u, 0), self.img_width - 1)
            tex_v = ti.min(ti.max(tex_v, 0), self.img_height - 1)

            # Return color from the texture
            color = self.texture_field[tex_v, tex_u]
        else:
            # Return a default or background color if outside the disk
            print("Outside the accretion disk")
            color = ti.Vector([0.0, 0.0, 0.0])  # Black or any desired background color
        return color
