import numpy as np
from PIL import Image
import taichi as ti

@ti.data_oriented
class Skymap:
    def __init__(self, image_path, r_max):
        """
        Initializes the Skymap with the given image.

        Parameters:
        - image_path: str, path to the .jpg or .png image file.
        """
        self.image_path = image_path
        self.texture = self.load_texture(image_path)
        self.img_height, self.img_width, _ = self.texture.shape
        self.texture_field = ti.Vector.field(3, dtype=ti.f32, shape=(self.img_height, self.img_width))
        self.texture_field.from_numpy(self.texture)
        self.r_max = r_max

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
        print(f"Loaded texture with shape: {texture.shape}")
        return texture

    @ti.func
    def get_color_from_ray_ti(self, D):
        D = D.normalized()
        x, y, z = D[0], D[1], D[2]

        # Compute spherical coordinates
        theta = ti.acos(z)
        phi = ti.atan2(y, x)
        if phi < 0:
            phi += 2 * ti.math.pi  # Use Taichi's pi

        # Compute texture coordinates (u, v)
        u = phi / (2 * ti.math.pi)
        v = theta / ti.math.pi

        # Map (u, v) to texture pixel coordinates
        tex_u = ti.cast(u * (self.img_width - 1), ti.i32)
        # tex_v = ti.cast((1 - v) * (self.img_height - 1), ti.i32)  # Flip v to match image coordinate system
        tex_v = ti.cast(v * (self.img_height - 1), ti.i32) # Test

        # Ensure indices are within bounds
        tex_u = ti.min(ti.max(tex_u, 0), self.img_width - 1)
        tex_v = ti.min(ti.max(tex_v, 0), self.img_height - 1)
        return self.texture_field[tex_v, tex_u]