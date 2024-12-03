import numpy as np
from PIL import Image


class Skymap:
    def __init__(self, image_path, r=1000):
        """
        Initializes the Skymap with the given image.

        Parameters:
        - image_path: str, path to the .jpg or .png image file.
        """
        self.image_path = image_path
        self.texture = self.load_texture(image_path)

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
        texture = np.array(image)
        print(f"Loaded texture with shape: {texture.shape}")
        return texture

    def create_sphere(self, num_latitudes, num_longitudes):
        """
        Creates a sphere mesh grid.

        Parameters:
        - num_latitudes: int, number of latitude divisions.
        - num_longitudes: int, number of longitude divisions.

        Returns:
        - x, y, z: numpy.ndarray, coordinates of the sphere's surface.
        """
        theta = np.linspace(0, np.pi, num_latitudes)
        phi = np.linspace(0, 2 * np.pi, num_longitudes)
        theta, phi = np.meshgrid(theta, phi)
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        return x, y, z

    def get_texture_coordinates(self, num_latitudes, num_longitudes):
        """
        Computes texture coordinates for mapping.

        Parameters:
        - num_latitudes: int, number of latitude divisions.
        - num_longitudes: int, number of longitude divisions.

        Returns:
        - u, v: numpy.ndarray, texture coordinates.
        """
        theta = np.linspace(0, np.pi, num_latitudes)
        phi = np.linspace(0, 2 * np.pi, num_longitudes)
        theta, phi = np.meshgrid(theta, phi)
        u = phi / (2 * np.pi)
        v = theta / np.pi
        return u, v

    def map_texture_to_sphere(self, num_latitudes=100, num_longitudes=100):
        """
        Maps the texture onto the sphere.

        Parameters:
        - num_latitudes: int, optional, default=100.
        - num_longitudes: int, optional, default=100.

        Returns:
        - x, y, z: numpy.ndarray, coordinates of the sphere's surface.
        - colors: numpy.ndarray, color data from the texture.
        """
        # Get sphere coordinates
        x, y, z = self.create_sphere(num_latitudes, num_longitudes)
        # Get texture coordinates
        u, v = self.get_texture_coordinates(num_latitudes, num_longitudes)
        # Map u, v to image pixel coordinates
        img_height, img_width, _ = self.texture.shape
        tex_u = (u * (img_width - 1)).astype(int)
        tex_v = ((1 - v) * (img_height - 1)).astype(int)  # Flip v to match image coordinate system
        # Ensure indices are within bounds
        tex_u = np.clip(tex_u, 0, img_width - 1)
        tex_v = np.clip(tex_v, 0, img_height - 1)
        # Get texture colors
        colors = self.texture[tex_v, tex_u]
        return x, y, z, colors

    def get_color_from_ray(self, ray):
        # Normalize the ray direction
        D = ray.direction / np.linalg.norm(ray.direction)
        x, y, z = D

        # Compute spherical coordinates
        theta = np.arccos(z)
        phi = np.arctan2(y, x)
        if phi < 0:
            phi += 2 * np.pi  # Ensure phi is in [0, 2π]

        # Compute texture coordinates (u, v)
        u = phi / (2 * np.pi)
        v = theta / np.pi

        # Map (u, v) to texture pixel coordinates
        img_height, img_width, _ = self.texture.shape
        tex_u = int(u * (img_width - 1))
        tex_v = int((1 - v) * (img_height - 1))  # Flip v to match image coordinate system

        # Ensure indices are within bounds
        tex_u = np.clip(tex_u, 0, img_width - 1)
        tex_v = np.clip(tex_v, 0, img_height - 1)
        return self.texture[tex_v, tex_u]
