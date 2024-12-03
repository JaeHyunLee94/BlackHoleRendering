import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skymap import Skymap

# Replace 'path_to_your_image.jpg' with the actual image file path
image_path = 'texture/space_texture1.jpg'

# Initialize the Skymap
skymap = Skymap(image_path)

# Map the texture to the sphere
x, y, z, colors = skymap.map_texture_to_sphere(num_latitudes=200, num_longitudes=200)

# Visualize the textured sphere
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(
    x, y, z, facecolors=colors/255.0, rstride=1, cstride=1
)
ax.set_axis_off()
plt.show()
