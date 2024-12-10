# main.py

import numpy as np
import matplotlib.pyplot as plt

from camera import Camera
from solver import Solver
from skymap import Skymap
from scene import Scene

import taichi as ti
import os


def main():
    # Camera parameters
    pov = [8, 3, 1]
    focal = 1.5
    fov = 75
    sky_texture = 'texture/high_res/Marienplatz.jpg'
    accretion_texture = 'texture/high_res/Marienplatz.jpg'
    h = 0.01
    ar1 = 2
    ar2 = 6
    num_frames = 600
    ti.init(arch=ti.gpu)

    resol = np.array([3840, 2160])

    print('Welcome to Math/CS714 Project')

    # Orbit parameters
    cam_init_pos = np.array(pov, dtype=np.float32)
    radius = np.sqrt(cam_init_pos[0] ** 2 + cam_init_pos[1] ** 2)
    cam_z = cam_init_pos[2]
    look_at = np.array([0, 0, 0], dtype=np.float32)  # Assuming looking at the origin

    # Set up scene and solver once.
    scene = Scene(blackhole_r=ti.cast(1.0, ti.f32),
                  accretion_r1=ti.cast(ar1, ti.f32),
                  accretion_r2=ti.cast(ar2, ti.f32),
                  accretion_temp=ti.cast(400., ti.f32),
                  accretion_alpha=ti.cast(1, ti.f32),
                  skymap=Skymap(sky_texture, r_max=10))
    scene.set_accretion_disk_texture(accretion_texture)
    my_solver = Solver(scene, h=ti.cast(h, ti.f32))

    # Initialize the camera
    my_camera = Camera(
        pos=cam_init_pos,
        focal_length=focal,
        look_at=look_at,
        img_res=resol,
        fov=fov
    )

    # Ensure output directory exists
    output_dir = "frames"
    os.makedirs(output_dir, exist_ok=True)

    # Determine the angle step for each frame
    initial_angle = np.arctan2(cam_init_pos[1], cam_init_pos[0]) if radius > 0 else 0.0
    d_angle = 2.0 * np.pi / num_frames

    for frame_idx in range(num_frames):
        angle = initial_angle + frame_idx * d_angle

        # Update camera position
        cam_x = radius * np.cos(angle)
        cam_y = radius * np.sin(angle)
        new_pos = np.array([cam_x, cam_y, cam_z], dtype=np.float32)
        my_camera.pos[None] = new_pos

        # Update camera vectors based on the new position
        my_camera.update_camera()

        print(f'Generating rays for frame {frame_idx}...')
        my_camera.generate_rays()
        positions, directions = my_camera.positions, my_camera.directions

        # Initialize Taichi fields for colors (define once outside the loop)
        if frame_idx == 0:
            image_width = my_camera._image_width
            image_height = my_camera._image_height
            colors = ti.Vector.field(3, dtype=ti.f32, shape=(image_width, image_height))

        colors.fill(0.0)

        print(f'Solving ODE for frame {frame_idx}...')
        my_solver.solve_rk4(positions, directions, colors)

        print(f'Rendering frame {frame_idx}...')
        img = my_camera.render(colors)
        print('Image resolution: ', img.shape)

        img_width, img_height = 3840, 2160

        # Plot and save the figure
        plt.figure(figsize=(img_width / 100, img_height / 100), dpi=100)
        plt.imshow(np.transpose(img, (1, 0, 2)))
        plt.axis('off')

        frame_filename = f"{output_dir}/frame_{frame_idx:03d}.png"
        # Save the figure with the appropriate resolution
        plt.savefig(frame_filename, dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f'Frame {frame_idx} saved as {frame_filename}')

    print("All frames rendered. Use an external tool to compile images into a video.")


if __name__ == '__main__':
    main()