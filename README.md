# BlackHole rendering

<img src="https://github.com/user-attachments/assets/514cad6f-638f-497b-8c4f-c2053e914103" alt="gallery_teaser" width="700" />

This project implements a ray-tracing-based rendering system for visualizing a Schwarzschild black hole with customizable settings. The rendering process is grounded in general relativity, allowing for realistic depictions of light bending and photon rings caused by the black hole's intense gravitational field. The implementation supports various numerical integrators and configurations to tailor the rendering experience.

## Features
- **Customizable Camera**: Adjust camera position, focal length, and field of view (FoV).
- **Resolution Options**: Choose between 4K (`3840x2160`) or FHD (`1920x1080`).
- **Custom Textures**: Apply textures for the Sky Box and accretion disk.
- **Numerical Integrators**: Select from the following integrators:
  - Forward Euler (`euler`)
  - Fourth-Order Runge-Kutta (`rk4`)
  - Leapfrog (`leapfrog`)
  - Adams-Bashforth 2-step (`ab2`)
  - Adams-Moulton 4-step (`am4`)
- **Device Support**: Use GPU for faster rendering (default) or CPU via the `--cpu` flag.
- **Output Customization**: Save rendered images with a specified filename.

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/JaeHyunLee94/BlackHoleRendering.git
   cd BlackHoleRendering
   ```

2. Install required dependencies:
    ```
    pip install -r requirements.txt
    ```

## Example Usage
Render a 4K black hole image using the RK4 integrator:

    python main.py -integrator rk4 -resolution 4k -output blackhole_rk4.png
    
Render an image with custom camera settings and accretion disk parameters:

    python main.py -pov 0 5 2 -focal 2.0 -ar1 3 -ar2 8 -integrator am4 -output custom_blackhole.png

Generate image sequence:

    python export_animation.py



## Arguments

| Argument      | Description                                                                                                  | Default                                 |
|---------------|--------------------------------------------------------------------------------------------------------------|-----------------------------------------|
| -pov or -p    | Camera position in Cartesian coordinates (x, y, z). Controls where the camera is placed.                     | [6, 0, 0.5]                          |
| -focal or -f  | Focal length of the camera. Determines how "zoomed in" the image appears.                                    | 1.8                                     |
| -fov          | Field of View in degrees (0-180). Wider FoV values result in more of the scene being captured.               | 60                                      |
| -resolution or -r | Resolution of the rendered image. Options are 4k (3840x2160) or fhd (1920x1080).                       | 4k                                      |
| -texture or -t | Path to the Sky Box texture file. Specifies the background texture for the visualization.                   | texture/high_res/space_texture_high1.jpg |
| -at           | Path to the accretion disk texture file. Specifies the visual texture for the black hole’s accretion disk.   | texture/ad/adisk.jpg                    |
| -integrator or -i | Numerical integrator to use for solving light trajectories. Options: euler, rk4, leapfrog, ab2, am4.    | euler                                   |
| --cpu         | Flag to use the CPU for rendering instead of the GPU. This may increase rendering time.                      | Disabled (GPU used by default)          |
| -output or -o | Name of the output file for the rendered image.                                                              | result.png                              |
| -lamb         | Time step size for integration. Smaller step sizes result in higher accuracy but slower computation.         | 0.01                                     |
| -ar1          | Inner radius of the accretion disk. Determines how close the accretion disk starts relative to the black hole.| 2                                       |
| -ar2          | Outer radius of the accretion disk. Determines how far the accretion disk extends outward.                   | 3                                       |

## Gallery

<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/a3063396-3064-4edb-b363-0962e8878b1d" alt="gallery_1" width="400"></td>
    <td><img src="https://github.com/user-attachments/assets/91e26930-475e-4925-af3f-e05f9430e3f8" alt="gallery_2" width="400"></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/c554a73e-2294-44fa-bec8-d25ffe0dc08f" alt="gallery_starry" width="400"></td>
    <td><img src="https://github.com/user-attachments/assets/44994327-6654-4f37-9142-934404f5d208" alt="gallery_3" width="400"></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/d35795cf-b81e-447b-8a5e-9c8944487535" alt="gallery_4" width="400"></td>
    <td><img src="https://github.com/user-attachments/assets/e4307b0c-ac5f-4e3b-a215-8586c8ce49a3" alt="gallery_5" width="400"></td>
  </tr>
</table>

## References
### Blogs
1. [Visualizing Black Holes with General Relativistic Ray Tracing](https://blog.seanholloway.com/2022/03/13/visualizing-black-holes-with-general-relativistic-ray-tracing/)
2. [Ray Tracing Black Holes](https://eliot1019.github.io/Black-Hole-Raytracer/)
3. [How to draw a Black Hole](https://rantonels.github.io/starless/)
### Codes
1. [Black Hole visualization with C#](https://github.com/HollowaySean/BlackHoleViz_v2)
2. [black-hole-raytracing with python](https://github.com/bguesman/black-hole-raytracing)
3. [Curved Spacetime Raytracer](https://github.com/silvaan/blackhole_raytracer)
4. [Blackstar](https://github.com/flannelhead/blackstar)
5. [BlackHoleRaytracer](https://github.com/dbrant/BlackHoleRaytracer)