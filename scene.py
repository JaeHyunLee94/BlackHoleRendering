import numpy as np
import taichi as ti

from skymap import Skymap


@ti.data_oriented
class Scene:
    def __init__(self, blackhole_r: ti.f32, accretion_r1: ti.f32,
                 accretion_r2: ti.f32, accretion_temp: ti.f32, skymap: Skymap):
        self.blackhole_r = blackhole_r
        self.accretion_r1 = accretion_r1
        self.accretion_r2 = accretion_r2
        self.accretion_temp = accretion_temp
        self.skymap = skymap
