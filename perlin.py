import math
import argparse

import imageio
import numpy as np
from noise import snoise2, snoise3, snoise4
from pygifsicle import optimize
import matplotlib.pyplot as plt


def _simplex_noise3d(shape, scale, octaves, random):
    img = np.zeros(shape)

    if random:
        offset = np.random.rand() * 100
    else:
        offset = 0
    
    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                img[x, y, z] = snoise3(x * scale[0] + offset, y * scale[1] + offset, z * scale[2], octaves=octaves)
    
    img = ((img - img.min()) * (1 / (img.max() - img.min()) * 255)).astype('uint8')
    
    return img


def _simplex_noise4d(shape, scale, octaves, radius, random):
    img = np.zeros(shape)

    if random:
        offset = np.random.rand() * 100
    else:
        offset = 0

    for i in range(shape[2]):
        cos_value = radius * math.cos(2 * math.pi * (i / shape[2]))
        sin_value = radius * math.sin(2 * math.pi * (i / shape[2]))
        for x in range(shape[0]):
            for y in range(shape[1]):
                img[x, y, i] = snoise4(x * scale[0] + offset, y * scale[1] + offset, cos_value, sin_value)
    
    img = ((img - img.min()) * (1 / (img.max() - img.min()) * 255)).astype('uint8')
    
    return img


class PerlinGif():
    def __init__(self, **kwargs):
        self.noise_dim = kwargs["d"]
        self.n = kwargs["n"]
        self.fps = kwargs["fps"]
        self.frames = kwargs["frames"]
        self.scale = kwargs["s"]
        self.octaves = kwargs["o"]
        self.radius = kwargs["r"]
        self.compress = kwargs["compress"]
        self.output_file = kwargs["out"]
        self.pipeline = kwargs["pipeline"]
        self.random = kwargs["R"]
        self.shape = (*self.n, self.frames)
    
    def _to_gif(self):
        kwargs = {'duration': 1 / self.fps}
        imageio.mimsave(self.output_file, self.images, **kwargs)
        if self.compress:
            optimize(self.output_file)

    def _make_3d_gif(self):
        images = _simplex_noise3d(self.shape, self.scale, self.octaves, self.random)
        images = images.transpose(2, 0, 1)

        return images
        
    def _make_4d_gif(self):
        images = _simplex_noise4d(self.shape, self.scale, self.octaves, self.radius, self.random)
        images = images.transpose(2, 0, 1)
    
        return images

    def render(self):
        if self.noise_dim == 3:
            self.images = self._make_3d_gif()
        else:
            self.images = self._make_4d_gif()
        
        if not self.pipeline.is_empty():
            print("Running pipeline")
            self.images = self.pipeline.run(self.images)
        
        self._to_gif()


class PerlinFlowField():
    def __init__(self, **kwargs):
        self.noise_dim = kwargs["d"]
        self.n = kwargs["n"]
        self.fps = kwargs["fps"]
        self.frames = kwargs["frames"]
        self.scale = kwargs["s"]
        self.octaves = kwargs["o"]
        self.radius = kwargs["r"]
        self.compress = kwargs["compress"]
        self.output_file = kwargs["out"]
        self.pipeline = kwargs["pipeline"]
        self.random = kwargs["R"]
        self.shape = (*self.n, self.frames)

    def flow_field(self):
        # generate vector field
        flow_field = np.zeros((*shape, 2)) # if 256, 256, 30 -> 256, 256, 30, 2
        radius = 0.1
        scale = (0.1, 0.1)

        for i in range(shape[2]):
            cos_value = self.radius * math.cos(2 * math.pi * (i / shape[2]))
            sin_value = self.radius * math.sin(2 * math.pi * (i / shape[2]))
            for x in range(shape[0]):
                for y in range(shape[1]):
                    angle = snoise4(x * self.scale[0], y * self.scale[1], cos_value, sin_value)
                    flow_field[x, y, i, 0] = np.cos(angle) * 10
                    flow_field[x, y, i, 1] = np.sin(angle) * 10
       
        # x, y = np.meshgrid(np.arange(0, shape[0]*scale[0], scale[0]), np.arange(0, shape[1]*scale[1], scale[1]))
        # u, v = flow_field[:, :, 0, 0], flow_field[:, :, 0, 1]

        # figs, axs = plt.subplots(2, 2)
        # axs[0][0].quiver(x, y, flow_field[:, :, 0, 0] * -1, flow_field[:, :, 0, 1] * -1)
        # axs[0][1].quiver(x, y, flow_field[:, :, 1, 0], flow_field[:, :, 1, 1])
        # axs[1][0].quiver(x, y, flow_field[:, :, 2, 0], flow_field[:, :, 2, 1])
        # axs[1][1].quiver(x, y, flow_field[:, :, 3, 0], flow_field[:, :, 3, 1])
        # plt.show()

    def add_particles(self):
        # add particles at random location
        pass

    def update_positions(self):
        # for each particle, find associated vector, apply force
        # Make sure they wrap around the frame
        pass

    def draw_frame(self):
        # Draw each particle as a point
        pass
