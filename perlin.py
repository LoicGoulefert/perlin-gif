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


def _to_gif(images, fps, output_file, compression):
    kwargs = {'duration': 1 / fps}
    imageio.mimsave(output_file, images, **kwargs)
    if compression:
        optimize(output_file)


class PerlinGif():
    def __init__(self, **config):
        self.noise_dimension = config["noise_dimension"]
        self.size = config["size"]
        self.fps = config["fps"]
        self.frames = config["frames"]
        self.scale = config["scale"]
        self.octaves = config["octaves"]
        self.radius = config["radius"]
        self.compression = config["compression"]
        self.output_file = config["output_file"]
        self.pipeline = config["pipeline"]
        self.random_seed = config["random_seed"]
        self.shape = (*self.size, self.frames)

    def _make_3d_images(self):
        images = _simplex_noise3d(self.shape, self.scale, self.octaves, self.random_seed)
        images = images.transpose(2, 0, 1)

        return images

    def _make_4d_images(self):
        images = _simplex_noise4d(self.shape, self.scale, self.octaves, self.radius, self.random_seed)
        images = images.transpose(2, 0, 1)

        return images

    def render(self):
        if self.noise_dimension == 3:
            self.images = self._make_3d_images()
        else:
            self.images = self._make_4d_images()

        if not self.pipeline.is_empty():
            print("Running pipeline")
            self.images = self.pipeline.run(self.images)

        _to_gif(self.images, self.fps, self.output_file, self.compression)


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
        self.images = np.zeros(self.shape)

    def flow_field(self):
        flow_field = np.zeros((*self.shape, 2)) # if 256, 256, 30 -> 256, 256, 30, 2
        radius = 0.1
        scale = (0.1, 0.1)

        for i in range(self.shape[2]):
            cos_value = self.radius * math.cos(2 * math.pi * (i / self.shape[2]))
            sin_value = self.radius * math.sin(2 * math.pi * (i / self.shape[2]))
            for x in range(self.shape[0]):
                for y in range(self.shape[1]):
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
        self.flow_field = flow_field

        return self.flow_field

    def add_particles(self, n_particles):
        # add particles at random location along the y=0 axis
        self.particles = []
        rand_x = np.random.randint(0, self.shape[0], size=n_particles)
        for x in rand_x:
            pos = Vec2D(x, 0)
            vel = Vec2D(0, 0)
            acc = Vec2D(0, 0)
            self.particles.append(Particle(pos, vel, acc))

    def update_particles(self):        
        # for each particle, find associated vector, apply force
        for i in range(self.frames):
            for particle in self.particles:
                force = Vec2D(*self.flow_field[round(particle.pos.x), round(particle.pos.y), i, :])
                particle.apply(force)

                # Wrap particle around frame
                if particle.pos.x > self.shape[0]:
                    particle.pos.x = 0
                if particle.pos.x < 0:
                    particle.pos.x = self.shape[0] - 1
                if particle.pos.y > self.shape[1]:
                    particle.pos.y = 0
                if particle.pos.y < 0:
                    particle.pos.y = self.shape[1] - 1
            
            self._draw_frame(i)
        self.images = ((self.images - self.images.min()) * (1 / (self.images.max() - self.images.min()) * 255)).astype('uint8')

    def _draw_frame(self, index):
        for particle in self.particles:
            x = particle.pos.x
            y = particle.pos.y
            self.images[x, y, index] = 1
    
    def render(self):
        pass

    def _to_gif(self):
        kwargs = {'duration': 1 / self.fps}
        imageio.mimsave(self.output_file, self.images, **kwargs)
        if self.compress:
            optimize(self.output_file)


class Particle():
    def __init__(self, pos, velocity, acceleration):
        self.pos = Vec2D(0, 0)
        self.velocity = Vec2D(0, 0)
        self.acceleration = Vec2D(0, 0)

    def update(self):
        self.velocity += self.acceleration
        self.position += self.velocity
        self.acceleration.reset()

    def apply(self, force):
        print("Acc =", self.acceleration)
        self.acceleration += force
    
    def __repr__(self):
        return f"Particle(pos={self.pos}, vel={self.velocity}, acc={self.acceleration})"


class Vec2D():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        self.x += other.x
        self.y += other.y
    
    def __repr__(self):
        return f"Vec2D(x={self.x}, y={self.y})"

    def reset(self):
        self.x = 0
        self.y = 0
