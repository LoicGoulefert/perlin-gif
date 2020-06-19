# Standard libs
import math
import argparse

# Third-party libs
import imageio
import numpy as np
from noise import snoise2, snoise3, snoise4
from pygifsicle import optimize
import matplotlib.pyplot as plt
from skimage.draw import circle_perimeter, line
from scipy.spatial import Voronoi


def _simplex_noise3d(shape, scale, octaves, random):
    """Return a sequence of images representing 3D simplex noise."""
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
    """Returns a sequence of images representing 4D (looping) simplex noise."""
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
    """Convert a sequence of images to a gif."""
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
    def __init__(self, **config):
        self.noise_dimension = config["noise_dimension"]
        self.size = config["size"]
        self.fps = config["fps"]
        self.frames = config["frames"]
        self.scale = config["scale"]
        self.radius = config["radius"]
        self.compression = config["compression"]
        self.output_file = config["output_file"]
        self.pipeline = config["pipeline"]
        self.random = config["random_seed"]
        self.shape = (*self.size, self.frames)
        self.images = np.zeros(self.shape)
    
    def _get_force(self, x, y, t, magnitude=1):
        cos_value = self.radius * math.cos(2 * math.pi * t)
        sin_value = self.radius * math.sin(2 * math.pi * t)
        angle = snoise4(x * self.scale[0], y * self.scale[1], cos_value, sin_value)

        return Vec2D(np.cos(angle) * magnitude, np.sin(angle) * magnitude)

    def _add_particles(self, n_particles):
        self.particles = []
        rand_x = np.random.randint(0, self.shape[0], size=n_particles)
        rand_y = np.random.randint(0, self.shape[1], size=n_particles)

        for x, y in zip(rand_x, rand_y):
            pos = Vec2D(x, y)
            vel = Vec2D(0, 0)
            acc = Vec2D(0, 0)
            self.particles.append(Particle(pos, vel, acc))

    def _update_particle(self, particle, i):
        x = int(round(particle.pos.x))
        y = int(round(particle.pos.y))
        if x == self.size[0]:
            x -= 1
        if y == self.size[1]:
            y -= 1

        force = self.get_force(x, y, i, magnitude=3)
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

    def _draw_frame(self, index):
        for particle in self.particles:
            self.images[int(particle.pos.x), int(particle.pos.y), index] = 1

            # Circle particles
            # rr, cc = circle_perimeter(int(particle.pos.x), int(particle.pos.y), 2)
            # rr = [r if r < self.shape[0]  else self.shape[0] - 1 for r in rr]
            # cc = [c if c < self.shape[1] else self.shape[1] - 1 for c in cc]
            # self.images[rr, cc, index] = 1

    def update_particles(self):
        for i in range(self.frames):
            for particle in self.particles:
                self._update_particle(particle, i)

            self._draw_frame(i)

        self.images = ((self.images - self.images.min()) * (1 / (self.images.max() - self.images.min()) * 255)).astype('uint8')

    def render(self, n_particles=100):
        self.add_particles(n_particles)
        self.update_particles()
        self.images = self.images.transpose(2, 0, 1)

        if not self.pipeline.is_empty():
            print("Running pipeline")
            self.images = self.pipeline.run(self.images)

        _to_gif(self.images, self.fps, self.output_file, self.compression)


class Particle():
    def __init__(self, pos, velocity, acceleration):
        self.pos = pos
        self.velocity = Vec2D(0, 0)
        self.acceleration = Vec2D(0, 0)

    def update(self):
        self.velocity += self.acceleration
        self.pos += self.velocity
        self.acceleration.reset()
        self.velocity.reset()

    def apply(self, force):
        self.acceleration += force
        self.update()

    def __repr__(self):
        return f"Particle(pos={self.pos}, vel={self.velocity}, acc={self.acceleration})"


class Vec2D():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        self.x += other.x
        self.y += other.y
        return self

    def __repr__(self):
        return f"Vec2D(x={self.x}, y={self.y})"

    def reset(self):
        self.x = 0
        self.y = 0
