import math
import argparse

import imageio
import numpy as np
from noise import snoise2, snoise3, snoise4
from pygifsicle import optimize

from postprocessing import Quantize, Pipeline, FromFunction, AdjustBrightness, Mask, circular_mask


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI tool to create perlin noise gifs")
    parser.add_argument("-d", type=int, choices=[3, 4], help="noise dimension", default=4)
    parser.add_argument("-n", type=int, nargs='+', help="specify the gif dimension", default=(256, 256))
    parser.add_argument("-fps", type=int, help="specify the framerate", default=30)
    parser.add_argument("-frames", type=int, help="how many frames in the gif", default=30)
    parser.add_argument("-s", type=float, nargs='+', help="specify the scale (tuple of floats in the [0, 1] range)", default=(0.01, 0.01))
    parser.add_argument("-o", type=int, choices=[1, 2, 3, 4], help="how many octaves to use", default=1)
    parser.add_argument("-r", type=float, help="radius (for 4D noise)", default=0.1)
    parser.add_argument("-c", "--compress", action="store_true", help="set this flag to enable gif compression", default=False)
    parser.add_argument("-out", type=str, help="output file name (will be created)", default="out.gif")
    parser.add_argument("-R", action="store_true", help="set this flag to use a random starting point in the noise function", default=False)

    args = vars(parser.parse_args())

    # Sanity check
    if args['d'] == 3:
        assert len(args['s']) == 3, "3 dimension scale needed for 3D noise. Got {} ({}D).".format(args['s'], len(args['s']))

    # Create post-processing pipeline
    mask = circular_mask(args['n'])
    pipeline = Pipeline(AdjustBrightness(gamma=0.4), Quantize(bins=16), Mask(mask))
    # pipeline = Pipeline()
    args['pipeline'] = pipeline

    # Render gif
    pg = PerlinGif(**args)
    pg.render()
