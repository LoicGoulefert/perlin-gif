import math
import argparse

import imageio
import numpy as np
from noise import snoise2, snoise3, snoise4
from pygifsicle import optimize


def _simplex_noise3d(shape, scale, octaves=1):
    img = np.zeros(shape)
    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                img[x, y, z] = snoise3(x * scale[0], y * scale[1], z * scale[2], octaves=octaves)
    
    img = ((img - img.min()) * (1 / (img.max() - img.min()) * 255)).astype('uint8')
    
    return img


def _simplex_noise4d(shape, scale, radius=1.5, octaves=1):
    img = np.zeros(shape)
    for i in range(shape[2]):
        for x in range(shape[0]):
            for y in range(shape[1]):
                cos_value = radius * math.cos(2 * math.pi * (i / shape[2]))
                sin_value = radius * math.sin(2 * math.pi * (i / shape[2]))
                img[x, y, i] = snoise4(x * scale[0], y * scale[1], cos_value, sin_value)
    
    img = ((img - img.min()) * (1 / (img.max() - img.min()) * 255)).astype('uint8')
    
    return img


def _to_gif(images, output, fps):
    kwargs = {'duration': 1 / fps}
    imageio.mimsave(output, images, **kwargs)


def make_3d_gif(N, shape, scale, frames, octaves=1, fps=30, output='perlin3d.gif'):
    images = _simplex_noise3d(shape, scale, octaves=octaves)
    images = images.transpose(2, 0, 1)
    _to_gif(images, output, fps)


def make_4d_gif(N, shape, scale, frames, octaves=1, radius=1.5, fps=30, output='perlin4d.gif'):
    images = _simplex_noise4d(shape, scale, octaves=octaves, radius=radius)
    images = images.transpose(2, 0, 1)
    _to_gif(images, output, fps)

 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI tool to create perlin noise gifs")
    parser.add_argument("-d", type=int, choices=[3, 4], help="noise dimension", default=4)
    parser.add_argument("-n", type=int, nargs='+', help="specify the gif dimension", default=(256, 256))
    parser.add_argument("-fps", type=int, help="specify the framerate", default=30)
    parser.add_argument("-frames", type=int, help="how many frames in the gif", default=30)
    parser.add_argument("-s", type=float, nargs='+', help="specify the scale (tuple of floats in the [0, 1] range)", default=(0.01, 0.01))
    parser.add_argument("-o", type=int, choices=[1, 2, 3, 4], help="how many octaves to use", default=1)
    parser.add_argument("-r", type=float, help="radius (for 4D noise)", default=1)
    parser.add_argument("-c", "--compress", action='store_true', help="set this flag to enable gif compression", default=False)
    parser.add_argument("-out", type=str, help="output file name (will be created)", default="out.gif")
    args = parser.parse_args()

    noise_dim = args.d
    N = args.n
    frames = args.frames
    fps = args.fps
    shape = (*N, frames)
    scale = args.s
    octaves = args.o
    radius = args.r
    compress = args.compress
    output = args.out

    if noise_dim == 3:
        make_3d_gif(N, shape, scale, frames, octaves=octaves, fps=fps, output=output)
    else:
        make_4d_gif(N, shape, scale, frames, octaves=octaves, fps=fps, radius=radius, output=output)

    if compress:
        optimize(output)
