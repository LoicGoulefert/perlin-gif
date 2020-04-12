import math
import imageio
import numpy as np
from noise import snoise2, snoise3, snoise4


def simplex_noise3d(shape, scale, octaves=1):
    img = np.zeros(shape)
    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                img[x, y, z] = snoise3(x * scale[0], y * scale[1], z * scale[2], octaves=octaves) * 128.0 + 127.0
    
    return img


def simplex_noise4d(shape, scale, radius=1.5, octaves=1):
    img = np.zeros(shape)
    for i in range(shape[2]):
        for x in range(shape[0]):
            for y in range(shape[1]):
                cos_value = radius * math.cos(2 * math.pi * (i / shape[2]))
                sin_value = radius * math.sin(2 * math.pi * (i / shape[2]))
                img[x, y, i] = snoise4(x * scale[0], y * scale[1], cos_value, sin_value)
                img[x, y, i] = img[x, y, i] * 128.0 + 127.0
    
    return img


def make_3d_gif(N, shape, scale, frames, octaves=1, fps=30, output='perlin3d.gif'):
    images = simplex_noise3d(shape, scale, octaves=octaves)
    images = images.astype('uint8')
    images = images.transpose(2, 0, 1)

    kwargs = {'duration': 1 / fps}
    imageio.mimsave(output, images, **kwargs)


def make_4d_gif(N, shape, scale, frames, octaves=1, radius=1.5, fps=30, output='perlin3d.gif'):
    images = simplex_noise4d(shape, scale, octaves=octaves, radius=radius)
    images = images.astype('uint8')
    images = images.transpose(2, 0, 1)

    kwargs = {'duration': 1 / fps}
    imageio.mimsave(output, images, **kwargs)

 
if __name__ == "__main__":
    N = 512
    frames = 90
    shape = (N, N, frames)
    scale4D = (0.005, 0.005)
    octaves = 1
    radius = 0.5

    make_4d_gif(N, shape, scale4D, frames, octaves=octaves, radius=radius)
