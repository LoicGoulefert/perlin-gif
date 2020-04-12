import numpy as np
import matplotlib.pyplot as plt
import imageio
from noise.perlin import SimplexNoise
from noise import snoise2, snoise3, snoise4


def perlin2D(shape, res):
    def smoothstep(t):
        return 6*t**5 - 15*t**4 + 10*t**3
    
    dim = len(shape)
    
    delta = tuple(res[i] / shape[i] for i in range(dim))
    d = tuple(shape[i] // res[i] for i in range(dim))

    mgrid_slices = [slice(0, res[i], delta[i]) for i in range(dim)]
    grid = np.mgrid[mgrid_slices]

    transpose_args = list(range(1, dim+1))
    transpose_args.append(0) 
    grid = grid.transpose(transpose_args) % 1

    # Gradients
    # angles = 2*np.pi*np.random.rand(*[res[i]+1 for i in range(dim)])
    gradients = np.random.rand(*[res[i]+1 for i in range(dim)], dim)
    g00 = gradients[0:-1,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1,1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:,1:].repeat(d[0], 0).repeat(d[1], 1)

    # Ramps
    n00 = np.sum(grid * g00, 2)
    n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:,:,0], grid[:,:,1]-1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)

    # Interpolation
    t = smoothstep(grid)
    n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
    n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11

    return np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1)


def simplex_noise3d(shape, scale, octaves=1):
    img = np.zeros(shape)
    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                img[x, y, z] = snoise3(x * scale[0], y * scale[1], z * scale[2], octaves=octaves) * 128.0 + 127.0
    
    return img


def make_2d_gif(N, scale, frames, octaves=1, fps=30, output='perlin2d.gif'):
    shape = (N, N)
    images = []

    for i in range(frames):
        images.append(simplex_noise2d(shape, scale, octaves))
        noise.randomize()
    
    # Range [0, 255]
    images = [(255 * (image - np.min(image)) / np.ptp(image)).astype('uint8') for image in images]

    kwargs = {'duration': 1 / fps}
    imageio.mimsave(output, images, **kwargs)


def make_3d_gif(N, scale, frames, octaves=1, fps=30, output='perlin3d.gif'):
    shape = (N, N, frames)
    
    images = simplex_noise3d(shape, scale)
    images = images.astype('uint8')
    
    images = images.transpose(2, 0, 1)

    kwargs = {'duration': 1 / fps}
    imageio.mimsave(output, images, **kwargs)


def make_4d_gif(N, scale, frames, octaves=1, fps=30, output='perlin3d.gif'):
    shape = (N, N, frames)
    
    images = simplex_noise3d(shape, scale)
    images = images.astype('uint8')
    
    images = images.transpose(2, 0, 1)

    kwargs = {'duration': 1 / fps}
    imageio.mimsave(output, images, **kwargs)


if __name__ == "__main__":
    N = 512
    scale2D = (0.01, 0.01)
    scale3D = (0.005, 0.005, 0.02)
    frames = 60

    make_3d_gif(N, scale3D, frames)
