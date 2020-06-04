import numpy as np
from abc import ABC, abstractmethod
from skimage.draw import circle, rectangle, polygon


def circular_mask(shape):
    mask = np.zeros(shape, dtype=np.uint8)
    rr, cc = circle(shape[0]/2, shape[1]/2, radius=shape[0] / 3, shape=shape)
    mask[rr, cc] = 1
    
    return mask


def striped_mask(shape):
    mask = np.zeros(shape, dtype=np.uint8)
    mask[::2] = 1

    return mask


def concentric_rectangle_mask(shape, width):
    mask = np.ones(shape, dtype=np.uint8)
    rect = np.zeros(shape, dtype=np.uint8)

    for i in range(1, int(shape[0] / (2 * width)), 2):
        rect = np.zeros(shape, dtype=np.uint8)
        start = (i*width, i*width)
        end = (shape[0] - i*width, shape[1] - i*width)
        rr, cc = rectangle(start=start, end=end, shape=mask.shape)
        rect[rr, cc] = 1
        mask -= rect

        rect = np.zeros(shape, dtype=np.uint8)
        start = ((i+1) * width, (i+1) * width)
        end = (shape[0] - (i+1) * width, shape[1] - (i+1) * width)
        rr, cc = rectangle(start=start, end=end, shape=mask.shape)
        rect[rr, cc] = 1
        mask += rect


    return mask


def triangle_mask(shape, p1, p2, p3):
    mask = np.zeros(shape, dtype=np.uint8)
    r = np.array([p1[0], p2[0], p3[0]])
    c = np.array([p1[1], p2[1], p3[1]])
    rr, cc = polygon(r, c)
    mask[rr, cc] = 1

    return mask


class AbstractProcessing(ABC):
    """ Base class for post-processing. """
    @abstractmethod
    def apply(self, *args, **kwargs):
        pass


class Quantize(AbstractProcessing):
    """ Apply quantization to each frame of a given 3D input. """
    def __init__(self, bins=2):
        self.bins = bins

    def apply(self, images):
        w = images.max() / self.bins

        for i in range(images.shape[0]):
            images[i, :, :] -= (images[i, :, :] - (images[i, :, :] // w) * w).astype('uint8')

        return images


class AdjustBrightness(AbstractProcessing):
    """ Gamma < 1 will decrease brightness, Gamma > 1 will increase it. """
    def __init__(self, gamma):
        self.gamma = gamma
    
    def apply(self, images):
        # Normalize, then apply brightness correction
        images = (images / images.max()) ** (1 / self.gamma)
        # Convert back to grayscale [0, 255]
        images = ((images - images.min()) * (1 / (images.max() - images.min()) * 255)).astype('uint8')

        return images


class Mask(AbstractProcessing):
    """ Apply a binary mask to each frame of a given 3D input. """
    def __init__(self, mask):
        self.mask = mask
    
    def apply(self, images):
        images *= self.mask

        return images


class Border(AbstractProcessing):
    def __init__(self, margin, width):
        self.margin = margin
        self.width = width
    
    def apply(self, images):
        # White border
        images[:, self.margin:self.margin + self.width, :] = 255
        images[:, -self.margin - self.width:-self.margin, :] = 255
        images[:, :, self.margin:self.margin + self.width] = 255
        images[:, :, -self.margin - self.width:-self.margin] = 255
        
        # Black margin
        images[:, 0:self.margin, :] = 0
        images[:, -self.margin:, :] = 0
        images[:, :, 0:self.margin] = 0
        images[:, :, -self.margin:] = 0
        
        return images


class FromFunction(AbstractProcessing):
    """ Not tested, not fully compatible yet. """
    def __init__(self, fn=None, *args, **kwargs):
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
    
    def apply(self, images):
        for i in range(images.shape[0]):
            images[i, :, :] = self.fn(images[i, :, :], *self.args, **self.kwargs)

        return images


class Pipeline():
    """ Define an AbstractProcessing pipeline object. """
    def __init__(self, *args):
        self._processing_list = args

    def run(self, images):
        if not self.is_empty():
            for f in self._processing_list:
                images = f.apply(images)

        return images
    
    def is_empty(self):
        return len(self._processing_list) == 0
