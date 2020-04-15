import numpy as np
from abc import ABC, abstractmethod
from sklearn.cluster import KMeans


class AbstractProcessing(ABC):
    @abstractmethod
    def apply(self, *args, **kwargs):
        pass


class Quantize(AbstractProcessing):
    """ Not quite what I wanted but it looks cool """
    def __init__(self, bins=2):
        self.bins = bins

    def apply(self, images):
        w = images.max() / self.bins

        for i in range(images.shape[0]):
            images[i, :, :] -= (images[i, :, :] - (images[i, :, :] // w) * w).astype('uint8')

        return images


class FromFunction(AbstractProcessing):
    """ Not tested """
    def __init__(self, fn=None, **fn_kwargs):
        self.fn = fn
        self.fn_kwargs = fn_kwargs
    
    def apply(self, images):
        for i in range(images.shape[0]):
            images[i, :, :] = self.fn(images[i, :, :])

        return images


class Pipeline():
    def __init__(self, *args):
        self._processing_list = args

    def run(self, images):
        if not self.is_empty():
            for f in self._processing_list:
                images = f.apply(images)

        return images
    
    def is_empty(self):
        return len(self._processing_list) == 0
