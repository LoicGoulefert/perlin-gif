import argparse
import numpy as np

from perlin import PerlinGif, PerlinFlowField
from postprocessing import *


if __name__ == "__main__":
    size = (512, 512)

    gif_config = {
        "noise_dimension": 4,
        "size": size,
        "fps": 30,
        "frames": 90,
        "scale": (0.005, 0.005),
        "octaves": 1,
        "radius": 0.3,
        "compression": False,
        "output_file": "out.gif",
        "random_seed": True,
        # "pipeline": Pipeline()
        "pipeline": Pipeline(
            AdjustBrightness(gamma=0.5), 
            Quantize(bins=20),
            # Mask(triangle_mask(size, [60, size[0] / 2], [size[0] - 60, 60], [size[0] - 60, size[0] - 60])),
            Border(margin=60, width=4),
        )
    }
    
    # Sanity check
    if gif_config['noise_dimension'] == 3:
        assert len(gif_config['scale']) == 3, \
        "3 dimension scale needed for 3D noise. Got {} ({}D).".format(gif_config['scale'], len(gif_config['scale']))

    # Render gif
    pg = PerlinGif(**gif_config)
    # pg = PerlinFlowField(**gif_config)
    pg.render()
