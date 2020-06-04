import argparse
import numpy as np

from perlin import PerlinGif, PerlinFlowField
from postprocessing import Pipeline, AdjustBrightness, Quantize, Border, Mask, concentric_rectangle_mask


if __name__ == "__main__":

    gif_config = {
        "noise_dimension": 4,
        "size": (256, 256),
        "fps": 30,
        "frames": 30,
        "scale": (0.01, 0.01),
        "octaves": 1,
        "radius": 0.1,
        "compression": False,
        "output_file": "out.gif",
        "random_seed": False,
        "pipeline": Pipeline()
        # pipeline = Pipeline(AdjustBrightness(gamma=0.5), Quantize(bins=20), Border(margin=60, width=4))
    }
    
    # Sanity check
    if gif_config['noise_dimension'] == 3:
        assert len(gif_config['scale']) == 3, \
        "3 dimension scale needed for 3D noise. Got {} ({}D).".format(gif_config['scale'], len(gif_config['scale']))

    # Render gif
    pg = PerlinGif(**gif_config)
    # pg = PerlinFlowField(**gif_config)
    pg.render()
