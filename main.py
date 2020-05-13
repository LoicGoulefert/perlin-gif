import argparse

from perlin import PerlinGif
from postprocessing import Pipeline, AdjustBrightness, Quantize


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
    pipeline = Pipeline(AdjustBrightness(gamma=0.4), Quantize(bins=16))
    # pipeline = Pipeline()
    args['pipeline'] = pipeline

    # Render gif
    pg = PerlinGif(**args)
    pg.render()
