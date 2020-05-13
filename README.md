# perlin-gif
Tool for perlin gif creation.

## Installation
```
sudo apt-get install gifsicle

# Create virtual env
python3 -m venv env
source env/bin/activate

# Upgrade pip
pip install -U pip

git clone https://github.com/LoicGoulefert/perlin-gif.git

cd perlin-gif
pip install -r requirements.txt
```

## Usage

```
python main.py -h
usage: main.py [-h] [-d {3,4}] [-n N [N ...]] [-fps FPS] [-frames FRAMES]
               [-s S [S ...]] [-o {1,2,3,4}] [-r R] [-c] [-out OUT] [-R]

CLI tool to create perlin noise gifs

optional arguments:
  -h, --help      show this help message and exit
  -d {3,4}        noise dimension
  -n N [N ...]    specify the gif dimension
  -fps FPS        specify the framerate
  -frames FRAMES  how many frames in the gif
  -s S [S ...]    specify the scale (tuple of floats in the [0, 1] range)
  -o {1,2,3,4}    how many octaves to use
  -r R            radius (for 4D noise)
  -c, --compress  set this flag to enable gif compression
  -out OUT        output file name (will be created)
  -R              set this flag to use a random starting point in the noise
                  function


```

For example, `python perlin.py -n 400 400 -frames 60 -fps 60 --compress` will yield a 400x400 compressed gif, 1 second at 60FPS.
