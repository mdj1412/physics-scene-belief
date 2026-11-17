"""Set paths before running the video reconstruction."""

from pathlib import Path
import os

VIDEO = Path(os.environ["PRESTON_VIDEO"])
OUTPUT = Path(os.environ["PRESTON_OUTPUT"])
CACHE = OUTPUT / "centers"
