from pathlib import Path
import os
os.chdir(Path(__file__).resolve().parent)
from aligned import run
run()
