"""Draw fresh catalog supports on a Slurm worker, with a balanced environment design."""
import os
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT / "image_pressure_v45"))
from runner45 import old, infer, r, v38, torch

old.BASE = 480000000
old.SUPPORT_BASE = 800
FAMILIES = ("ogee", "cove", "reed", "flute")
HEIGHTS = (19, 31)
OFFSETS = (-.5, 0., .5)
old.world.TARGET_SHAPES = FAMILIES


def make_population(panel):
    seed, alpha, signs, regimes = old.population(panel)
    theta, idx, surfaces, laws = [], [], [], []
    for e in range(4):
        k = (e + panel) % 4
        surface = old.world.TARGET_SURFACES[k // 2]
        law = old.world.TARGET_LAWS[k % 2]
        ctx = (FAMILIES[0], "hollow", .019, surface, law, int(signs[e]), 800 + panel)
        theta.append(old.world.draw_object(128, seed+200+e, 20, alpha[e], ctx).cpu().numpy())
        idx.append(old.world.w.stream(seed+200+e, 4320).integers(old.world.w.SUPPORT_SIZE, size=128))
        surfaces.append(surface)
        laws.append(law)
    path = ROOT/"worlds"/f"panel_{panel:02d}.npz"
    path.parent.mkdir(exist_ok=True)
    v38.atomic_npz(path, theta=theta, object_index=idx, alphas=alpha, signs=signs,
                   regimes=regimes, seed=seed, support_panel=800+panel, surface=surfaces, law=laws)


if __name__ == "__main__":
    assert os.getenv("SLURM_JOB_ID") and not torch.cuda.is_available()
    for panel in range(12):
        make_population(panel)
