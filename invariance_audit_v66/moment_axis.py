"""How much of the frozen axis is inside the unweighted quadratic-profile subspace?"""
import json
import os
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent
assert os.getenv("SLURM_JOB_ID") and os.getenv("CUDA_VISIBLE_DEVICES") == ""
axis = json.loads((root.parent/"image_pressure_v45"/"source_axis45.json").read_text())
x = 2 * np.arange(-3., 4.)
basis = np.linalg.qr(np.column_stack((np.ones(7), x, .5*x*x)))[0]
info = {}
for h in ("19","31"):
    u = np.array(axis[h]["axis"])
    np.testing.assert_allclose(u@u, 1, atol=1e-8)
    squares = (basis.T @ u)**2
    info[h] = dict(squared_norm_fraction=float(squares.sum()), ordered_components=squares.tolist())
(root/"moment_axis.json").write_text(json.dumps(dict(columns=["constant","linear","quadratic residualized on constant and linear"], by_height_mm=info),indent=2)+"\n")
