"""Post-hoc paired sole contrast at fold 31 mm, +0.5 mm, and support diagnostics."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import t

root = Path(__file__).resolve().parent
panel_effects = []
for panel in sorted((root / "labels44").glob("panel_*")):
    effects = []
    for phys in sorted(panel.glob("fold_*_h31.npz")):
        with np.load(phys) as a, np.load(root / "results" / panel.name / phys.name) as p:
            assert a["valid"].all()
            event_difference = a["labels"][:,1,2].astype(float) - a["labels"][:,0,2]
            prob_difference = (p["pred"][1,0,:,:8] - p["pred"][0,0,:,:8]).mean(axis=-1)
            loss_effect = (-2 * prob_difference * event_difference).mean(axis=(1,2))
            support_move = abs(a["support_mm"][:,1,2] - a["support_mm"][:,0,2]).mean(axis=(1,2))
            effects.append(np.stack((loss_effect, support_move), axis=-1))
    assert len(effects) == 12
    panel_effects.append(np.mean(effects, axis=0))
v = np.array(panel_effects)
assert v.shape == (12,2,2)
m = v.mean(axis=0)
width = t.ppf(1-.025/2,11) * v.std(axis=0,ddof=1) / np.sqrt(12)
(root / "sole_contrast44.json").write_text(json.dumps(dict(
    axes=["radius 0.5 / 2 mm","D(L2)-D(L0) / abs initial support shift (mm)","mean, lower, upper"],
    estimate=np.stack((m,m-width,m+width),axis=-1).tolist(),
    inference="Exploratory paired panel t; two radii per measurement family"),indent=2)+"\n")
