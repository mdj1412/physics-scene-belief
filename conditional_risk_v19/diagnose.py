"""Post hoc paired comparisons and support for the pre-specified primary results."""
from analyze import interval
from run import NAMES, CELLS
from common import R, write, require
import numpy as np

require()
report = {}
for c, k in CELLS:
    data = np.load(R / f"policy_{c}_{k}_0.npz")
    choice, risk = data["choices"], data["risks"]
    loss = np.load(R / f"loss_{c}_{k}.npz")["world_action_loss"]
    a = choice[:, NAMES.index("conditional/4096")]
    rows = {}
    for name in ("self_label/4096", "external/4096", "v18_global", "v18_adaptive", "probe", "nested2048"):
        b = choice[:, NAMES.index(name)]
        i = np.arange(len(a))
        delta = loss[i, a-1] - loss[i, b-1]
        rows[name] = dict(different=int((a!=b).sum()), delta95=interval(delta))
    rows["range_tie_counts"] = {name: int((np.ptp(risk[:,NAMES.index(name)], axis=1)<=1e-12).sum())
                                for name in ("self_label/4096", "conditional/4096", "external/4096", "v18_global")}
    report[f"{c}_{k}"] = rows
write(R / "diagnostics.json", report)
