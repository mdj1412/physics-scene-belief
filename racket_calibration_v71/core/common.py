"""Rubber identity, sample ordering and calibration draws used throughout v71."""
from pathlib import Path
import numpy as np
import csv

HERE = Path(__file__).resolve().parent
V70 = HERE.parent.parent / "racket_transfer_v70"
GROUPS = {"hexer_duro": (1,3,9), "power_3": (2,4), "grass_dtecs": (5,6),
          "diamant": (7,), "blowfish": (8,), "abs_ii_soft": (10,)}
BUDGETS = (0,1,2,4,8,16,32)
DRAWS = 256
RADIUS = .02

def load_state(configs, split):
    rows = []
    for config in sorted(configs):
        path = V70 / "data" / "upstream" / split / f"racket_{config:02}.csv"
        with path.open() as handle:
            rows.extend(csv.DictReader(handle))
    def state(v, w):
        return np.array([[float(row[f"v_{v}_x"]),float(row[f"v_{v}_z"]),
                          RADIUS * float(row[f"spin_{w}_y"])] for row in rows])
    return state("in","before"), state("out","after")

def draw_indices(size, configs, count=DRAWS):
    return np.stack([np.random.default_rng(np.random.SeedSequence([710071,min(configs),draw]))
                     .choice(size, size=32, replace=False) for draw in range(count)])

if __name__ == "__main__":
    np.savez_compressed(HERE/"draw_indices.npz", **{name:draw_indices(len(load_state(g,"train")[0]),g)
                        for name,g in GROUPS.items()})
