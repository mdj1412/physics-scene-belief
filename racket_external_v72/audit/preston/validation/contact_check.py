"""Independent screen-plane release check using stored centroids and manual stripe readings."""
import csv
import json
import math
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
WINDOWS = {
    3: ((13, 33), (44, 65)),
    15: ((17, 39), (45, 55)),
    17: ((8, 30), (37, 58)),
}

def points(centres, window):
    start, end = window
    return np.array([(i, *centres[str(i)]) for i in range(start, end + 1)
                     if str(i) in centres]).T

with (ROOT / 'manual_angles.csv').open() as f:
    angles = {(int(r['image_ref']), r['phase']): r for r in csv.DictReader(f)}
for ref, windows in WINDOWS.items():
    centres = json.loads((ROOT / f'centres_ref{ref}.json').read_text())
    x_pre, x_post = [np.polyfit(*points(centres, w)[:2], 1) for w in windows]
    t0 = (x_post[1] - x_pre[1]) / (x_pre[0] - x_post[0])
    for phase, w in zip(('before', 'after'), windows):
        r = angles[ref, phase]
        dt = int(r['i1']) - int(r['i0'])
        da = float(r['angle1_deg_cw']) - float(r['angle0_deg_cw'])
        v_range = []
        for offset in (-2, 0, 2):
            t, x, y = points(centres, (w[0], w[1] + offset))
            v_range.append(np.polyfit(t - t0, y, 2)[1])
        omega = math.radians(da) / dt
        slip = v_range[1] + 15.5 * omega
        bounds = [v + radius * math.radians(da + angular_error) / dt
                  for v, radius, angular_error in product(v_range, [14., 17.], [-12., 12.])]
        print(ref, phase, f't0={t0:.3f}', f'vy={v_range[1]:.3f}', f'omega_deg={da/dt:.3f}',
              f'contact_s={slip:.3f}', f'perturbation=[{min(bounds):.3f},{max(bounds):.3f}]')
