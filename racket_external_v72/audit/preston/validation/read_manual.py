"""Evaluate four visually annotated intervals; positive y is screen-down."""
import csv
import math
from pathlib import Path

root = Path(__file__).resolve().parent
with (root / 'manual_angles.csv').open() as handle:
    for row in csv.DictReader(handle):
        n = int(row['i1']) - int(row['i0'])
        y = float(row['center_y1_px']) - float(row['center_y0_px'])
        a = float(row['angle1_deg_cw']) - float(row['angle0_deg_cw'])
        # Range is a deterministic perturbation audit, not a confidence interval.
        vals = [(y + yd + rad * math.radians(a + ad)) / n
                for yd in [-1., 1.] for ad in [-12., 12.] for rad in [14., 17.]]
        print(row['image_ref'], row['phase'], round(y / n, 3),
              round(a / n, 3), round((y + 15.5 * math.radians(a)) / n, 3),
              round(min(vals), 3), round(max(vals), 3))
