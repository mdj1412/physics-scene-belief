"""Recompute Cross's outbound translation from publisher Fig. 9, manuscript Fig. 8.

Pixel coordinates are measured in the original 600 x 227 publisher graphic.
The digitization envelope permits independent ±1-px error at each centroid/tick.
It excludes measurement and across-bounce variation.
"""
from pathlib import Path
from statistics import linear_regression
import csv

PANELS = [
    dict(panel="a", vx_in=5.8, vy_in=-6.8, hz=300,
         ticks_x=(50, 277), x_span=0.15, ticks_y=(31, 190), y_span=0.10,
         xs=(178.2, 190.9, 203.4, 216.0), ys=(153, 128.1, 105.3, 81),
         xs_in=(53.3, 83.4, 110.3, 141.3), length=42, width=40),
    dict(panel="b", vx_in=21.1, vy_in=-6.8, hz=600,
         ticks_x=(363, 592.5), x_span=0.30, ticks_y=(30, 191), y_span=0.06,
         xs=(526.9, 546.8, 567.3, 587.7), ys=(155.1, 135.3, 113.8, 92.7),
         xs_in=(380.3, 407.5, 434.1), length=99, width=44),
]

def summarize(p):
    frames = range(len(p["xs"]))
    slope = linear_regression(frames, p["xs"]).slope
    width = p["ticks_x"][1] - p["ticks_x"][0]
    height = p["ticks_y"][1] - p["ticks_y"][0]
    factor = p["hz"] * p["x_span"]
    # OLS slope is a linear combination of centroids, so their worst-case
    # effect under independent bounded errors equals the L1 norm of its weights.
    centered = [index - sum(frames) / len(frames) for index in frames]
    shift = sum(map(abs, centered)) / sum(index**2 for index in centered)
    return dict(case=p["panel"], vx_in_ms=p["vx_in"], vy_in_ms=p["vy_in"],
                omega_in_rad_s=0, radius_m=0.033, mass_kg=0.0581,
                footprint_length_mm=p["length"], footprint_width_mm=p["width"],
                vx_out_digitized_ms=round(factor*slope/width, 5),
                vx_out_digitization_low_ms=round(factor*(slope-shift)/(width+2),5),
                vx_out_digitization_high_ms=round(factor*(slope+shift)/(width-2),5),
                vy_out_digitized_ms=round(-linear_regression(frames,p["ys"]).slope * p["y_span"] / height * p["hz"],5),
                vx_in_digitized_control_ms=round(linear_regression(range(len(p["xs_in"])),p["xs_in"]).slope * factor/width,5),
                fps=p["hz"], n_outbound_points=len(frames))
if __name__ == "__main__":
    rows = [summarize(p) for p in PANELS]
    with Path(__file__).with_name("cross_2014_examples.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
