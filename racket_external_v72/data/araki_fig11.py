"""Convert manually identified Feint OX square centres in Fig. 11 to normal restitution.

Source PDF p.28 (printed p.44) rendered using pdftoppm -r 110 with a 910x1287 page.
The stored audit coordinates refer to a 3x cubic enlargement of a cropped raster.
"""
import csv
from pathlib import Path

# Grid corners in original page pixels. The scan is tilted, so calibrate y using both axes.
AXIS_LEFT = 320.0
AXIS_RIGHT = 802.8
BOTTOM_LEFT = 599.0
BOTTOM_RIGHT = 592.0
TOP_LEFT = 260.0
TOP_RIGHT = 254.0
# (x,y) on the audit overlay: page region (307,344)-(710,416); scale 3; offset (21,22).
# Centers were read independently of the outcome panels in Fig. 12.
CENTERS = [
    (60,135), (145,129), (184,154), (183,175), (233,187), (304,214),
    (355,168), (382,181), (509,175), (550,186), (649,184), (741,154),
    (783,166), (901,173), (992,152), (992,181), (1105,94),
]

def convert(x_mark, y_mark):
    x = 307 + (x_mark - 21) / 3
    y = 344 + (y_mark - 22) / 3
    horizontal = (x - AXIS_LEFT) / (AXIS_RIGHT - AXIS_LEFT)
    bottom = BOTTOM_LEFT + horizontal * (BOTTOM_RIGHT - BOTTOM_LEFT)
    top = TOP_LEFT + horizontal * (TOP_RIGHT - TOP_LEFT)
    return 250 * horizontal, (bottom - y) / (bottom - top)

def main():
    destination = Path(__file__).with_name("araki_fig11.csv")
    with destination.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["square_index","x_overlay_px","y_overlay_px","n_in_rps_fig11","normal_restitution","normal_velocity_nominal_m_s","pairing_status"])
        for i, (x,y) in enumerate(CENTERS):
            spin, restitution = convert(x,y)
            status = "duplicate_spin_exclude" if i in (2,3,14,15) else "unique"
            writer.writerow([i+1,x,y,round(spin,3),round(restitution,6),round(restitution*30/3.6,6),status])

if __name__ == "__main__":
    main()
