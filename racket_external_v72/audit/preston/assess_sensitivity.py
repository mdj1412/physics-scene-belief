from fit_kinematics import edge, omega
from paths import CACHE as P, OUTPUT as OUT, VIDEO
from concurrent.futures import ProcessPoolExecutor
import cv2
import numpy as np
import csv
import itertools


def audit(r):
    j = int(r["id"])
    d = np.loadtxt(P / ("r%02d.csv" % j), delimiter=",", skiprows=1)
    first = int(d[0, 0])
    c = cv2.VideoCapture(str(VIDEO))
    c.set(cv2.CAP_PROP_POS_FRAMES, first)
    imgs = []
    for _ in d:
        ok, fr = c.read()
        imgs.append(cv2.cvtColor(fr[:240, :315], cv2.COLOR_BGR2GRAY))
    c.release()
    bg = np.median(imgs[::3], axis=0).astype(np.uint8)
    res = {"id": j}
    for tag in ["in", "out"]:
        inds = np.arange(
            int(r["win_start_" + tag]) - first, int(r["win_end_" + tag]) - first + 1
        )
        ts = inds + first - int(r["contact"])
        R = np.array([edge(imgs[i], bg, *d[i, 1:3]) for i in inds])
        xs = R[:, 0].copy()
        ys = R[:, 1].copy()
        px = np.polyfit(ts, xs, 1)
        py = np.polyfit(ts, ys, 2)
        R[:, 0] = np.polyval(px, ts)
        R[:, 1] = np.polyval(py, ts)
        R[:, 2] = np.median(R[:, 2])
        cut = len(inds) // 2
        L = np.polyfit(ts[:cut], xs[:cut], 1)[0]
        H = np.polyfit(ts[-cut:], xs[-cut:], 1)[0]
        cx = R[:, 0]
        cy = R[:, 1]
        avg = float(np.polyfit(ts, ys, 1)[0])
        acc = float(r["accel_" + tag])
        vy = float(r["vx_" + tag])
        vz = float(r["vz_" + tag])
        ws = [float(r["omega_" + tag])]
        # At least 9 frame angular estimates. Capture texture and center uncertainty across overlapping subwindows.
        n = max(9, round(0.67 * len(inds)))
        for sub in [slice(0, n), slice(-n, None)]:
            ws.append(
                -omega([imgs[i] for i in inds[sub]], R[sub], [0.4, 0.55, 0.7, 0.8])[0]
            )
        rad = float(r["radius_px"])
        # Downward acceleration bracket is a sensitivity model anchored to aggregate free flight over delivered frames, not a true bound.
        vels = [vy, avg, avg - 0.10 * ts.mean()]
        base = vy - rad * ws[0]
        options = [
            (val * np.cos(tilt) - vz * np.sin(tilt)) - (rad + dr) * w
            for val, w, dr, tilt in itertools.product(
                vels, ws, [-2, 2], np.deg2rad([-5, 5])
            )
        ]
        ff = np.array(imgs)
        # Check the moving ball's full patch in consecutive video frames for exact repeated images.
        repeated = 0
        for ii in range(len(inds) - 1):
            x = int(cx[ii])
            y = int(cy[ii])
            p1 = ff[inds[ii], y - 15 : y + 16, x - 15 : x + 16]
            p2 = ff[inds[ii + 1], y - 15 : y + 16, x - 15 : x + 16]
            repeated += np.array_equal(p1, p2)
        res.update(
            {
                f"{name}_{tag}": v
                for name, v in [
                    ("s", base),
                    ("min", min(options)),
                    ("max", max(options)),
                    ("dx_half_ratio", abs(L / H)),
                    ("exact_duplicate_pairs", repeated),
                    ("w_min", min(ws)),
                    ("w_max", max(ws)),
                    ("avg_vy", avg),
                    ("dt_avg", float(ts.mean())),
                ]
            }
        )
    return res


def run():
    with (OUT / "anti_eligible.csv").open() as f:
        rr = list(csv.DictReader(f))
    with ProcessPoolExecutor(4) as ex:
        rs = list(ex.map(audit, rr))
    for r in rs:
        print(
            r["id"],
            [
                round(r[k], 2)
                for k in [
                    "s_in",
                    "min_in",
                    "max_in",
                    "s_out",
                    "min_out",
                    "max_out",
                    "dx_half_ratio_in",
                    "dx_half_ratio_out",
                ]
            ],
        )
    with (OUT / "sign_sensitivity.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=rs[0])
        w.writeheader()
        w.writerows(rs)
