import cv2
import numpy as np
import csv

from scipy.ndimage import gaussian_filter1d


from concurrent.futures import ProcessPoolExecutor
from paths import CACHE as P, OUTPUT as OUT, VIDEO

ANG = 240
angles = np.arange(ANG) * 2 * np.pi / ANG


def rem(im, xx, yy):
    return cv2.remap(
        im,
        np.array(xx, dtype="float32"),
        np.array(yy, dtype="float32"),
        cv2.INTER_LINEAR,
    ).astype(float)


def xy(x, y, rs):
    return (x + np.outer(rs, np.cos(angles)), y + np.outer(rs, np.sin(angles)))


def edge(im, bg, x, y):
    rs = np.arange(5, 19, 0.5)
    m = rem(cv2.absdiff(im, bg), *xy(x, y, rs))
    # First transition inward from outside where foreground significantly different.
    m = gaussian_filter1d(m, 1, axis=0)
    rr = np.nanmax(np.where(m > 20, rs[:, None], 0), axis=0)
    mask = (rr > 7) & (rr < 17.5)
    if mask.sum() < 60:
        return x, y, 12.0
    disp = np.linalg.lstsq(
        np.stack([np.cos(angles[mask]), np.sin(angles[mask]), np.ones(mask.sum())], 1),
        rr[mask],
        rcond=None,
    )[0]
    return x + np.clip(disp[0], -3, 3), y + np.clip(disp[1], -3, 3), disp[2]


def omega(Fs, rr, fracs):
    # normalized concentric ring texture. Image-coordinate angle increases clockwise.
    tt = []
    for im, (x, y, r) in zip(Fs, rr):
        rings = rem(im, *xy(x, y, r * np.array(fracs)))
        rings -= rings.mean(-1, keepdims=True)
        rings /= np.maximum(np.sqrt(np.mean(rings**2, -1, keepdims=True)), 5)
        tt.append(rings)
    ff = np.fft.rfft(tt, axis=-1)
    lags = [1, 2, 3, 5]
    omgrid = np.arange(-1.55, 1.551, 0.005)
    score = 0.0
    for lag in lags:
        cor = np.fft.irfft((ff[lag:] * ff[:-lag].conj()).mean((0, 1)), n=ANG) / ANG
        score += np.interp(
            omgrid * ANG / (2 * np.pi) * lag % ANG,
            np.arange(ANG + 1),
            np.r_[cor, cor[0]],
        )
    score /= len(lags)
    return float(omgrid[np.argmax(score)]), float(np.max(score))


def one(j):
    f = P / ("r%02d.csv" % j)
    d = np.loadtxt(f, delimiter=",", skiprows=1)
    first = int(d[0, 0])
    c = cv2.VideoCapture(str(VIDEO))
    c.set(cv2.CAP_PROP_POS_FRAMES, first)
    ims = []
    for _ in d:
        ok, fr = c.read()
        ims.append(cv2.cvtColor(fr[:240, :315], cv2.COLOR_BGR2GRAY))
    c.release()
    bg = np.median(ims[::3], axis=0).astype(np.uint8)
    # Position where measured ball approaches face most closely
    cand = np.where((d[:, 3] > 25) & (d[:, 1] > 231))[0]
    if not len(cand):
        return {"id": j, "reason": "no_face"}
    # use the contiguous contact run with most physical evidence
    runs = np.split(cand, np.where(np.diff(cand) > 1)[0] + 1)
    run = max(runs, key=len)
    k = int(run[np.argmax(d[run, 1])])
    res = {"id": j, "first": first, "contact": first + k}
    both = []
    for tag, di in [("in", -1), ("out", 1)]:
        inds = []
        seen = False
        for step in range(1, 80):
            i = k + di * step
            if i < 0 or i >= len(d):
                break
            x, y, e = d[i, 1:]
            ok = 55 < x < d[k, 1] - 8 and 66 < y < 221 and e >= 25
            if not ok:
                if seen:
                    break
                continue
            inds.append(i)
            seen = True
            if len(inds) >= 24:
                break
        # Before/after near-contact free flight only. Table contact creates a sharp y-velocity change.
        inds = np.array(sorted(inds), dtype=int)
        if len(inds) < 9:
            res["reason"] = tag + "_window"
            return res
        R = np.array([edge(ims[i], bg, *d[i, 1:3]) for i in inds])
        xs = R[:, 0]
        ys = R[:, 1]
        ts = inds - k
        # Shared quadratic vertical ballistic slope unknown in edited source, x nearly linear.
        px = np.polyfit(ts, xs, 1)
        py = np.polyfit(ts, ys, 2)
        rms = float(
            np.sqrt(
                np.mean((xs - np.polyval(px, ts)) ** 2 + (ys - np.polyval(py, ts)) ** 2)
            )
        )
        R[:, 0] = np.polyval(px, ts)
        R[:, 1] = np.polyval(py, ts)
        R[:, 2] = np.median(R[:, 2])
        o1, cor1 = omega([ims[i] for i in inds], R, [0.30, 0.45, 0.55, 0.65])
        o2, cor2 = omega([ims[i] for i in inds], R, [0.55, 0.65, 0.75, 0.83])
        rad = float(np.median(R[:, 2]))
        vx = py[1]
        vz = -px[0]
        om = -(o1 + o2) / 2
        res.update(
            {
                f"{key}_{tag}": val
                for key, val in [
                    ("vx", vx),
                    ("vz", vz),
                    ("omega", om),
                    ("r", rad),
                    ("corr", min(cor1, cor2)),
                    ("drate", abs(o1 - o2)),
                    ("rms", rms),
                    ("n", len(inds)),
                    ("win_start", first + inds.min()),
                    ("win_end", first + inds.max()),
                    ("accel", 2 * py[0]),
                ]
            }
        )
        both.append(rad)
    res["radius_px"] = float(np.mean(both))
    bad = [
        tag
        for tag in ["in", "out"]
        if res["rms_" + tag] > 2
        or res["drate_" + tag] > 0.10001
        or res["corr_" + tag] < 0.45
        or (res["vz_" + tag] >= 0 if tag == "in" else res["vz_" + tag] <= 0)
    ]
    res["reason"] = "OK" if not bad else "quality_" + "+".join(bad)
    return res


def run():
    with ProcessPoolExecutor(4) as ex:
        rows = list(ex.map(one, range(1, 25)))
    for r in rows:
        print(
            r["id"],
            r["reason"],
            *[
                round(r.get(k, -9), 2)
                for k in [
                    "omega_in",
                    "omega_out",
                    "rms_in",
                    "rms_out",
                    "drate_in",
                    "drate_out",
                    "corr_in",
                    "corr_out",
                    "radius_px",
                    "accel_in",
                    "accel_out",
                ]
            ],
            flush=True,
        )
    fields = ["id", "reason"] + sorted(
        set().union(*[set(r) for r in rows]) - {"id", "reason"}
    )
    for fn, rr in [
        ("all_tracks.csv", rows),
        ("anti_eligible.csv", [r for r in rows if r["reason"] == "OK"]),
    ]:
        with (OUT / fn).open("w") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rr)
