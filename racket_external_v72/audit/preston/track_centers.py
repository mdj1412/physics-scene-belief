import cv2
import numpy as np


from paths import VIDEO, CACHE


def track(a, b, name):
    c = cv2.VideoCapture(str(VIDEO))
    c.set(cv2.CAP_PROP_POS_FRAMES, a)
    Fs = []
    for i in range(a, b):
        ok, f = c.read()
        if not ok:
            break
        Fs.append(cv2.cvtColor(f[52:235, 32:301], cv2.COLOR_BGR2GRAY))
    c.release()
    Fs = np.array(Fs)
    bg = np.median(Fs[::3], axis=0).astype(np.uint8)
    # foreground centroid by local ball energy, include blade region up to x245 (global x277)
    radi = 12
    yy, xx = np.mgrid[-radi : radi + 1, -radi : radi + 1]
    disk = (xx**2 + yy**2 <= radi**2).astype("float32")
    disk /= disk.sum()
    rec = []
    for i, F in enumerate(Fs):
        delta = cv2.absdiff(F, bg)
        energy = cv2.filter2D(delta.astype("float32"), -1, disk)
        energy[:3] = 0
        energy[-3:] = 0
        energy[:, 244:] = 0
        energy[:, :2] = 0
        y, x = np.unravel_index(energy.argmax(), energy.shape)
        val = energy[y, x]
        X0 = max(0, x - 15)
        X1 = min(269, x + 16)
        Y0 = max(0, y - 15)
        Y1 = min(183, y + 16)
        yy, xx = np.mgrid[Y0:Y1, X0:X1]
        weights = np.minimum(
            np.maximum(delta[Y0:Y1, X0:X1].astype(float) - 12, 0), 60
        ) * ((xx - x) ** 2 + (yy - y) ** 2 <= 15**2)
        if weights.sum() > 0:
            x = float((xx * weights).sum() / weights.sum())
            y = float((yy * weights).sum() / weights.sum())
        rec.append([a + i, x + 32, y + 52, val])

    np.savetxt(
        CACHE / (name + ".csv"),
        rec,
        delimiter=",",
        header="frame,x,y,energy",
        comments="",
    )
