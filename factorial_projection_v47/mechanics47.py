"""Outcome-only centered finite-pressure solver; consumes no optical alignment or images."""
import argparse
from itertools import product
import os
from pathlib import Path
import sys
import numpy as np
from world47 import (ROOT, FAMILIES, HEIGHTS, VARIANTS, COUPLINGS, RADII, OFFSETS, loaded_shape)
sys.path.insert(0, str(ROOT.parent / 'image_pressure_v45'))
import pressure45 as pressure

AXES = ("family", "height", "variant", "coupling", "radius", "offset", "environment", "object")


def panel_outcomes(panel, fine):
    path = ROOT / ("labels_fine" if fine else "labels") / f"panel_{panel:02d}.npz"
    if path.exists():
        return
    with np.load(ROOT / "worlds" / f"panel_{panel:02d}.npz") as raw:
        theta = raw["theta"]
        signs, regimes, surfaces, laws = [raw[x] for x in ("signs", "regimes", "surface", "law")]
        object_index = raw["object_index"]
    assert theta.shape == (4, 128, 6)
    E, N, _ = theta.shape
    t = theta.reshape(-1, 6)
    speed = np.empty((len(FAMILIES), len(HEIGHTS), len(VARIANTS), len(COUPLINGS),
                      len(RADII), len(OFFSETS), E, N), np.float64)
    nx, nt = (24, 12) if fine else (12, 6)
    for jh, (ja, r), (jx, offset), (jc, coupling) in product(
            range(len(HEIGHTS)), enumerate(RADII), enumerate(OFFSETS), enumerate(COUPLINGS)):
        height = HEIGHTS[jh]
        center = height + offset
        dx, w = pressure.quadrature(r, nx, nt)
        h = center + dx
        for jf, family in enumerate(FAMILIES):
            for jv, variant in enumerate(VARIANTS):
                if coupling == "null" and (jf, jv) != (0, 0):
                    # All unloaded wings lie strictly outside this patch. Verify before reuse.
                    for at in (center, h):
                        np.testing.assert_array_equal(loaded_shape(at, height, family, variant, coupling)[1],
                                                      loaded_shape(at, height, FAMILIES[0], VARIANTS[0], coupling)[1])
                    speed[jf, jh, jv, jc, ja, jx] = speed[0, jh, 0, jc, ja, jx]
                    continue
                g0 = t[:, 2] + t[:, 3]*center/25. + loaded_shape(center, height, family, variant, coupling)[1]
                g = t[:, 2, None] + t[:, 3, None]*h/25. + loaded_shape(h, height, family, variant, coupling)[1]
                drive, resistance, normal, vt = pressure.impulse_geometry(g, g0, t[:, 5], w)
                assert np.all(normal > 0)
                for e, (sf, law) in enumerate(zip(surfaces, laws)):
                    sl = slice(N*e, N*(e+1))
                    mu = pressure.material(t[sl, 4], vt[sl], h, sf, law,
                                           np.repeat(signs[e], N), np.repeat(regimes[e], N))
                    speed[jf, jh, jv, jc, ja, jx, e] = pressure.residual(drive[sl], resistance[sl], mu)
    assert np.isfinite(speed).all() and speed.min() >= 0
    path.parent.mkdir(exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, labels=speed <= pressure.THRESHOLD, slip_speed=speed.astype(np.float32),
                        axes=AXES, families=FAMILIES, heights_mm=HEIGHTS, variants=VARIANTS,
                        couplings=COUPLINGS, radii_mm=RADII, offsets_mm=OFFSETS,
                        surfaces=surfaces, laws=laws, object_index=object_index, nx=nx, nt=nt)
    os.replace(tmp, path)
    print(path.name, "computed cases", speed.size, "fine", fine, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--panels", nargs="+", type=int, required=True)
    parser.add_argument("--fine", action="store_true")
    args = parser.parse_args()
    if not os.getenv("SLURM_JOB_ID") or os.getenv("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Use an allocated CPU step with CUDA hidden")
    if not (ROOT/"TARGET_GO").exists():
        raise RuntimeError("Frozen policy and study release required")
    for index in args.panels:
        panel_outcomes(index, args.fine)
