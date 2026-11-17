"""Source-only affine baselines and an explicit two-regime comparator for the Bristol data."""
from pathlib import Path
import argparse
import csv
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis"


def affine(x):
    return np.column_stack([np.ones(len(x)), x])


def fit_regimes(x, y):
    """Ridge logistic release-sign gate and two affine experts. All coefficients from source fit."""
    side = y[:, 0] + y[:, 2] > 0
    scales = x.std(0)
    center = x.mean(0)
    a = affine((x - center) / scales)
    beta = np.zeros(a.shape[1])
    penalty = np.diag([0, 1, 1, 1])
    for _ in range(35):
        p = 1 / (1 + np.exp(-np.clip(a @ beta, -35, 35)))
        hess = a.T @ ((p * (1 - p))[:, None] * a) + penalty
        step = np.linalg.solve(hess, a.T @ (p - side) + penalty @ beta)
        beta -= step
        if np.linalg.norm(step) < 1e-9:
            break
    experts = [np.linalg.lstsq(a[side == b], y[side == b], rcond=None)[0] for b in (False, True)]
    def predict(values):
        design = affine((values - center) / scales)
        branch = design @ beta > 0
        return np.where(branch[:, None], design @ experts[1], design @ experts[0])
    return predict


def evaluate(radius):
    data = {c: np.loadtxt(ROOT / "data69" / "ESM_2" / f"data_campaign{c}.csv", delimiter=",") * radius for c in "AB"}
    rows = []
    for src, tgt in (("A", "B"), ("B", "A")):
        fit = (np.arange(len(data[src])) // 24) % 4 != 1
        x, y = data[src][:, :3], data[src][:, 3:]
        tx, ty = data[tgt][:, :3], data[tgt][:, 3:]
        yin = tx[:, 0] + tx[:, 2]
        truth = ty[:, 0] + ty[:, 2]
        coeff = np.linalg.lstsq(affine(x[fit]), y[fit], rcond=None)[0]
        truths_cal = y[~fit, 0] + y[~fit, 2]
        for model, pred in [('affine', lambda v: affine(v) @ coeff),
                            ('two_regime', fit_regimes(x[fit], y[fit])),
                            ('source_frequency', lambda v: np.tile(y[~fit].mean(0), (len(v), 1)))]:
            cal = pred(x[~fit])
            err = truths_cal - cal[:, 0] - cal[:, 2]
            out = pred(tx)
            u = out[:, 0] + out[:, 2]
            samples = u[:, None] + err
            norm = len(err) + 1
            p = {str(t): ((abs(samples) <= t).sum(1) + .5) / norm for t in (.25, .5, 1.)}
            p_reverse = ((samples * yin[:, None] < 0).sum(1) + .5) / norm
            for i in range(len(tx)):
                rows.append(dict(source=src,target=tgt,row=i,model=model,
                    true_vx=ty[i,0],pred_vx=out[i,0],true_u=truth[i],pred_u=u[i],in_u=yin[i],
                    p025=p["0.25"][i],p050=p["0.5"][i],p100=p["1.0"][i],p_rev=p_reverse[i]))
    with (OUT / "empirical_predictions.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for target in "BA":
        for name in ("affine", "two_regime", "source_frequency"):
            rr = [r for r in rows if r["target"] == target and r["model"] == name]
            def column(k):
                return np.array([r[k] for r in rr])
            u, ui = column("true_u"), column("in_u")
            summary.append(dict(target=target,model=name,n=len(u),com_mae=np.mean(abs(column("true_vx") - column("pred_vx"))),
                                u_mae=np.mean(abs(u-column("pred_u"))),brier=np.mean((column('p050')-(abs(u)<=.5))**2),
                                rev_brier=np.mean((column('p_rev')-(ui*u<0))**2),near_count=int(sum(abs(u)<=.5)),
                                rev_count=int(sum(u*ui<0))))
    (OUT / "empirical_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--radius", type=float, required=True)
    evaluate(parser.parse_args().radius)
