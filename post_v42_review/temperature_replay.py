"""Exploratory temperature baseline. Fit v41 only, score all v42 noise draws."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, logit
from scipy.stats import t

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'estimated_projection_v41' / 'results'
TARGET = ROOT.parent / 'height_offset_v42' / 'results'
SOURCE_FILES = sorted(SOURCE.glob('*_*/catalog_*/block_*.npz'))
TARGET_FILES = sorted(TARGET.glob('*_*/catalog_*.npz'))
assert (len(SOURCE_FILES), len(TARGET_FILES)) == (576, 288)
NAMES = ('original', 'oracle', 'estimated')


def logits(p):
    return logit(np.clip(p, 1e-9, 1 - 1e-9))


def fit():
    xs, ys = [], []
    for file in SOURCE_FILES:
        with np.load(file) as data:
            # One observation-noise draw per object; all starts, panels and regimes.
            xs.append(logits(data['p_base'][..., 0]).reshape(3, -1))
            ys.append(data['labels'].mean(axis=1).ravel())
    labels = np.concatenate(ys)
    temps = []
    for x in np.concatenate(xs, axis=1):
        def gradient(inv_temp):
            return np.mean((expit(inv_temp * x) - labels) * x)
        temps.append(1 / brentq(gradient, 0.02, 50))
    return np.array(temps)


def interval(x):
    mean = x.mean(axis=0)
    half_width = t.ppf(.975, 11) * x.std(axis=0, ddof=1) / np.sqrt(12)
    return np.stack((mean, mean - half_width, mean + half_width), axis=-1).tolist()


def main():
    temp = fit()
    sums = np.zeros((12, 2, 3, 2))
    count = np.zeros(12, dtype=int)
    for file in TARGET_FILES:
        panel = int(file.stem.split('_')[-1])
        with np.load(file) as data:
            base = data['p_base']
            np.testing.assert_array_equal(data['decoders'], NAMES)
            pred = expit(logits(base) / temp[:, None, None, None, None])
            y = data['labels'].mean(axis=2)
            groups = [y[data['offsets_mm'] == 0], y[data['offsets_mm'] != 0]]
            for k, group in enumerate(groups):
                truth = group.mean(axis=0)[..., None]
                for c, p in enumerate((base, pred)):
                    # For a binary event Y, E[(p-Y)^2] = p^2 - 2pE[Y] + E[Y].
                    sums[panel, k, :, c] += (p*p - 2*p*truth + truth).mean(axis=(1, 2, 3, 4))
        count[panel] += 1
    assert np.all(count == 24)
    panel_loss = sums / 24
    ref = json.loads((TARGET.parent / 'stats42.json').read_text())
    prior_loss = np.asarray(ref['per_panel_loss'])
    shifted = np.array(ref['offset_mm']) != 0
    raw_diff = panel_loss[:, 1, :, 0] - prior_loss[:, shifted].mean(axis=1)[:, :, 0]
    assert np.abs(raw_diff).max() < 1e-9
    opt = prior_loss[:, shifted].mean(axis=1)[:, :, 2]
    summary = dict(decoders=NAMES, temperatures=temp.tolist(), strata=['nominal','nonzero'],
                   policies=['zero','temperature'], loss=panel_loss.mean(axis=0).tolist(),
                   temperature_minus_zero=interval(panel_loss[...,1]-panel_loss[...,0]),
                   optimizer_minus_temperature_nonzero=interval(opt-panel_loss[:,1,:,1]),
                   max_baseline_error=float(np.abs(raw_diff).max()))
    (ROOT/'temperature_replay.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(summary)


if __name__ == '__main__':
    main()
