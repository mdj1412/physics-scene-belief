"""Independent split-half diagnostic on historical training targets, conditional on the original actions."""
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "adaptive_stopping_v29" / "results"

def conditional_half_audit():
    rows = []
    noise_rows = []
    for path in sorted(SOURCE.glob("block_*/metrics.npz")):
        with np.load(path) as z:
            gaps = z["audit_gap"]
            cost = z["cost"][:, :, -1, :]
            E, W, B = gaps.shape
            assert (E, W, B) == (8, 128, 3)
            probe = np.zeros_like(gaps, dtype=bool)
            order = np.argsort(-gaps, kind="stable", axis=1)
            np.put_along_axis(probe, order[:, :13], True, axis=1)
            risks = np.where(probe, cost[:, :, 1, None], cost[:, :, 0, None])
            gains = risks[:, :, :-1] - risks[:, :, 1:]
            # IDs refer to independent random test worlds. Both halves use the exact
            # same sealed decisions over the full 128-world batch (no re-ranking).
            halves = np.stack([gains[:, ::2].mean(1), gains[:, 1::2].mean(1)], axis=-1)
            rows.append(halves)
            # Conditional on the realized worlds, labels, actions and calibration future,
            # the 32 simulator repetitions isolate the action-observation noise component.
            pred = z["probabilities"][:, :, -1]
            label = z["label"][:, :, None, None]
            by_draw = (pred - label)**2
            chosen = np.where(probe[:, :, :, None], by_draw[:, :, 1, None, :],
                              by_draw[:, :, 0, None, :])
            delta_noise = (chosen[:, :, :-1, :] - chosen[:, :, 1:, :]).mean(axis=1)
            assert np.allclose(delta_noise.mean(-1), gains.mean(1), atol=1e-12)
            noise_rows.append(delta_noise.var(-1, ddof=1) / delta_noise.shape[-1])
    assert len(rows) == 24
    a = np.stack(rows)
    rng = np.random.default_rng(306601)
    draws = rng.integers(24, size=(3000, 24))
    result = {}
    for s, key in enumerate(("2to4", "4to8")):
        half = a[:, :, s, :]
        def rho(h):
            h = h.reshape(-1, 2)
            return float(spearmanr(h[:, 0], h[:, 1]).statistic)
        q = np.quantile([rho(half[ids]) for ids in draws], [.025, .975])
        d = half.reshape(-1, 2)
        result[key] = dict(split_half_spearman=rho(half), interval=q.tolist(),
                           mean_abs_half_gap=float(np.abs(d[:, 0]-d[:, 1]).mean()),
                           conditional_sensor_se_rms=float(np.sqrt(np.stack(noise_rows)[:, :, s].mean())),
                           signed_target_sd=float(d.mean(1).std(ddof=1)), cohorts=len(d))
    (ROOT / "source_target_noise.json").write_text(json.dumps(result, indent=2))
    print(result)

if __name__ == "__main__":
    conditional_half_audit()
