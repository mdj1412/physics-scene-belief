"""Charge each policy its own measured cohort setup over the deployment horizon."""
import json
from pathlib import Path

import numpy as np

from quota_latency29 import bootstrap_ratios

results = Path(__file__).resolve().parent / "results"
with np.load(results / "quota_latency29.npz") as data:
    environments = data["env_indices"]
    batch = data["batch_times_ms"].copy()
    stream = data["stream_times_ms"].copy()

setup = []
for block, ids in enumerate(environments):
    record = json.loads((results / f"block_{block:03d}" / "metrics.json").read_text())
    timings = record["setup_ms"]
    setup.append(sum(np.array(timings[key])[ids, :2] for key in ("fit", "score", "engines")))
setup = np.stack(setup)
charged = {
    "batch": batch + setup[:, :, None, :],
    "stream": stream + setup[:, :, None, None, :] / 128,
}
summary = {}
for kind, samples in charged.items():
    p90 = np.quantile(samples.reshape(-1, 2), 0.9, axis=0)
    summary[kind] = dict(p90_ms=p90.tolist(), ratio=p90[0] / p90[1],
                         ratio_ci95=bootstrap_ratios(samples, 29052412))
(results / "cost_with_setup.json").write_text(json.dumps(summary, indent=2))
