"""Soften one axis of the profile likelihood and transfer source-selected strength."""
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import t

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT / 'blind_decoder_v43'))
import inference43 as inference
from inference43 import r
import torch

ALPHAS = np.linspace(0, 1, 5)
CATEGORIES = ('base', 'hollow', 'bulge')
HEIGHTS = (19, 31)


def store(name, value):
    path = HERE / name
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def intervals(scores, multiplicity):
    scores = np.asarray(scores)
    mean = scores.mean(axis=0)
    half = t.ppf(1 - .025 / multiplicity, len(scores) - 1) * scores.std(axis=0, ddof=1) / np.sqrt(len(scores))
    return np.stack([mean, mean - half, mean + half], axis=-1).tolist()


def probabilities(bank, template, axis, xx, selected, paid, archive):
    # A paid probe has no profile factor. Preserve that same forecast at every alpha.
    white, norm, label = template
    full = np.repeat(archive[0:1], len(ALPHAS), axis=0)
    mask = selected == r.Q
    assert np.isfinite(paid[mask]).all() and np.isnan(paid[~mask]).all()
    indices = np.flatnonzero(mask.ravel())
    evidence = xx.reshape(-1, xx.shape[-1])
    measures = paid.reshape(-1, *paid.shape[-2:])
    coordinate = white @ axis
    for offset in range(0, len(indices), 96):
        take = indices[offset:offset + 96]
        prior = bank.fit(r.v38.t(evidence[take])).clamp_min(1e-300).log()[:, None, :]
        view = r.v38.t(measures[take]) / bank.sigma[r.Q]
        logit = prior + view @ white.T - norm
        penalty = .5 * coordinate.square() - (view @ axis)[..., None] * coordinate
        for j, alpha in enumerate(ALPHAS):
            beta = 1 - (1 - alpha)**2
            posterior = torch.softmax(logit + beta * penalty, dim=-1)
            full[j].reshape(-1, full.shape[-1])[take] = (posterior @ label).detach().cpu().numpy()
    err = np.max(np.abs(full[[0, -1]] - archive[:2]))
    assert err < 2e-9, err
    return full, float(err)


def score(version, bank, templates, axes):
    if version == 45:
        folder, families = 'image_pressure_v45', ('fold', 'ripple')
    else:
        folder, families = 'normal_variance_v46', ('terrace', 'dimple')
    source = PROJECT / folder
    out = []
    max_error = 0.
    cache_count = 0
    for catalog in range(12):
        sub = f'panel_{catalog:02d}'
        cache = {}
        family_rates = []
        for family in families:
            cell_risks = []
            for variation, height in itertools.product(CATEGORIES, HEIGHTS):
                geom = f'{family}_{variation}_h{height}.npz'
                if version == 45:
                    with np.load(source / 'camera45' / sub / geom) as data:
                        route = data['visual_project'].astype(bool)
                else:
                    with np.load(source / 'route46' / sub / geom) as data:
                        route = data['v45'].astype(bool)
                for surf, law in itertools.product(('inlay', 'woven'), ('rational', 'gaussian_rate')):
                    file = f'{family}_{surf}_{law}_{variation}_h{height}.npz'
                    with np.load(source / 'results' / sub / file) as data:
                        xx = data['xx']
                        paid = data['query_paid']
                        selected = data['selected']
                        archive = data['pred'][:2, 0]
                    assert archive.shape == (2, 4, 128, 4)
                    key = (height, hashlib.sha256(xx.tobytes()+selected.tobytes()+paid.tobytes()+archive[0].tobytes()).digest())
                    if key not in cache:
                        estimates, error = probabilities(bank, templates[height], axes[height], xx, selected, paid, archive)
                        cache[key] = estimates
                        max_error = max(max_error, error)
                        cache_count += 1
                    forecasts = cache[key]
                    error = float(abs(forecasts[[0, -1]] - archive).max())
                    max_error = max(max_error, error)
                    assert error < 2e-9, error
                    with np.load(source / f'labels{version}' / sub / file) as data:
                        labels = data['labels'][1:].astype(float)
                    mixed = np.where(route[None, ..., None], forecasts[:, None], archive[0][None, None])
                    brier = ((mixed[:, None] - labels[None, ..., None])**2).mean(axis=(1, 2, 3, 4, 5))
                    cell_risks.append(brier)
            family_rates.append(np.mean(cell_risks, axis=0))
        out.append(family_rates)
        print('finished', version, catalog, 'max_error', max_error, flush=True)
    return np.array(out), max_error, cache_count


def main():
    assert os.getenv('SLURM_JOB_ID') and torch.cuda.device_count() == 1
    torch.set_num_threads(1)
    start = time.monotonic()
    engine = inference.make_bank()
    templates = {h:inference.make_template(engine,h/1000) for h in HEIGHTS}
    vectors = json.loads((PROJECT/'image_pressure_v45/source_axis45.json').read_text())
    axes = {h:r.v38.t(vectors[str(h)]['axis']) for h in HEIGHTS}
    with torch.inference_mode():
        source, err, n = score(45, engine.inner, templates, axes)
        source_mean = source.mean((0,1))
        choice = int(np.argmin(source_mean))
        store('soft_choice.json', dict(alpha=float(ALPHAS[choice]), source_mean=source_mean.tolist(), per_catalog_family=source.tolist(),
              max_endpoint_error=err, unique_decodes=n, code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), closed_unix=time.time()))
        target, err, n = score(46, engine.inner, templates, axes)
    pooled = target.mean(1)
    deltas = pooled[:,choice,None] - pooled[:,[0, -1]]
    store('soft_transfer.json', dict(alpha=float(ALPHAS[choice]), grid=ALPHAS.tolist(), mean=pooled.mean(0).tolist(),
          selected_minus_full_and_hard=intervals(deltas, 2), family_means=target.mean(0).tolist(),
          source_families=['fold','ripple'], target_families=['terrace','dimple'], per_catalog_family=target.tolist(),
          max_endpoint_error=err, unique_decodes=n, elapsed_s=time.monotonic()-start,
          gpu=torch.cuda.get_device_name(), job=os.environ['SLURM_JOB_ID']))


if __name__ == '__main__':
    main()
