"""Paired held-out Brier analysis. Each random catalog contributes one observation."""
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parent
OBJECTS = ('fold', 'ripple')
SURFACES = ('inlay', 'woven')
LAWS = ('rational', 'gaussian_rate')
HEIGHTS = ('h19', 'h31')
CONTOURS = ('base', 'hollow', 'bulge')
OFFSETS = np.array([-1., -.5, -.25, -.125, 0., .125, .25, .5, 1.])
NONZERO = OFFSETS != 0
DECODERS = ('original', 'robust', 'privileged')
POLICIES = ('zero', 'temperature', 'fixed', 'optimizer')
PANELS = 12


def interval(samples, multiplicity=1):
    samples = np.asarray(samples)
    n = len(samples)
    center = samples.mean(axis=0)
    radius = t.ppf(1 - .025 / multiplicity, n - 1) * samples.std(axis=0, ddof=1) / np.sqrt(n)
    return np.stack((center, center - radius, center + radius), axis=-1)


def comparisons(panel):
    """Input [catalog, signed offset, decoder, correction]."""
    nominal = panel[:, 4]
    shifted = panel[:, NONZERO].mean(axis=1)
    opt = shifted[:, 1, 3] - shifted[:, 1, 1]
    geo0 = nominal[:, 2, 0] - nominal[:, 0, 0]
    geo1 = shifted[:, 2, 0] - shifted[:, 0, 0]
    return np.stack((opt, geo0, geo1, geo1 - geo0), axis=-1)


def read_risk(path):
    with np.load(path) as data:
        y = data['labels']
        p = data['pred'].astype(float)
        assert y.shape == (9, 4, 128) and p.shape == (3, 4, 4, 128, 4), path
        assert np.isin(y, [0, 1]).all() and np.isfinite(p).all()
        assert p.min() >= -1e-12 and p.max() <= 1 + 1e-12
        y = y.astype(float)
        loss = ((p[None] - y[:, None, None, :, :, None]) ** 2).mean(axis=(3, 4, 5))
        if 'risk' in data:
            by_env = ((p[None] - y[:, None, None, :, :, None]) ** 2).mean(axis=(4, 5))
            np.testing.assert_allclose(data['risk'], by_env, rtol=0, atol=2e-12)
        means = p.mean(axis=-1).reshape(3, 4, -1)
        squared = (p ** 2).mean(axis=(2, 3, 4))
        flat_y = y.reshape(9, -1)
        independent = (squared[None] + flat_y.mean(axis=1)[:, None, None]
                       - 2 * np.einsum('sn,dcn->sdc', flat_y, means) / flat_y.shape[1])
        np.testing.assert_allclose(loss, independent, rtol=0, atol=2e-12)
        p0, p1 = p[0, 0], p[2, 0]
        proj_move = (p1 - p0).mean(axis=-1)
        # Symmetric control for moving farther from, or closer to, the observed target.
        step = p[1, 3] - p[1, 1]
        y_shift = y[NONZERO].mean(axis=0)[..., None]
        step_cost = np.mean(step ** 2)
        alignment = -2 * np.mean(step * (y_shift - p[1, 1]))
        assert abs(step_cost + alignment - (loss[NONZERO, 1, 3]-loss[NONZERO, 1, 1]).mean()) < 1e-12
        return y, loss, proj_move, (step_cost, alignment)


def analyze():
    dims = (PANELS, 2, 2, 2, 2, 3)
    risk = np.empty(dims + (9, 3, 4))
    geometry = np.empty(dims + (9, 3))
    terms = np.empty(dims + (2,))
    for panel, i, j, k, h in itertools.product(range(PANELS), *[range(2)] * 4):
        prefix = '_'.join((OBJECTS[i], SURFACES[j], LAWS[k]))
        for v, var in enumerate(CONTOURS):
            path = ROOT/'results'/f'panel_{panel:02d}'/f'{prefix}_{var}_{HEIGHTS[h]}.npz'
            y, loss, move, correction = read_risk(path)
            if v == 0:
                base = y.copy()
            np.testing.assert_array_equal(y[4], base[4])
            idx = panel, i, j, k, h, v
            risk[idx] = loss
            terms[idx] = correction
            flip = np.abs(y-base).mean(axis=(1,2))
            shape = -2 * ((y-base) * move[None]).mean(axis=(1,2))
            base_shift = -2 * ((base-base[4]) * move[None]).mean(axis=(1,2))
            diff = loss[:,2,0] - loss[:,0,0]
            np.testing.assert_allclose(diff-diff[4], shape+base_shift, rtol=0, atol=2e-12)
            geometry[idx] = np.stack((flip, shape, base_shift), axis=-1)
    panel = risk.mean(axis=(1, 2, 3, 4, 5))
    groups = risk.mean(axis=(4, 5))
    primary = comparisons(panel)
    leave = {name: {} for name in ('object','surface','law')}
    for axis, labels in enumerate((OBJECTS, SURFACES, LAWS)):
        for pos, name in enumerate(labels):
            retained = np.take(groups, [1-pos], axis=axis+1).mean(axis=(1, 2, 3))
            leave[('object','surface','law')[axis]][name] = comparisons(retained).mean(axis=0)
    shifted = panel[:, NONZERO].mean(axis=1)
    strata = np.stack((panel[:,4], shifted), axis=1)
    by_group = {}
    for i,j,k in itertools.product(range(2), repeat=3):
        key = '/'.join((OBJECTS[i], SURFACES[j], LAWS[k]))
        by_group[key] = interval(comparisons(groups[:,i,j,k]), 1)
    return dict(primary=interval(primary,4), primary_panel=primary, loss=interval(strata),
                wins=(primary<0).sum(axis=0), group=by_group, leave_one_family_out=leave,
                opt_temp_all=interval(shifted[:,:,3]-shifted[:,:,1]),
                opt_zero_all=interval(shifted[:,:,3]-shifted[:,:,0]),
                opt_fixed_all=interval(shifted[:,:,3]-shifted[:,:,2]),
                robust_original=interval(strata[:,:,1]-strata[:,:,0]),
                geometry_by_offset=interval(panel[:,:,2,0]-panel[:,:,0,0],9),
                temp_geometry=interval(strata[:,:,2,1]-strata[:,:,0,1],2),
                nonzero_step_alignment=interval(terms.mean(axis=(1,2,3,4,5))),
                contour_flip=interval(geometry[:,:,:,:,:,1:,:,0].mean(axis=(1,2,3,4,5)),9),
                geometry_mechanism=interval(geometry.mean(axis=(1,2,3,4,5))[:,:,1:]),
                offsets_mm=OFFSETS, decoders=DECODERS, policies=POLICIES,
                panel_risk=panel, archive_count=int(np.prod(dims)),
                max_direct_check_tolerance=2e-12)


def timing():
    all_times = sorted((ROOT/'results'/'timing').glob('panel_*.npz'))
    if not all_times:
        all_times = sorted((ROOT/'timing').glob('panel_*.npz'))
    assert len(all_times)==PANELS
    raw = []
    for path in all_times:
        with np.load(path) as data:
            value = data['warm_ms']
            assert value.shape == (3,12) and (value > 0).all()
            np.testing.assert_array_equal(np.sort(data['order'], axis=1), np.tile(np.arange(12), (3,1)))
            raw.append(value)
    x = np.stack(raw)
    ids = np.random.default_rng(430081).integers(0,len(x),size=(20000,len(x)))
    q = np.quantile(x,.9,axis=(0,1))
    boot = np.quantile(x[ids],.9,axis=(1,2))
    ratio = boot[:,7]/boot[:,5]
    return dict(p90=q.reshape(3,4), ci=np.quantile(boot,[.025,.975],axis=0).T.reshape(3,4,2),
                ratio=[q[7]/q[5],*np.quantile(ratio,[.025,.975])], raw_ms=x,
                order_assumption='decoder major; warm_ms indexed by method')


if __name__=='__main__':
    results = analyze()
    results['timing'] = timing()
    (ROOT/'stats43.json').write_text(json.dumps(results,indent=2,allow_nan=False,
        default=lambda x: x.tolist())+'\n')
