"""Separate paired comparison of continuous policies using signs or amplitudes."""
import json
import os
import time

import numpy as np
import torch
from bench53 import ROOT, actions, coefficients, engine50, fixture, load_model, r, routed_predictions
from policy53 import mixed53


def vote_weight(raw):
    ow, of, nw, nf, sign, _ = raw
    return mixed53(ow, of, nw[:, :3], nf[:, :3], nw[:, 3], sign, method='vote')['alpha_joint']


def main():
    assert os.getenv('SLURM_JOB_ID') and torch.cuda.device_count() == 1
    torch.set_num_threads(1)
    model = load_model()
    clock = np.empty((12, 4, 2))
    order = np.empty((12, 4, 2), dtype=int)
    with torch.inference_mode():
        engine, templates = engine50()
        signed, unsigned = coefficients()
        for k in range(12):
            xx,bought,disp,h,raw,selected,expected,_,alpha,pre,post,err,rep = fixture(k, model)
            weights = vote_weight(raw)
            with np.load(ROOT / 'votes53' / f'panel_{k:02d}.npz') as dat:
                # At kappa=.5 either event induces the same evidence by construction.
                event = ((np.arange(4) + k) % 2).astype(int)
                saved = dat['alpha'][err, event, pre, np.arange(4), rep]
                np.testing.assert_allclose(saved, weights, atol=1e-12, rtol=0)
            rng = np.random.default_rng(536027 + k)
            one = np.stack([rng.permutation(2) for _ in range(2)])
            order[k] = np.stack((one[0], one[0, ::-1], one[1], one[1, ::-1]))
            for j, seq in enumerate([rng.permutation(2), *order[k]], start=-1):
                for m in seq:
                    torch.cuda.synchronize()
                    start = time.perf_counter_ns()
                    if m == 0:
                        ac, weight = actions('impact_joint', raw, model)
                    else:
                        weight = vote_weight(raw)
                        ac = np.zeros_like(weight, dtype=np.int8)
                    obs = r.infer.actions43(xx, engine)
                    paid = r.infer.paid_sensors(bought, obs)
                    p0 = r.infer.decode43(engine, templates[h], xx, obs, paid)[..., 0]
                    arms = (obs == r.r.Q).astype(np.intp)
                    pred = routed_predictions(p0, disp, arms, signed[1, h], unsigned[h], ac, weight)
                    torch.cuda.synchronize()
                    if j >= 0:
                        clock[k,j,m] = (time.perf_counter_ns() - start)*1e-6
                    np.testing.assert_array_equal(obs, selected)
                    np.testing.assert_allclose(weight, alpha[0] if m == 0 else saved, atol=1e-12, rtol=0)
                    if m == 0:
                        np.testing.assert_allclose(pred, expected[0], atol=2e-9, rtol=0)
            print('finished', k, flush=True)
    p90 = np.quantile(clock,.9,axis=(0,1))
    b = np.random.default_rng(5308821).integers(12,size=(16000,12))
    boot = np.quantile(clock[b],.9,axis=(1,2))
    data = dict(methods=['continuous likelihood','continuous sign count'],p90_ms=p90.tolist(),
                ci95_diff=np.quantile(boot[:,0]-boot[:,1],[.025,.975]).tolist(),diff_ms=p90[0]-p90[1],
                samples_per_method=48,gpu=torch.cuda.get_device_name(),job=os.getenv('SLURM_JOB_ID'))
    (ROOT/'vote_timing53.json').write_text(json.dumps(data,indent=2))
    np.savez_compressed(ROOT/'vote_timing53.npz',ms=clock,order=order)
    print(data, flush=True)


if __name__ == '__main__':
    main()
