"""Paired evaluation of frozen correction policies and actual end-to-end inference."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT / 'probe_speed_v37'))
import run37 as historical
from gate36 import observations as features, log_density, posterior, expected_gain
from gate37 import load_artifacts, predict as frozen_predict
from run36 import (Engine, Detector, PosteriorPredictiveEngine, batch_gap, v33, t, torch,
                   observations, atomic_json, atomic_npz, digest, apply, Q, P)
import physics38 as world

CELLS = ('shoulder_weakening', 'shoulder_strengthening', 'bead_weakening', 'bead_strengthening')
REGIMES = ('strong', 'weak', 'null')
METHODS = ('fixed', 'binary', 'ternary', 'continuous')
SPEED, HEIGHT, SHRINK, BETA = 1.10, .023, .325, historical.BETA
ALPHAS = np.asarray(json.loads((PROJECT/'bidirectional_probe_v34/policy_design34.json').read_text())['alphas'])
BLOCKS = 8
BASE = 38000000
OUT = ROOT / 'results'
TIMED = (('strong_plus','weak_minus','null_plus','strong_minus'),
         ('weak_plus','null_minus','strong_plus','weak_minus'))


def policy(artifact, records, method, use_fourth):
    x, y, dirs = observations(records)
    ll = np.zeros((len(x), len(artifact['sign'])), dtype=np.float64)
    for j in range(3):
        obs = features(x[:, j], y[:, j])
        for action, sign in enumerate((-1, 1)):
            selected = dirs[:, j] == sign
            if selected.any():
                ll[selected] += log_density(artifact, obs[selected], action)
    w = posterior(ll)
    z = w @ artifact['far_sign']
    if method == 'fixed':
        return z, np.full(z.shape, SHRINK)
    if use_fourth:
        w = posterior(ll + log_density(artifact, features(x[:, 3], y[:, 3]), 2))
    if method == 'continuous':
        return z, w @ artifact['near_strength']
    full = expected_gain(artifact, w, z)
    if method == 'binary':
        return z, (full > 0).astype(float)
    if method != 'ternary':
        raise ValueError(method)
    middle = expected_gain(artifact, w, SHRINK * z)
    choices = np.stack([middle, np.zeros_like(z), full], axis=-1)
    return z, np.asarray([SHRINK, 0., 1.])[choices.argmax(axis=-1)]


def prefix_actions(det, records, pair, xx):
    output = []
    for (x, ids, y, d), x_deploy in zip(records, xx):
        # The acquisition sees all factual rows but only the first three paid responses.
        h = det.fit_actions((x, ids[:3], y[:3], d[:3]), False)
        eng = PosteriorPredictiveEngine(pair, h)
        output.append(v33.decisions(batch_gap(eng, x_deploy)))
    out = np.stack(output)
    assert np.all((out == P).sum(1) == 13)
    return out


def paired_decode(decs, xx, actions, sensors):
    out = np.empty((len(decs), xx.shape[1], sensors[0][P].shape[1]))
    for e, (dec, x, a_e, obs) in enumerate(zip(decs, xx, actions, sensors)):
        for a in (Q, P):
            ids = np.flatnonzero(a_e == a)
            for j in range(0, len(ids), 8):
                ix = ids[j:j+8]
                prior = dec.inner.fit(t(x[ix]))[:, None, :]
                pred = dec.inner.update(prior, a, t(obs[a][ix])) @ dec.inner.label[:, 0]
                out[e, ix] = pred.cpu().numpy()/np.sqrt(2.)
    return out


def online(det, artifact, method, records, pair, decs, xx, sensor):
    torch.cuda.synchronize()
    start = time.perf_counter_ns()
    actions = prefix_actions(det, records, pair, xx)
    p = np.stack([v33.decode_selected(d, x, a, s) for d, x, a, s in zip(decs, xx, actions, sensor)])
    z, weight = policy(artifact, records, method, True)
    result = apply(p, (z*weight)[:,None], BETA)
    torch.cuda.synchronize()
    return (time.perf_counter_ns()-start)/1e6, actions, result


def timings(det, artifact, records, pair, decs, xx, obs, actions, p, z, weights, seed, timed):
    elapsed = np.zeros((3 if timed else 0, len(METHODS)))
    orders = np.zeros(elapsed.shape, dtype=int)
    if timed:
        sensor = [{a:o[a][:,0] for a in (Q, P)} for o in obs]
        rng = np.random.default_rng(seed)
        for rep in (-1,0,1,2):
            order = rng.permutation(len(METHODS))
            if rep >= 0:
                orders[rep] = order
            for m in order:
                ms, a, pred = online(det, artifact, METHODS[m], records, pair, decs, xx, sensor)
                np.testing.assert_array_equal(a, actions)
                np.testing.assert_allclose(pred, apply(p[:,:,0], (z*weights[m])[:,None], BETA),
                                           rtol=0., atol=1e-8)
                if rep >= 0:
                    elapsed[rep,m] = ms
    return elapsed, orders


def block(det, artifact, fingerprint, cell_index, index, out):
    name = CELLS[cell_index]
    shape, law = name.split('_')
    dst = out/name/f'block_{index:03d}'
    dst.mkdir(parents=True, exist_ok=True)
    meta = dst/'meta.json'
    if meta.exists():
        values = json.loads(meta.read_text())
        if values['fingerprint'] != fingerprint or any(digest(dst/n) != h for n,h in values['sha256'].items()):
            raise RuntimeError(f'Incomplete or incompatible block {dst}')
        return
    base = BASE + 1000000*cell_index + 10000*index
    alpha = ALPHAS[np.random.default_rng(base+401).permutation(8)]
    pair = [base+301, base+302]
    start = (-1, 1)[index%2]
    decs = [Engine(pair,*al) for al in alpha]
    files = {}
    for sign, orientation in ((1,'plus'),(-1,'minus')):
        context = (shape,law,sign)
        reference = None
        for regime in REGIMES:
            cal = [world.calibration(base+100+i, al, context, regime, start, [SPEED], HEIGHT)
                   for i,al in enumerate(alpha)]
            records = [r[0][0] for r in cal]
            x, y, d = observations(records)
            xx, labels, _, obs = zip(*[world.deployment(base+200+i, al, context, regime)
                                       for i,al in enumerate(alpha)])
            xx, labels = np.stack(xx), np.stack(labels)
            if reference is None:
                actions = prefix_actions(det, records, pair, xx)
                p = paired_decode(decs, xx, actions, obs)
                reference = (x, y[:,:3], xx, obs)
            else:
                for actual,expected in zip((x,y[:,:3],xx),reference[:3]):
                    np.testing.assert_array_equal(actual, expected)
                for oo, ref in zip(obs, reference[3]):
                    for a in (Q,P):
                        np.testing.assert_array_equal(oo[a],ref[a])
            weights = np.empty((2, len(METHODS), len(alpha)))
            risks = np.empty_like(weights)
            for k, use_fourth in enumerate((False,True)):
                for m, method in enumerate(METHODS):
                    z, w = policy(artifact, records, method, use_fourth)
                    weights[k,m] = w
                    prob = apply(p, (z*w)[:,None,None], BETA)
                    risks[k,m] = np.square(prob-labels[:,:,None]).mean(axis=(1,2))
            legacy = frozen_predict(artifact, x, y[:,:3], d[:,:3], y[:,3], 'alternating')
            np.testing.assert_allclose(legacy['z'], z, atol=1e-12,rtol=0.)
            for k in range(2):
                np.testing.assert_array_equal(weights[k,1],legacy[f'accept{k+3}'])
                np.testing.assert_allclose(weights[k,3],legacy[f'near_strength{k+3}'],atol=1e-12,rtol=0.)
            np.testing.assert_array_equal(risks[0,0],risks[1,0])
            if regime == REGIMES[0]:
                shared = dst/f'shared_{orientation}.npz'
                sel = {a:np.stack([o[a][actions[e]==a] for e,o in enumerate(obs)]) for a in (Q,P)}
                atomic_npz(shared,xx=xx,p_base=p,actions=actions,alpha=alpha,z=z,far_x=x,
                           first3_y=y[:,:3],dirs=d[:,:3],ids=np.stack([r[1] for r in records]),
                           probe_obs=sel[P],query_obs=sel[Q],Q=Q,P=P)
                files[shared.name] = digest(shared)
            tag = f'{regime}_{orientation}'
            timed = index < len(TIMED) and tag == TIMED[index][cell_index]
            ms, order = timings(det,artifact,records,pair,decs,xx,obs,actions,p,z,weights[1],
                                base+601+int(sign<0),timed)
            path = dst/(tag+'.npz')
            atomic_npz(path,risk=risks,weights=weights,methods=METHODS,stages=[3,4],labels=labels,
                       fourth_y=y[:,3],fourth_stick=np.asarray([r[3][0] for r in cal]),full_ms=ms,timing_order=order)
            files[path.name] = digest(path)
    atomic_json(meta,dict(cell=name,block=index,base=base,calibration_seeds=[base+100+i for i in range(8)],
                          deployment_seeds=[base+200+i for i in range(8)],pair=pair,start=start,fingerprint=fingerprint,sha256=files))
    print('done',name,index,flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke',action='store_true',help='use an independent seed for a single full block')
    args = parser.parse_args()
    if not (os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available() and torch.cuda.device_count()==1):
        raise RuntimeError('Requires one assigned GPU inside a Slurm allocation')
    torch.set_num_threads(2)
    old = json.loads((PROJECT/'probe_speed_v37/results/run.json').read_text())['hashes']
    for n,h in old.items():
        if digest(PROJECT/n) != h:
            raise RuntimeError(f'Frozen dependency changed: {n}')
    hashes = {**old,**{n:digest(ROOT/n) for n in ('run38.py','physics38.py')}}
    fp = '|'.join(hashes.values())
    out = ROOT/'smoke_results' if args.smoke else OUT
    out.mkdir(exist_ok=True)
    meta = dict(hashes=hashes,gpu=torch.cuda.get_device_name(),job=os.environ['SLURM_JOB_ID'],
                torch=str(torch.__version__),cuda=torch.version.cuda,methods=METHODS,
                speed=SPEED,height=HEIGHT,beta=BETA,shrink=SHRINK,regimes=REGIMES)
    atomic_json(out/'run.json',meta)
    with torch.inference_mode():
        det = Detector()
        artifact = load_artifacts()[SPEED]
        if args.smoke:
            block(det,artifact,fp,0,90,out)
        else:
            for b in range(BLOCKS):
                for c in range(len(CELLS)):
                    block(det,artifact,fp,c,b,out)
    atomic_json(out/'completion.json',dict(**meta,blocks=1 if args.smoke else BLOCKS*len(CELLS)))


if __name__ == '__main__':
    main()
