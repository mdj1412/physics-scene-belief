"""Recompute scores from immutable outcomes, raw inputs, and the v38 table on CPU."""
import hashlib
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent/'probe_transfer_v39'))
from audit39 import source_reconstruction, load_artifacts, digest, BETA, SHRINK

CELLS = tuple('_'.join(t) for t in itertools.product(
    ('shelf','ridge'), ('base','hollow','bulge'), ('saturating','crossover')))


def read(path):
    with np.load(path) as arch:
        return {key: arch[key] for key in arch.files}


def cpu_audit():
    if os.environ.get('BELIEF_DEVICE') != 'cpu':
        raise RuntimeError('Set BELIEF_DEVICE=cpu to run the independent CPU audit')
    directory = ROOT/'results'
    run = json.loads((directory/'completion.json').read_text())
    for key, expected in run['hashes'].items():
        assert digest(ROOT.parent/key) == expected, key
    assert run['blocks'] == 288 and tuple(run['cells']) == CELLS
    art = load_artifacts()[1.10]
    source_advantage = {key: [] for key in CELLS}
    timing = []
    seedset = set()
    sensor_rows = {key: [] for key in CELLS}
    max_brier = max_weight = max_cue = 0.
    hard_matches = 0
    probes = 0
    min_p, max_p = 1., 0.
    for catalog in range(12):
        for b in range(2):
            seed = 400_000_000 + 100_000*catalog + 10_000*b
            cal_seeds = [seed+100+e for e in range(8)]
            test_seeds = [seed+200+e for e in range(8)]
            pair = [seed+301,seed+302]
            seeds = cal_seeds + test_seeds + pair
            assert not seedset.intersection(seeds)
            seedset.update(seeds)
            refs = {}
            source = {}
            for c, cell in enumerate(CELLS):
                family, shape, law = cell.split('_')
                f = directory/cell/f'catalog_{catalog:02d}'/f'block_{b:02d}.npz'
                meta = json.loads(f.with_suffix('.json').read_text())
                assert digest(f) == meta['sha256'] and meta['fingerprint'] == run['fingerprint']
                assert (meta['catalog'], meta['block'], meta['base_seed'], meta['start']) == (
                    catalog, b, seed, (-1,1)[b])
                assert meta['calibration_seeds']==cal_seeds and meta['deployment_seeds']==test_seeds
                assert meta['pair']==pair
                a = read(f)
                assert tuple(a['methods'])==('fixed','optimizer') and tuple(a['signs'])==(1,-1)
                assert tuple(a['regimes'])==('strong','weak','null')
                assert a['risk'].shape==(2,3,2,8) and a['labels'].shape==(2,3,8,128)
                assert a['p_base'].shape==(2,8,128,32) and a['labels'].dtype==bool
                assert np.all((a['actions']==1).sum(-1)==115) and np.all((a['actions']==3).sum(-1)==13)
                if b==0 and catalog in (c,(c+6)%12):
                    st = (c+3*(catalog!=c))%6
                    assert a['full_ms'].shape==(3,2) and np.all(np.isfinite(a['full_ms']))
                    assert a['full_ms'].min()>0
                    np.testing.assert_array_equal(np.sort(a['order'],axis=1), [[0,1]]*3)
                    timing.append([catalog,c,int(st)])
                else:
                    st = -1
                    assert a['full_ms'].shape==(0,2)
                assert int(a['timing_stratum'])==st
                reference = refs.setdefault((family,law), a)
                if shape!='base':
                    for key in ('cal_x','first3_y','ids','dirs','fourth_y','fourth_stick',
                                'weights','z','alpha','labels','actions'):
                        np.testing.assert_array_equal(a[key], reference[key])
                    is_probe = a['actions']==3
                    np.testing.assert_array_equal(a['p_base'][is_probe], reference['p_base'][is_probe])
                    hard_matches += a['labels'].size
                    probes += is_probe.sum()*32
                base = a['p_base'][:,None,:,:,:]
                labels = a['labels'][:,:,:,:,None]
                min_p = min(min_p, float(base.min()))
                max_p = max(max_p, float(base.max()))
                assert np.isfinite(base).all() and min_p>=-1.e-12 and max_p<=1+1.e-12
                if shape=='base':
                    for s in range(2):
                        for r in range(3):
                            source[family,law,s,r] = source_reconstruction(
                                art, a['cal_x'][s], a['first3_y'][s], a['dirs'][s], a['fourth_y'][s,r])
                z = np.stack([source[family,law,s,0][0] for s in range(2)])
                w = np.array([[source[family,law,s,r][1] for r in range(3)] for s in range(2)])
                adv = np.array([[source[family,law,s,r][2] for r in range(3)] for s in range(2)])
                np.testing.assert_allclose(z,a['z'],rtol=0,atol=2.e-12)
                np.testing.assert_allclose(w,a['weights'],rtol=0,atol=2.e-10)
                max_weight=max(max_weight,float(np.max(np.abs(w-a['weights']))))
                max_cue=max(max_cue,float(np.max(np.abs(z-a['z']))))
                losses=[]
                for scale in (np.full(w.shape, SHRINK),w):
                    factor = np.exp(BETA*z[:,None,:]*scale)[:,:,:,None,None]
                    prob = base*factor/(1+base*(factor-1))
                    losses.append((prob-labels)**2)
                calculated = np.stack([l.mean(axis=(-1,-2)) for l in losses],axis=2)
                np.testing.assert_allclose(calculated,a['risk'],rtol=0,atol=2.e-12)
                max_brier = max(max_brier,float(np.max(np.abs(calculated-a['risk']))))
                delta = (losses[1]-losses[0]).mean(-1)
                sensor_delta = []
                for action in (1,3):
                    mask = np.broadcast_to(a['actions'][:,None,:,:]==action,delta.shape)
                    sensor_delta.append(float(delta[mask].mean()))
                np.testing.assert_allclose(np.average(sensor_delta,weights=[115,13]),delta.mean())
                sensor_rows[cell].append(sensor_delta)
                source_advantage[cell].append([float(adv.mean()), float(-delta.mean())])
    assert len(timing)==24 and len(seedset)==24*18
    np.testing.assert_array_equal(np.bincount(np.array(timing)[:,2]),[4]*6)
    result = dict(blocks=288,independent_panels=12,panel_blocks=24,evaluation_seeds=24*16,
                  qmc_bank_seeds=24*2,timing_clusters=timing,
                  max_abs_recomputed_brier=max_brier, max_abs_recomputed_weight=max_weight,
                  max_abs_recomputed_cue=max_cue,base_range=[min_p,max_p],
                  paired_labels_matched=int(hard_matches),paired_probe_probabilities_matched=int(probes),
                  source_expected_gain_and_observed_reduction={
                      k:np.asarray(v).mean(0).tolist() for k,v in source_advantage.items()},
                  delta_by_selected_query_view_and_probe={
                      k:np.asarray(v).mean(0).tolist() for k,v in sensor_rows.items()})
    (ROOT/'audit40.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    print(json.dumps(cpu_audit(),ensure_ascii=False))
