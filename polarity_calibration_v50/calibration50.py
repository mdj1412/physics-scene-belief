"""Generate independent sacrificial witness pairs and estimate session-level polarity."""
import json
import os
from pathlib import Path

import numpy as np

from estimator50 import DEAD_ZONE, infer
import world50 as world

ROOT = Path(__file__).resolve().parent
KAPPAS = (0., .25, .50, .75, 1.)
BUDGETS = (0, 1, 4, 8)
REPEATS = 256


def instrument(seed, site):
    # Reconstruct only the shared weak instrument's scalar gain/bias, leaving
    # the independent calibration support's per-object drift and readouts fresh.
    rng = np.random.default_rng(np.random.SeedSequence([seed, 500714, site]))
    rng.uniform(.9, 1.1, (4,128,1))
    return rng.uniform(.93,1.07), rng.uniform(-.012,.012)


def simulate(panel):
    with np.load(ROOT/'worlds'/f'panel_{panel:02d}.npz') as w:
        seed, cal_seed = int(w['seed']), int(w['cal_seed'])
        assert int(w['cal_support_panel']) != int(w['support_panel'])
        cal_ids = w['cal_object_index']
        assert all(len(np.unique(row))==128 for row in cal_ids)
    rng = np.random.default_rng(np.random.SeedSequence([cal_seed,501124]))
    # Geometry & sensor coupling are attached to the individual witness; the
    # data rows below are different sessions on the same conditional population.
    families = np.stack([rng.permutation(np.repeat(np.arange(4),32)) for _ in range(4)])
    specimen_gain = rng.uniform(.90,1.10,(4,128))
    specimen_bias = rng.normal(0,.007,(4,128))
    specimen_full_bias = rng.normal(0,.017,(4,128))
    full_inst_gain = rng.uniform(.94,1.06,(4,2))
    full_inst_bias = rng.uniform(-.020,.020,(4,2))
    chosen = np.argsort(rng.random((4,REPEATS,128)),axis=-1)[...,:8]
    assert np.all(np.diff(np.sort(chosen,axis=-1),axis=-1)>0)
    env = np.arange(4)[:,None,None]
    family = families[env,chosen]
    height = rng.integers(2,size=chosen.shape)
    weak_state = rng.choice([0,-1,1],p=[.5,.25,.25],size=chosen.shape)
    keep_coin = rng.random(chosen.shape)
    w_noise, f_noise = rng.normal(size=(2,*chosen.shape))
    geometry = np.array([[world.gauge_mm(f,h,1,load_fraction=1.) for h in world.HEIGHTS] for f in world.FAMILIES])
    world_gain, world_bias = np.array([instrument(seed,h) for h in world.HEIGHTS]).T
    amp = geometry[family,height]*specimen_gain[env,chosen]
    dw = world_gain[height]*.20*amp*weak_state + world_bias[height]+specimen_bias[env,chosen]+.028*w_noise
    full = np.stack([full_inst_gain[env,height]*amp*np.where(keep_coin<k, weak_state,-weak_state)
                     +full_inst_bias[env,height]+specimen_full_bias[env,chosen]+.060*f_noise for k in KAPPAS])
    weak = np.broadcast_to(dw,full.shape).copy()
    # Keep precisely the fp32 meter outputs used in both estimation and timed replay.
    weak,full = weak.astype(np.float32),full.astype(np.float32)
    khat = np.stack([infer(weak[...,:n], full[...,:n]) for n in BUDGETS],axis=1)
    shuffled=np.stack([infer(weak[...,:n],np.roll(full,1,axis=2)[...,:n]) for n in BUDGETS],axis=1)
    usable=(abs(weak)>=DEAD_ZONE[0]) & (abs(full)>=DEAD_ZONE[1])
    assert np.array_equal(weak[0],weak[-1]) and np.all(khat[:,0]==.5)
    check=abs(infer(weak[...,:4],-full[...,:4])+khat[:,2]-1.).max()
    assert check<1.e-12
    dest=ROOT/'cal50'/f'panel_{panel:02d}.npz'
    dest.parent.mkdir(exist_ok=True)
    tmp=dest.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, weak_mm=weak,full_mm=full,khat=khat,khat_unpaired=shuffled,
                        usable=usable,cal_object_id=cal_ids[env,chosen],height=np.array(world.HEIGHTS)[height],
                        family=family,cal_seed=cal_seed,kappa=KAPPAS,K=BUDGETS)
    tmp.replace(dest)
    return khat,shuffled,usable


def main():
    assert os.getenv('SLURM_JOB_ID')
    outputs=[simulate(i) for i in range(12)]
    k=np.stack([x[0] for x in outputs])
    unpaired=np.stack([x[1] for x in outputs])
    usable=np.stack([x[2] for x in outputs])
    err=k-np.array(KAPPAS)[None,:,None,None,None]
    count=np.stack([usable[..., :n].sum(axis=-1).mean((0,2,3)) for n in BUDGETS])
    # Endpoint contrast across hidden κ regimes is a negative control for pairing.
    result=dict(kappa=KAPPAS,K=BUDGETS,repeats=REPEATS,catalogs=12,environments_per_catalog=4,
                axes='rows kappa, columns paid K; mean over environments/repeats within equal-weight catalogs',
                mean=k.mean((0,3,4)),mae=abs(err).mean((0,3,4)),rmse=np.sqrt((err**2).mean((0,3,4))),
                bias=err.mean((0,3,4)),flip_rate=(k<.5).mean((0,3,4)),tie_rate=(k==.5).mean((0,3,4)),
                unpaired_mean=unpaired.mean((0,3,4)),unpaired_flip_rate=(unpaired<.5).mean((0,3,4)),
                mean_readable=count.T,
                scope='Weak observations identical across true kappa at each paired population, calibration slots unique within prefix, disjoint support IDs; no deployment Y loaded.')
    (ROOT/'calibration50.json').write_text(json.dumps(result,default=lambda v:np.asarray(v).tolist(),indent=1))
    print('calibration complete',result['flip_rate'])


if __name__=='__main__':
    main()
