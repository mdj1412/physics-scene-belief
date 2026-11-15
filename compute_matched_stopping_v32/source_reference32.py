"""Read-only reconstruction of independent, archived v31 source prefixes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
V31 = ROOT.parent / 'probe_gain_reliability_v31'
V30 = ROOT.parent / 'learned_probe_value_v30'
OUT = ROOT / 'results_reference'
SEED = 32093151
STAGES = (2, 4)
ARMS = ('broad', 'repeated')
TOL = 1.e-8


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def save(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def dependencies():
    sys.path.insert(0, str(V30))
    from score30 import NAMES, features, predict
    from estimator28 import ProbeShiftEstimator, torch
    if os.environ.get('BELIEF_DEVICE') != 'cpu' or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Run with BELIEF_DEVICE=cpu and CUDA_VISIBLE_DEVICES empty.')
    if not os.environ.get('SLURM_JOB_ID') or torch.cuda.is_available():
        raise RuntimeError('An allocated CPU and CUDA disabled are required.')
    torch.set_num_threads(1)
    return NAMES, features, predict, ProbeShiftEstimator, torch


def ref_block(b, pool=False):
    return V31 / ('results_pool' if pool else 'results') / 'coulomb' / f'block_{b:03d}'


def extract():
    names, features, predict, estimator, torch = dependencies()
    old_file = V30/'results/model_lock.json'
    old = read(old_file)
    assert names == old['features']
    features_all = np.empty((6, 2, 8, len(names)))
    seed_rows = []
    archive = []
    errors = np.zeros(2)
    with torch.inference_mode():
        detector = estimator(seed=3003913, bank_size=8192)
        assert detector.centers.device.type == detector.joint_centers.device.type == 'cpu'
        for block in range(6):
            folder = ref_block(block)
            meta = read(folder/'metrics.json')
            prefix = folder/'prefix.npz'
            assert (meta['block'], meta['law']) == (block,'coulomb')
            assert meta['model_sha256'] == sha(old_file) and meta['prefix_sha256'] == sha(prefix)
            with np.load(prefix) as z:
                xx, ids, yy = z['factual_pool'], z['paid_ids'], z['paid_y']
                assert xx.shape==(8,1024,11) and ids.shape==(8,4) and yy.shape==(8,4,2)
                np.testing.assert_equal(z['pool_seeds'], meta['pool_seeds'])
                seed_rows.append(z['pool_seeds'].tolist())
                for e,(x,ii,y) in enumerate(zip(xx,ids,yy)):
                    assert len(set(ii))==4
                    ll=detector.component_loglikelihood(x)
                    joint=detector.joint_component_loglikelihood(x[ii],y)
                    for s,k in enumerate(STAGES):
                        est=detector.posterior_from_components(ll,ii[:k],joint[:k])
                        feats=features(est,x,ii[:k],y[:k])
                        features_all[block,s,e]=feats
                        delta=np.abs([predict(old['models'][s],feats)-z['score'][s,e],
                                      est['sd'][1]-z['uncertainty'][s,e]])
                        errors=np.maximum(errors,delta)
            archive.append({'prefix':sha(prefix),'archive_meta':sha(folder/'metrics.json')})
            print('Restored v31 source block',block,'max errors',errors,flush=True)
    assert (errors < TOL).all() and np.isfinite(features_all).all()
    OUT.mkdir(exist_ok=True)
    f=OUT/'prefix_features.npz'
    tmp=f.with_suffix('.tmp.npz')
    np.savez_compressed(tmp,features=features_all,pool_seeds=seed_rows)
    os.replace(tmp,f)
    save(OUT/'prefix_seal.json',dict(host=platform.node(),job=os.environ['SLURM_JOB_ID'],
         torch_version=torch.__version__,device=str(detector.centers.device),detector_seed=3003913,
         bank_size=8192,feature_names=names,archive=archive,features_sha256=sha(f),
         v30_model_sha256=sha(old_file),source_sha256=sha(Path(__file__)),
         plan_sha256=sha(ROOT/'SOURCE_REFERENCE_PLAN_KO.md'),max_abs_v30_score_and_sd=errors.tolist()))


def corr(pred, target):
    pc=pred-pred.mean(-1,keepdims=True)
    yc=target-target.mean(-1,keepdims=True)
    den=np.sqrt((pc**2).sum(-1)*(yc**2).sum(-1))
    if (den<=0).any():
        raise ValueError('Pearson correlation is undefined for constant resamples.')
    return (pc*yc).sum(-1)/den


def summarize(x, y, constants):
    # x inputs have shape [draw,arm,stage,environment]. y has singleton arm.
    mse=((x-y)**2).mean(-1)
    baseline=((constants-y)**2).mean(-1)
    cc=corr(x,y)
    return {'mse':mse, 'intercept_mse':baseline,'mse_minus_intercept':mse-baseline,
            'correlation':cc,'repeated_minus_broad_mse':mse[:,1]-mse[:,0],
            'repeated_minus_broad_correlation':cc[:,1]-cc[:,0]}


def evaluate():
    names, _, predict, _, _=dependencies()
    seal=read(OUT/'prefix_seal.json')
    feature_file=OUT/'prefix_features.npz'
    assert sha(feature_file)==seal['features_sha256'] and seal['feature_names']==names
    assert seal['plan_sha256']==sha(ROOT/'SOURCE_REFERENCE_PLAN_KO.md')
    lock=ROOT/'results_train/model_lock.json'
    model=read(lock)
    assert model['features']==names
    with np.load(feature_file) as z:
        x=z['features']; source_seeds=set(z['pool_seeds'].ravel())
    # Ref predictions are fixed with no v31 gain inputs.
    preds=np.array([[predict(model['models'][arm][s], x[:,s]) for s in range(2)]
                    for arm in ARMS])
    constants=np.array([[m['intercept'] for m in model['models'][arm]] for arm in ARMS])
    for a,arm in enumerate(ARMS):
        trained=set()
        for row in model['training'][arm]['blocks']:
            f=ROOT/'results_train'/arm/f"block_{row['block']:03d}"/'data.npz'
            assert sha(f)==row['data_sha256']
            with np.load(f) as z:
                trained.update(z['prefix_seed'].ravel())
        assert trained.isdisjoint(source_seeds)
        np.testing.assert_allclose(constants[a],model['training'][arm]['mean_targets'],atol=1e-12,rtol=0)
    target=[]; hashes=[]
    for b in range(6):
        folder=ref_block(b,True); q=folder/'metrics.npz'
        meta=read(folder/'metrics.json')
        assert (meta['law'],meta['block'])==('coulomb',b)
        assert meta['prefix_sha256']==seal['archive'][b]['prefix'] and meta['metrics_sha256']==sha(q)
        assert meta['model_sha256']==seal['v30_model_sha256']
        with np.load(q) as z:
            gain=z['gain']
            assert gain.shape==(2,8,4,4)
            np.testing.assert_allclose(gain,z['before'][:,:,None,:]-z['after'],rtol=0,atol=1e-12)
            target.append(gain.mean((-1,-2)))
        hashes.append(sha(q))
    y=np.stack(target,axis=1)[None,...]
    def stats(ix):
        xx=np.take(preds,ix,axis=2).transpose(2,0,1,3,4).reshape(len(ix),2,2,-1)
        yy=np.take(y,ix,axis=2).transpose(2,0,1,3,4).reshape(len(ix),1,2,-1)
        return summarize(xx,yy,constants[None,:,:,None])
    est=stats(np.arange(6)[None])
    # Law is fixed; resample entire independent blocks. Keep stage and method pairing.
    bs=stats(np.random.default_rng(SEED).integers(6,size=(10000,6)))
    summary={k:{'value':vv[0].tolist(),'pointwise95':np.moveaxis(np.quantile(bs[k],[.025,.975],axis=0),0,-1).tolist()}
             for k,vv in est.items()}
    save(OUT/'source_reference32.json',dict(summary=summary,arm_order=ARMS,stage_order=['2to4','4to8'],
         n=48,blocks=6,bootstrap_seed=SEED,bootstrap_draws=10000,
         reference_seal_sha256=sha(OUT/'prefix_seal.json'),lock_sha256=sha(lock),input_metrics_sha256=hashes,
         target_mean=y.mean((2,3)).ravel().tolist(),target_sd=y.reshape(2,-1).std(1,ddof=1).tolist(),
         intercepts=constants.tolist(),feature_reconstruction_errors=seal['max_abs_v30_score_and_sd'],
         source_sha256=sha(Path(__file__)),host=platform.node(),job=os.environ['SLURM_JOB_ID']))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('part',choices=['features','score'])
    (extract if ap.parse_args().part=='features' else evaluate)()
