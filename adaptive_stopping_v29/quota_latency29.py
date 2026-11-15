"""Actual end-to-end locked quota and streaming latencies using the same decoder.

The full 128-world batch is contemporaneously available at the deployment API.
The measured execution starts with raw host factual inputs and ends with host stick
probabilities. Only each selected Y is transferred. Sensor wait/physical reset and
offline bank/estimator setup are excluded. Calibration setup is saved by runner29.
"""
import os
from pathlib import Path
import json
import hashlib
import numpy as np
from runner29 import (ROOT, Q, P, Engine, PosteriorPredictiveEngine, atomic_json, atomic_npz,
                      digest, evaluation_only, t, timed_call, torch)

METHODS = ('adaptive','fixed4')
BLOCKS = 12
N_ENVS = 2
BATCH_REPEATS = 3
STREAM_WORLDS = 10
STREAM_REPEATS = 2
SEED = 29021713


def risk_gap(selector, host_x):
    risks = selector.risk(selector.fit(t(host_x)),'keep256m2048')
    return float(risks[0]-risks[2])


def decode(decoder, host_x, action, host_y):
    w = decoder.inner.fit(t(host_x))
    return (decoder.inner.update(w,action,t(host_y)) @ decoder.inner.label[:,0]).item()/np.sqrt(2.)


def quota_prediction(selector,decoder,xx,ys,n):
    gaps=np.empty(len(xx))
    for j,x in enumerate(xx):
        gaps[j]=risk_gap(selector,x)
    rank=np.argsort(-gaps,kind='stable')
    acts=np.full(len(xx),Q,dtype=np.int8)
    acts[rank[:n]]=P
    probs=np.empty(len(xx))
    for j,(x,act) in enumerate(zip(xx,acts)):
        probs[j]=decode(decoder,x,int(act),ys[int(act)][j])
    return acts,probs


def stream_prediction(selector,decoder,x,q,probe):
    a = P if risk_gap(selector,x)>1.e-12 else Q
    return a,decode(decoder,x,a,probe if a==P else q)


def array_hash(*arrs):
    h=hashlib.sha256()
    for a in arrs:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def measure_environment(folder, meta, arrays, index, cfg, rng):
    env=meta['environments'][index];pair=meta['bank_pair'];n=cfg['quota_probes_per_environment']
    with np.load(folder/'choice_lock.npz') as locked:
        xx=locked['x_test'][index]
    # Obtain only the simulator's potential observation buffers for the timing harness;
    # inference functions choose action before looking up the selected host buffer.
    # Keep the original number of noise repeats to reproduce the same first draw per world.
    _,_,all_y=evaluation_only(len(xx),cfg['repeats'],env['seed'],env['alpha'])
    ys={a:all_y[a][:,0] for a in (Q,P)}
    known=Engine(pair,*env['alpha'])
    if 'quota_action' not in arrays:
        raise ValueError('Sealed quota choices missing.')
    known_index=list(arrays['methods']).index('known')
    # The reference uses recorded first-noise-draw probabilities from the common
    # known decoder under BOTH potential observations. No scoring path reads it.
    aa=arrays['quota_action'][index,:,:2]
    stream_aa=arrays['action'][index,:,:2]
    preds=arrays['probabilities'][index,:,known_index,:2,0]
    expected=preds[np.arange(len(xx))[:,None],(aa==P).astype(int)]
    streaming=preds[np.arange(len(xx))[:,None],(stream_aa==P).astype(int)]
    selectors=[]
    for i in range(len(METHODS)):
        k=str(int(arrays['counts'][index,i]))
        selectors.append(PosteriorPredictiveEngine(pair,meta['estimates'][index][k]))
    batch=np.empty((BATCH_REPEATS,2));raw_preds=np.empty((BATCH_REPEATS,2,len(xx)))
    for repeat in range(-1,BATCH_REPEATS):
        for policy in rng.permutation(2):
            (actions,pred),ms=timed_call(quota_prediction,selectors[policy],known,xx,ys,n)
            if not np.array_equal(actions,aa[:,policy]) or np.max(abs(pred-expected[:,policy]))>1.e-9:
                raise ValueError(f"Quota or predictions mismatch: {env['id']}/{policy}")
            if repeat>=0:
                batch[repeat,policy]=ms
                raw_preds[repeat,policy]=pred
    ids=rng.choice(len(xx),STREAM_WORLDS,replace=False)
    stream=np.empty((STREAM_WORLDS,STREAM_REPEATS,2))
    for repeat in range(-1,STREAM_REPEATS):
        slots=range(min(3,len(ids))) if repeat<0 else rng.permutation(len(ids))
        for j in slots:
            idx=ids[j]
            for policy in rng.permutation(2):
                (act,pred),ms=timed_call(stream_prediction,selectors[policy],known,xx[idx],
                                          ys[Q][idx],ys[P][idx])
                if act!=stream_aa[idx,policy] or abs(pred-streaming[idx,policy])>1.e-9:
                    raise ValueError('Streaming result differs from locked known-decoder output.')
                if repeat>=0:
                    stream[j,repeat,policy]=ms
    assert (batch>0).all() and (stream>0).all()
    return batch,stream,ids,array_hash(aa,raw_preds),float(np.max(abs(raw_preds-expected.T[None])))


def bootstrap_ratios(times,seed):
    # World samples are nested within environments for stream and repeated timings
    # carried together within each collection for both policies.
    rng=np.random.default_rng(seed)
    b,e=times.shape[:2]
    ratios=[]
    for _ in range(3000):
        bi=rng.integers(b,size=b)
        ei=rng.integers(e,size=(b,e))
        sample=times[bi[:,None],ei]
        if times.ndim==5:
            w=times.shape[2];wi=rng.integers(w,size=(b,e,w))
            sample=sample[np.arange(b)[:,None,None],np.arange(e)[None,:,None],wi]
        q=np.quantile(sample.reshape(-1,2),.9,axis=0)
        ratios.append(q[0]/q[1])
    return np.quantile(ratios,[.025,.975]).tolist()


def main():
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.device_count()==1 and torch.cuda.is_available()
    cfg=json.loads((ROOT/'design.json').read_text())
    torch.set_num_threads(2)
    out=ROOT/'results'
    cache=out/'quota_timing';cache.mkdir(parents=True,exist_ok=True)
    # Re-running after interruption reuses only exact hash-matching finished blocks.
    fingerprint=digest(Path(__file__).resolve())
    block_batches=[];block_stream=[];env_ids=[];stream_ids=[];checks=[]
    with torch.inference_mode():
        for b in range(BLOCKS):
            folder=out/f'block_{b:03d}';m=json.loads((folder/'metrics.json').read_text())
            assert m['choices_sha256']==digest(folder/'choice_lock.npz') and m['npz_sha256']==digest(folder/'metrics.npz')
            ids=np.random.default_rng(np.random.SeedSequence([SEED,b])).choice(len(m['environments']),N_ENVS,replace=False)
            ck=cache/f'block_{b:03d}.npz';record=cache/f'block_{b:03d}.json'
            if record.is_file():
                prev=json.loads(record.read_text())
                assert prev['main_sha256']==m['npz_sha256'] and prev['fingerprint']==fingerprint and prev['sha256']==digest(ck)
                with np.load(ck) as z:
                    ba,st,si=z['batch'],z['stream'],z['stream_ids']
                logs=prev['checks']
            else:
                with np.load(folder/'metrics.npz') as f:
                    arrays={k:f[k] for k in ('methods','quota_action','action','probabilities','counts')}
                bs=[];ss=[];ii=[];logs=[]
                for j,idx in enumerate(ids):
                    rng=np.random.default_rng(np.random.SeedSequence([SEED,b,7,j]))
                    bt,st,si,h,err=measure_environment(folder,m,arrays,int(idx),cfg,rng)
                    bs.append(bt);ss.append(st);ii.append(si);logs.append(dict(env=m['environments'][idx]['id'],
                        output_sha256=h,max_abs_probability_error=err))
                ba,st,si=np.stack(bs),np.stack(ss),np.stack(ii)
                atomic_npz(ck,batch=ba,stream=st,stream_ids=si,env_indices=ids)
                atomic_json(record,dict(main_sha256=m['npz_sha256'],fingerprint=fingerprint,sha256=digest(ck),checks=logs))
            block_batches.append(ba);block_stream.append(st);env_ids.append(ids);stream_ids.append(si);checks.append(logs)
            print('quota measured block',b,flush=True)
    batch=np.stack(block_batches);stream=np.stack(block_stream);result=out/'quota_latency29.npz'
    atomic_npz(result,batch_times_ms=batch,stream_times_ms=stream,env_indices=env_ids,stream_ids=stream_ids,methods=METHODS)
    metrics={}
    for label,v in [('batch',batch),('stream',stream)]:
        q=np.quantile(v.reshape(-1,2),.9,axis=0)
        metrics[label]=dict(p90_ms=q.tolist(),p50_ms=np.median(v.reshape(-1,2),axis=0).tolist(),
            adaptive_over_fixed=float(q[0]/q[1]),ratio_ci95=bootstrap_ratios(v,SEED+34),n_per_method=int(v.size//2))
    atomic_json(out/'quota_latency29.json',dict(metrics=metrics,methods=METHODS,npz_sha256=digest(result),
        checks=checks,blocks=BLOCKS,n_envs_per_block=N_ENVS,quota_per_env=cfg['quota_probes_per_environment'],
        n_worlds_per_batch=cfg['n_test'],n_complete_warmups_per_env=1,
        scope='Held-out common known-prior decoder, raw host inputs through verified output probabilities; bank setup and physical sensing excluded.',
        gpu=torch.cuda.get_device_name(),job=os.environ['SLURM_JOB_ID'],host=os.environ.get('SLURMD_NODENAME')))
    print(metrics,flush=True)

if __name__=='__main__':
    main()
