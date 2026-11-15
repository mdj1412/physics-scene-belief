"""Persist exact source model provenance and the policies' measured held-out GPU cost."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    out=ROOT/'results_eval'
    cfg=json.loads((ROOT/'design_eval.json').read_text())
    state=json.loads((out/'completion.json').read_text())
    locks=[]
    for suffix in ('','_rep2','_rep3'):
        folder=ROOT.parent/f'compute_matched_stopping_v32{suffix}'/'results_train'
        lf=folder/'model_lock.json'
        lock=json.loads(lf.read_text())
        locks.append(dict(model_path=str(lf.relative_to(ROOT.parent)),sha256=sha(lf),
                          source_training={arm:{key:lock['training'][arm][key] for key in
                            ('n_environments','labeled_cells','labels_s','fit_s','labels_plus_fit_s')}
                                  for arm in ('broad','repeated')}))
    assert state['model_sha256']==locks[0]['sha256']
    time_rows=[];devices=set();jobs=set();durations=[]
    for i,law in enumerate(cfg['laws']):
        for j in range(cfg['blocks_per_law']):
            folder=out/law/f'block_{j:03d}'
            meta=json.loads((folder/'meta.json').read_text())
            assert meta['model_sha256']==state['model_sha256']
            assert meta['fingerprint']==state['fingerprint']
            assert sha(folder/'metrics.npz')==meta['metrics_sha256']
            assert sha(folder/'choices.npz')==meta['choices_sha256']
            assert sha(folder/'allocation.json')==meta['allocation_sha256']
            jobs.add(meta['job']);devices.add(meta['gpu'])
            durations.append(meta['elapsed_s'])
            with np.load(folder/'metrics.npz') as rec:
                assert tuple(rec['full_ms'].shape)==((3 if j<cfg['timed_blocks_per_cell'] else 0),4)
                assert np.all(rec['counts'].sum(0)==32)
                assert np.all((rec['actions']==3).sum(1)==13)
                for repeat,(full,warm) in enumerate(zip(rec['full_ms'],rec['warm_ms'])):
                    time_rows.append(dict(law=law,block=j,repeat=repeat,full=full.tolist(),
                                          deployment_only=warm.tolist()))
    full=np.array([row['full'] for row in time_rows])
    warm=np.array([row['deployment_only'] for row in time_rows])
    setup=full-warm
    assert np.all(setup>0)
    rng=np.random.default_rng(33047093)
    # Resample the same object/surface blocks for both signed orientations and all policy timings.
    n=len(cfg['laws'])//2
    timed=cfg['timed_blocks_per_cell']
    per=np.zeros((n,timed,2,3,4))
    for row in time_rows:
        i=cfg['laws'].index(row['law'])
        per[i//2,row['block'],i%2,row['repeat']]=row['full']
    assert per.min()>0
    pick=rng.integers(timed,size=(10000,n,timed))
    q=np.quantile(per[np.arange(n)[None,:,None],pick],.9,axis=(1,2,3,4))
    ratios=q/q[:,2:3]
    d=dict(models=locks,all_training_was_historical=True,additional_v33_fit_seconds=0.,methods=cfg['methods'],
           distinct_timing_clusters=n*timed,paired_signed_timing_blocks=n*timed*2,
           n_full_calls_per_method=len(time_rows),calls_batch_size=1024,
           jobs=sorted(jobs),devices=sorted(devices),sum_block_wall_seconds=sum(durations),
           full_p90_ms=np.quantile(full,.9,axis=0).tolist(),deployment_only_p90_ms=np.quantile(warm,.9,axis=0).tolist(),
           calibration_allocation_setup_p90_ms=np.quantile(setup,.9,axis=0).tolist(),
           full_ratio_over_fixed4=(np.quantile(full,.9,axis=0)/np.quantile(full[:,2],.9)).tolist(),
           full_ratio_over_fixed4_pointwise95=np.quantile(ratios,[.025,.975],axis=0).T.tolist(),
           bootstrap='10000 draws; resample two paired timing blocks in each of four object/surface cells with signs/repeats/methods together; conditional on the first frozen source model pair and a single GPU.',
           raw_timing=time_rows)
    (out/'gpu_ledger33.json').write_text(json.dumps(d,indent=2)+'\n')
if __name__=='__main__':
    main()
