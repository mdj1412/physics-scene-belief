"""Independent audit of allocation, predictive Brier, sensor noise, and factual control."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
R = ROOT / 'results'
LAWS = ('coulomb','split_static','speed_weak')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cols=[]
    seed_log=[]
    checks=[]
    clock=[]
    flips=[]
    mcse=[]
    blocks=[]
    fingerprint=None
    model=sha(ROOT.parent/'learned_probe_value_v30'/'results'/'model_lock.json')
    for law in LAWS:
        for block in range(6):
            folder=R/law/f'block_{block:03d}'
            meta=json.loads((folder/'metrics.json').read_text())
            assert meta['law']==law and meta['block']==block and meta['model_sha256']==model
            fingerprint = meta['fingerprint'] if fingerprint is None else fingerprint
            assert meta['fingerprint']==fingerprint
            for n in ['metrics','prefix','choices']:
                assert sha(folder/(n+'.npz'))==meta[n+'_sha256']
            clock.append(meta['elapsed_s'])
            for key in ('pool_seeds','future_seeds','batch_seeds','bank_pair'):
                seed_log.extend(np.asarray(meta[key]).ravel().tolist())
            checks.extend(meta['reference_max_errors'])
            with np.load(folder/'metrics.npz') as z, np.load(folder/'choices.npz') as c, np.load(folder/'prefix.npz') as prefix:
                prob=z['probabilities']; labels=z['labels']
                assert prob.shape==(8,4,128,2,32) and labels.shape==(8,4,128)
                assert np.all((prob>=-1e-12)&(prob<=1+1e-12))
                draw_loss=(prob-labels[:,:,:,None,None])**2
                assert np.allclose(draw_loss.mean(-1),z['cost'],rtol=0,atol=1e-12)
                assert all(len(set(row))==4 for row in prefix['paid_ids'])
                assert np.array_equal(meta['future_seeds'],prefix['future_seeds'])
                actions=np.zeros_like(c['gap'],dtype=np.int8)+1
                rank=np.argsort(-c['gap'],axis=-1,kind='stable')
                np.put_along_axis(actions,rank[...,:13],3,axis=-1)
                old=actions[:,:,0];new=actions[:,:,1:]
                assert np.array_equal(old,c['action_before']) and np.array_equal(new,c['action_after'])
                # Before minus after = action change multiplied by paired advantage (Q-P loss).
                # +/- signs reverse if switching probe -> query.
                switches=(new==3).astype(int)-(old[:,:,None]==3)
                advantage = draw_loss[:,:,:,0,:]-draw_loss[:,:,:,1,:]
                dg=(switches[:,:,:,:,:,None]*advantage[None,:,None,:,:,:]).mean(-2)
                assert dg.shape==(2,8,4,4,32)
                assert np.allclose(z['gain'],dg.mean(-1),rtol=0,atol=1.e-12)
                flips.append((switches!=0).mean((1,2,3,4)))
                mcse.append((dg.var(-1,ddof=1)/32).mean((1,2,3)))
                if block==0:
                    assert np.allclose(z['gain_probe_only']+z['gain_factual_only'],z['gain'],rtol=0,atol=2e-14)
                    co=c['factual_only_action']
                    assert ((co==3).sum(-1)==13).all()
                    risk_x=np.where(co==3,z['cost'][None,:,None,:,:,1],z['cost'][None,:,None,:,:,0]).mean(-1)
                    assert np.allclose(risk_x,z['factual_only_brier'],rtol=0,atol=1e-12)
                    row=[]
                    for name in ['gain','gain_factual_only','gain_probe_only']:
                        g=z[name]
                        row.append([g.mean((1,2,3)),np.abs(g).mean((1,2,3))])
                    cols.append(row)
                blocks.append(str(folder.relative_to(ROOT)))
    assert len(set(seed_log))==len(seed_log)
    z=np.array(cols)  # 3 laws × (total,x,y|x) × (mean,MAE) × stage
    data=dict(laws=LAWS,seed_codes=len(seed_log),metric_hashes=len(blocks)*3,
              reference_batches=len(checks),max_ref_gap=float(np.max(checks)),block_seconds=sum(clock),
              action_flip=np.mean(flips,axis=0).tolist(),conditional_sensor_se_rms=np.sqrt(np.mean(mcse,axis=0)).tolist(),
              control_metrics=['total','factual only','probe given factual'],control_law=z.tolist(),control_pooled=z.mean(0).tolist(),
              model_sha256=model,blocks=blocks)
    (R/'cell_replay31.json').write_text(json.dumps(data,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in data.items() if k not in ['blocks','control_law']}))

if __name__=='__main__':
    main()
