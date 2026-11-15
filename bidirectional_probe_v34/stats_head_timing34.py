from pathlib import Path
import numpy as np
from stats_policy34 import ROOT,read,write,paired_timing,sha

def main():
    cfg=read(ROOT/'policy_design34.json');dst=ROOT/'results_head_time';completion=read(dst/'completion.json')
    full=np.empty((4,2,2,3,2,3));warm=np.empty_like(full)
    for i,law in enumerate(cfg['laws']):
        for b in range(2):
            d=dst/law/f'block_{b:03d}';m=read(d/'meta.json')
            assert m['fingerprint']==completion['fingerprint'] and m['data_sha256']==sha(d/'timing.npz')
            with np.load(d/'timing.npz') as z:
                assert list(z['methods'])==['signed','fixed4','uncertainty']
                assert (z['counts'].sum(1)==32).all() and ((z['actions']==3).sum(2)==13).all()
                assert np.isfinite(z['full_ms']).all() and np.all((z['warm_ms']>0)&(z['full_ms']>=z['warm_ms']))
                full[i//2,b,i%2]=z['full_ms'].transpose(1,0,2)
                warm[i//2,b,i%2]=z['warm_ms'].transpose(1,0,2)
    picks=np.random.default_rng(3405771).integers(2,size=(10000,4,2))
    res=dict(methods=['signed','fixed4','uncertainty'],schedules=['same','alternating'],gpu=completion['gpu'],
             job=completion['job'],full=paired_timing(full,picks),warm=paired_timing(warm,picks))
    write(ROOT/'head_timing34.json',res);print('signed-head GPU p90',res['full']['p90'])
if __name__=='__main__':main()
