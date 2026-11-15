"""Incremental CPU work within the measured GPU service, after original p is on host."""
from pathlib import Path
import sys, json, time, platform, os
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'bidirectional_probe_v34'))
from direct_head34 import cue_est
from anchored34 import apply
cfg=json.loads((ROOT/'design35.json').read_text())
models=json.loads((ROOT.parent/'bidirectional_probe_v34/diag_models.json').read_text())
out={}
rng=np.random.default_rng(350210)
for sc in ('same','alternating'):
    panel=[]
    for obj in cfg['objects']:
        for surf in cfg['primary_surfaces']:
            for sign in ('plus','minus'):
                for b in cfg['timing_blocks']:
                    f=ROOT/'results'/f'{obj}_{surf}_{sign}'/f'block_{b:03d}'/(sc+'.npz')
                    with np.load(f) as m:
                        rawx=m['paid_x']; ix=m['paid_ids']
                        # Original online function indexes a pool of 1024 rows per environment.
                        pools=np.zeros((8,1024,11),dtype=rawx.dtype)
                        pools[np.arange(8)[:,None],ix]=rawx
                        rec=list(zip(pools,ix,m['paid_y'],m['paid_direction']))
                        panel.append((rec,m['p0'][:,:,0].copy(),m['p1'][:,:,0].copy()))
    def run(case):
        rec,p,reference=case
        start=time.perf_counter_ns()
        x=np.stack([r[0][r[1]] for r in rec]); y=np.stack([r[2] for r in rec]); ds=np.stack([r[3] for r in rec])
        z=cue_est(models,0,x,y,ds,sc)
        ans=apply(p,z[:,None],cfg['primary_beta'])
        elapsed=(time.perf_counter_ns()-start)*1.e-6
        np.testing.assert_allclose(ans,reference,atol=1.e-9,rtol=0)
        return elapsed
    for case in panel: run(case)
    arr=[run(panel[i]) for _ in range(30) for i in rng.permutation(len(panel))]
    out[sc]=dict(n=len(arr),p50_ms=float(np.median(arr)),p90_ms=float(np.quantile(arr,.9)),case_calls=len(panel))
assert not os.environ.get('CUDA_VISIBLE_DEVICES','')
(ROOT/'head_cost35.json').write_text(json.dumps(dict(description='Separate post-main, warm CPU microbenchmark; inputs already on host; excludes all shared work; not the GPU service latency.',cpu=platform.processor(),node=platform.node(),schedule=out),indent=2)+'\n')
