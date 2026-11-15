"""Partition schedule differences into count allocation and downstream schedule risk at fixed count."""
from pathlib import Path
import json
import numpy as np
from stats_policy34 import ROOT,SCHEDULES,METHODS,STAGES,read,write,paired_boot
def main():
    cfg=read(ROOT/'policy_design34.json')
    chunks=np.empty((4,8,2,3,3))
    stage=np.empty((4,8,2,3))
    allocation_change=np.empty((4,8,2,3))
    observation_change=np.empty_like(allocation_change)
    for f in range(4):
        for b in range(8):
            for d in range(2):
                folder=ROOT/'results_policy'/cfg['laws'][2*f+d]/f'block_{b:03d}'
                cells=[np.load(folder/n/'metrics.npz') for n in SCHEDULES]
                count=np.stack([z['counts'] for z in cells]);rs=[z['oracle_stage_risk'] for z in cells]
                selected=[z['actions'] for z in cells]
                allocation_change[f,b,d]=(count[1]!=count[0]).mean(0)
                observation_change[f,b,d]=(selected[1]!=selected[0]).mean((0,1))
                stage[f,b,d]=(rs[1]-rs[0]).mean(axis=0)
                vv=np.array([[r[np.arange(8)[:,None],np.searchsorted(STAGES,k)].mean(0) for r in rs] for k in count])
                count_term=(vv[1,0]-vv[0,0]+vv[1,1]-vv[0,1])*.5
                conditional=(vv[0,1]-vv[0,0]+vv[1,1]-vv[1,0])*.5
                total=vv[1,1]-vv[0,0]
                np.testing.assert_allclose(total,count_term+conditional,atol=1e-15)
                np.testing.assert_allclose(total,cells[1]['brier'].mean(0)-cells[0]['brier'].mean(0),atol=1e-12)
                chunks[f,b,d]=np.stack((count_term,conditional,total),axis=-1)
                for z in cells:z.close()
    rng=np.random.default_rng(3409781);ids=2*rng.integers(4,size=(20000,4,8))+np.arange(8)[None,None,:]%2
    interval=lambda x:np.moveaxis(np.quantile(paired_boot(x.mean(2),ids),[.025,.975],axis=0),0,-1)
    res=dict(methods=METHODS,components=['counts','stage_risk','total'],mean=chunks.mean((0,1,2)),
             pointwise95=interval(chunks),mean_stage=stage.mean((0,1,2)),stage_pointwise95=interval(stage),
             changed_counts_fraction=allocation_change.mean((0,1,2)),changed_acquisition_fraction=observation_change.mean((0,1,2)),
             by_start_sign_and_orientation=chunks.reshape(4,4,2,2,3,3).mean((0,1)),
             estimand='Symmetric decomposition on the archived paired deployment risk for each environment; transfer of discrete counts is diagnostic, each cluster shares world IDs, both material orientations and two schedules.')
    np.testing.assert_allclose(res['mean'][:,2],read(ROOT/'primary_stats34.json')['delta'][:3],atol=2e-14)
    write(ROOT/'switch34.json',res)
if __name__=='__main__':main()
