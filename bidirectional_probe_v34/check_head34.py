"""Independent check that the paid-only callback actually selects the held-out sequential policy."""
from pathlib import Path
import json,numpy as np
from direct_head34 import load_artifact,score_direct34,SCHEDULES,FAMILIES
ROOT=Path(__file__).resolve().parent
def main():
    artifact=load_artifact();observed=np.empty((4,8,2,2));timed=0;budget=0
    for f,fam in enumerate(FAMILIES):
        for b in range(8):
            for d,ori in enumerate(('plus','minus')):
                for s,sc in enumerate(SCHEDULES):
                    p=ROOT/'results_policy'/(fam+'_'+ori)/f'block_{b:03d}'/sc
                    with np.load(p/'metrics.npz') as m,np.load(p/'choices.npz') as ch:
                        n=np.full(8,2);pool=list(range(8))
                        for st,k in enumerate([2,4]):
                            def key(e):
                                v=score_direct34(ch['features'][e,st],ch['paid_x'][e,:k],ch['paid_y'][e,:k],
                                      ch['paid_direction'][e,:k],sc,k,f,artifact)
                                return (-v,e)
                            pool=sorted(sorted(pool,key=key)[:len(pool)//2])
                            n[pool]=k*2
                        chosen=ch['gaps'][np.arange(8),np.searchsorted([2,4,8],n)]
                        ids=np.argsort(-chosen,axis=-1,kind='stable')[:,:13]
                        actions=np.zeros((8,128),bool);np.put_along_axis(actions,ids,True,axis=1)
                        probs=m['probabilities'];loss=((probs-m['labels'][:,:,None,None])**2).mean(-1)
                        direct=np.where(actions,loss[:,:,1],loss[:,:,0]).mean(1)
                        observed[f,b,d,s]=direct.mean()
                        np.testing.assert_allclose(direct,m['oracle_stage_risk'][np.arange(8),np.searchsorted([2,4,8],n)],rtol=0,atol=1e-12)
                        assert n.sum()==32 and actions.sum()==104;budget+=1
                        if b<2:
                            with np.load(ROOT/'results_head_time'/(fam+'_'+ori)/f'block_{b:03d}'/'timing.npz') as t:
                                np.testing.assert_array_equal(n,t['counts'][s,:,0])
                                np.testing.assert_allclose(direct,t['brier'][s,:,0],rtol=0,atol=1e-12);timed+=1
    expected=np.asarray(json.loads((ROOT/'decision_stats34.json').read_text())['means'])[:,4]
    v=observed.mean((0,1,2));np.testing.assert_allclose(v,expected,atol=1e-13,rtol=0)
    (ROOT/'independent_audit34.json').write_text(json.dumps(dict(replayed_records=budget,real_gpu_timing_record_matches=timed,
        recovered_brier=v.tolist(),max_abs_error=float(max(abs(v-expected))),model_target_sign_feature=False),indent=2)+'\n')
if __name__=='__main__':main()
