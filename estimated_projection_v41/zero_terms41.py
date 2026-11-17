"""Separate directional alignment from squared correction magnitude, estimated decoder only."""
import json
import numpy as np
import stats41 as s

def main():
    a=np.zeros((s.N,2,3,3,2))
    beta=json.loads((s.ROOT/'results/completion.json').read_text())['beta']
    for n in range(s.N):
        for cell in s.CELLS:
            li=s.LAWS.index(cell.split('_')[-1])
            for b in range(2):
                with np.load(s.block_path(cell,n,b)) as f:
                    p=f['p_base'][2]; z=f['z']; weights=f['weights']
                    choices=f['chosen_optimizer'][:,:,2]; y=f['labels']
                for ri in range(3):
                    base=p
                    k0=np.exp(beta*z*.325)[:,:,None,None]
                    k1=np.exp(beta*z*weights[:,ri])[:,:,None,None]
                    d0=p*(1-p)*(k0-1)/(1+p*(k0-1))
                    d1=p*(1-p)*(k1-1)/(1+p*(k1-1))
                    for pi,step in enumerate([d0,d1,np.where(choices[:,ri,:,:][:,:,None,:],d1,d0)]):
                        a[n,li,ri,pi]+=np.array([np.mean(2*(base-y[:,ri,:,:,None])*step),np.mean(step**2)])/24
    group=a.mean(0); arr=a.mean((1,2)); loss=arr.sum(-1)
    contrast=json.loads((s.ROOT/'no_correction41.json').read_text())
    np.testing.assert_allclose(loss.mean(0),contrast['paired_correction_minus_zero']['mean'][2],atol=1e-12,rtol=0)
    out=dict(laws=s.LAWS,regimes=['strong','weak','null'],method=s.POLICIES,terms=['alignment','squared_step'],
             means=arr.mean(0).tolist(),by_law_regime=group.tolist(),by_law=group.mean(1).tolist())
    (s.ROOT/'zero_terms41.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__': main()
