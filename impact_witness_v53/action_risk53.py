"""Post-hoc decision-layer diagnostic; v47 Brier coefficients are fixed from v52."""
import json
import os
import numpy as np
from score53 import ROOT,V52,G,quadratic,pair_ci,quantiles,bootstrap
from policy53 import load_model


def optimal(q,model):
    c=model[0]+q[...,None]*(model[1]-model[0])
    assert np.all(c[...,2]>0)
    weight=np.clip(-.5*c[...,1]/c[...,2],0.,1.)
    unsigned=c[...,3]<c[...,0]+weight*(c[...,1]+weight*c[...,2])
    return weight,unsigned


if __name__=='__main__':
    assert os.environ.get('SLURM_JOB_ID')
    ms=['impact_joint','impact_selector','impact_vote_selector','impact_source_opt',
        'impact_source_opt_or_unsigned','reference4_source_opt_or_unsigned']
    model=load_model()
    exact=[];sessions=[];uf=[]
    for i in range(12):
        n=f'panel_{i:02d}.npz'
        with np.load(ROOT/'decisions53'/n) as f, np.load(V52/'risk52'/n) as r:
            names=list(f['methods'].astype(str)); idx=[names.index(s) for s in ms[:3]]
            a=f['alpha'];u=f['unsigned']
            ak,uk=optimal(a[...,idx[0]],model); ar,ur=optimal(a[...,names.index('reference4_joint')],model)
            al=np.concatenate((a[...,idx],np.stack((ak,ak,ar),-1)),-1)
            un=np.concatenate((u[...,idx],np.stack((np.zeros_like(uk),uk,ur),-1)),-1)
            x = quadratic(r['exact'][...,None,None,:],al[None],un[None]).mean(-2)
            s = quadratic(r['realized'][...,None,:],al,un)
            exact.append(x);sessions.append(s.astype(np.float32))
            uf.append(un.mean((0,2,3,4)))
    exact=np.stack(exact)
    sessions=np.stack(sessions)
    vals=((1-G.mean())*exact[:,:,:,0]+G.mean()*exact[:,:,:,1]).mean((2,3,4))
    q=quantiles(sessions,G.mean());boot=bootstrap(sessions)
    def contrast(i,j):
        return dict(mean=pair_ci(vals[:,0,i]-vals[:,0,j],1),
                    p90=[q[i]-q[j],*np.quantile(boot[:,i]-boot[:,j],[.025,.975])])
    out=dict(methods=ms,mean=vals.mean(0)[0],ultra=vals.mean(0)[1],p90=q,
             unsigned_fraction=(np.mean(uf,axis=0)*np.array([[1-G.mean()],[G.mean()]])).sum(0),
             pure_opt_minus_joint=contrast(3,0),opt_unsigned_minus_joint=contrast(4,0),
             opt_unsigned_minus_selector=contrast(4,1),opt_unsigned_minus_vote=contrast(4,2),
             opt_unsigned_mixed_minus_reference4=contrast(4,5),
             scope='Secondary post-hoc decision-layer evaluation. Same fixed v47/v52 source risk and same mixed posterior for primary action comparisons; unadjusted 95% intervals. No target fit or GPU timing.')
    (ROOT/'action_risk53.json').write_text(json.dumps(out,default=lambda a:np.asarray(a).tolist(),indent=2))
    print(out)
