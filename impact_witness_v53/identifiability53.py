"""Catalog-averaged diagnostics for observing one impact sign, with no deployment labels."""
import json
import numpy as np
from policy53 import ROOT,METHODS,EPS


def auc(a,b,w):
    """Probability an e=.5 session outranks an e=0 session, ties half."""
    def one(x,y):
        x=np.sort(np.round(x.ravel(),12)); y=np.round(y.ravel(),12)
        return (np.searchsorted(x,y,'left')+np.searchsorted(x,y,'right')).mean()/(2*len(x))
    return sum(w[i]*w[j]*one(a[i],b[j]) for i,j in np.ndindex(2,2))


def main():
    w=[.5625,.4375]  # equal averaging over g in (0,.25,.5,1)
    auc_all=[]; auc_active=[]; err=[]; delta=[]; active_by=[]
    for i in range(12):
        with np.load(ROOT/'decisions53'/f'panel_{i:02d}.npz') as f:
            q=f['alpha'][...,0]; masked=f['alpha'][...,METHODS.index('impact_joint_masked')]
            e=f['post_e']; s=f['impact_sign']
        auc_all.append(auc(e[0],e[-1],w))
        auc_active.append(auc([x[y!=0] for x,y in zip(e[0],s[0])],
                                [x[y!=0] for x,y in zip(e[-1],s[-1])],w))
        ref=np.arange(5)*.25
        k=np.stack([ref,1-ref])
        truth=np.stack([ei+(1-2*ei)*k for ei in EPS])[...,None,None]
        pool=lambda a: np.einsum('ij,j->i',a.mean(axis=(2,3,4)),w)
        err.append([pool((q-truth)**2),pool((masked-truth)**2)])
        active_by.append(pool(s!=0))
        delta.append(pool(abs(q-masked))/pool(s!=0))
    out=dict(eps=list(EPS),mean_catalog_auc_e0_vs_e50=float(np.mean(auc_all)),
             active_only_auc=float(np.mean(auc_active)),target_q_mse=np.mean(err,0).tolist(),
             mean_abs_q_update_given_active=np.mean(delta,0).tolist(),
             scope='equal catalogs; two g-mixture independent session draws per AUROC comparison; same estimation settings.')
    (ROOT/'identifiability53.json').write_text(json.dumps(out,indent=2))
if __name__=='__main__':
    main()
