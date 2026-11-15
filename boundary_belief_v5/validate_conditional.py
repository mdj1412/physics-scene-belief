"""Independent Gaussian rectangle quadrature checks for the view proposal.

Check prior-volume and truncation normalization near both geometry bounds.
This checks the proposal algebra, not convergence of the factual physics term.
"""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.stats import multivariate_normal
from belief import Belief
from conditional_view import truncated

torch.manual_seed(9917)
rows=[]
for quality in ['high','low','pixel_only']:
    for mode in ['bbox','boundary']:
        model=Belief('gate',quality,mode)
        tt=torch.tensor([[.0078,.4,.2,.0222]],device='cuda',dtype=torch.float64)
        yy,cov=model.mean_cov(tt,torch.tensor([.003],device='cuda',dtype=torch.float64),'view')
        yy=yy[0];bias=model.curve[0,:2];n=131072
        if mode=='boundary':
            grid=(torch.arange(512,device='cuda',dtype=torch.float64)+.5)*(.006/512)
            curve=model.mean_cov(torch.zeros((512,4),device='cuda',dtype=torch.float64),grid,'view')[0][:,2]
            ll=-.5*(yy[2]-curve)**2/cov[2,2]-.5*torch.log(2*np.pi*cov[2,2])
            probs=torch.softmax(ll,0);zq=torch.exp(torch.logsumexp(ll,0))/512
            means=yy[:2]-bias-(yy[2]-curve)[:,None]*(cov[:2,2]/cov[2,2])
            cc=cov[:2,:2]-cov[:2,2,None]@cov[None,2,:2]/cov[2,2]
            ids=torch.multinomial(probs,n,replacement=True);sample_means=means[ids]
        else:
            means=(yy[:2]-bias)[None,:];cc=cov;probs=torch.ones(1,device='cuda',dtype=torch.float64);zq=torch.ones((),device='cuda',dtype=torch.float64)
            sample_means=means.expand(n,-1)
        gm,lzg=truncated(sample_means[:,0],torch.sqrt(cc[0,0]),-8.,8.)
        dmmean=sample_means[:,1]+cc[1,0]/cc[0,0]*(gm-sample_means[:,0])
        dm,lzd=truncated(dmmean,torch.sqrt(cc[1,1]-cc[1,0]**2/cc[0,0]),22.,34.)
        weights=torch.exp(lzg+lzd)
        mc=float(zq*weights.mean()/(16*12));se=float(zq*weights.std()/np.sqrt(n)/(16*12))
        c=cc.cpu().numpy()
        masses=np.array([multivariate_normal.cdf([8.,34.],mean=m,cov=c,lower_limit=[-8.,22.],maxpts=100000,abseps=1e-10,releps=1e-10) for m in means.cpu().numpy()])
        exact=float(zq)*float(np.dot(probs.cpu().numpy(),masses))/(16*12)
        relative=abs(mc-exact)/exact
        assert relative<.015,(quality,mode,mc,exact)
        assert bool(((gm>=-8)&(gm<=8)&(dm>=22)&(dm<=34)).all())
        rows.append(dict(quality=quality,mode=mode,mc=mc,quadrature=exact,mc_se=se,relative_error=relative,n=n))
out=Path('results/validation');out.mkdir(parents=True,exist_ok=True)
(out/'conditional_normalization.json').write_text(json.dumps(rows,indent=2))
print('conditional proposal normalization passed',max(r['relative_error'] for r in rows),flush=True)
