"""Supplementary numerical check of the bounded analytic likelihood, including tails."""
import numpy as np,torch
from nuisance import Model,tensor,decode,marginal_ll,save,ROOT,meta
x=tensor([[.4,.6,.3,.5,.4]]);mu,cov=Model().mean_cov(decode(x[:,:4]),x[:,4]*.006,'view');Q=torch.linalg.inv(cov);sd=float(torch.rsqrt(Q[1,1]));nodes,w=np.polynomial.legendre.leggauss(512);rows=[]
for center in [21.5,21.96,22.,22.04,28.,33.96,34.,34.04,34.5]:
 y=mu.clone();y[:,1]+=center-28.;ana=float(marginal_ll(y,mu,cov)[0,0]);lo=max(22.,min(34.-12*sd,center-12*sd));hi=min(34.,max(22.+12*sd,center+12*sd));d=tensor((nodes+1)*(hi-lo)/2+lo);mm=mu.repeat(len(d),1);mm[:,1]+=d-28.;r=y-mm;ll=-.5*torch.einsum('nd,dk,nk->n',r,Q,r);num=float(torch.logsumexp(ll+torch.log(tensor(w)*(hi-lo)/24.),0));err=abs(ana-num);assert err<1e-7,(center,err);rows.append(dict(center=center,analytic=ana,quadrature=num,error=err))
save(ROOT/'validation/nuisance_bounds.json',dict(passed=True,max_loglik_error=max(r['error'] for r in rows),rows=rows,**meta()));print('BOUNDED_LIKELIHOOD_VALIDATION_PASSED',flush=True)
