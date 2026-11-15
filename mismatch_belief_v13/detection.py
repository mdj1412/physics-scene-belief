import numpy as np
import torch
from physics import t,noise,measure,target

def predictive_score(method,state,a,observed):
    """Negative log predictive density in whitened measurement coordinates."""
    _,th=method.points(state);h=measure(th,a)/noise(a)
    mu=torch.einsum('q,bkqd->bkd',method.weights,h);d=h-mu[:,:,None,:]
    S=torch.einsum('q,bkqi,bkqj->bkij',method.weights,d,d)+torch.eye(h.shape[-1],device=h.device,dtype=h.dtype)
    ch=torch.linalg.cholesky(S);res=t(observed)[:,None,:]/noise(a)-mu
    ll=-.5*((res*torch.cholesky_solve(res[:,:,:,None],ch).squeeze(-1)).sum(-1)+2*torch.diagonal(ch,dim1=-2,dim2=-1).log().sum(-1)+h.shape[-1]*np.log(2*np.pi))
    return -torch.logsumexp(state.logp+ll,1)
def pre_score(method,x):
    profile_state=method.update(method.prior(len(x)),4,x[:,:9])
    return predictive_score(method,profile_state,5,x[:,9:])
def base_predict(method,state):
    _,th=method.points(state);return torch.einsum('bk,q,bkqd->bd',state.logp.exp(),method.weights,target(th))
def upper_threshold(scores,alpha=.025):
    a=np.sort(np.asarray(scores));rank=min(len(a),int(np.ceil((len(a)+1)*(1-alpha))))
    return float(a[rank-1])
