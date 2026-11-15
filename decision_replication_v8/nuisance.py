"""Supplementary pre-outcome analytic nuisance control; public inputs only."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from experiment import *
from pipeline import bank_features,pack

def logdiffcdf(lo,hi):
 positive=lo>0
 a=torch.where(positive,torch.special.log_ndtr(-lo),torch.special.log_ndtr(hi))
 b=torch.where(positive,torch.special.log_ndtr(-hi),torch.special.log_ndtr(lo))
 return a+torch.log(-torch.expm1((b-a).clamp_max(0)))
def marginal_ll(y,mu,cov):
 # mu second coordinate was evaluated at d=28 mm; remaining means do not depend on d.
 keep=[0,2];Q=torch.linalg.inv(cov);res=y[:,None,keep]-mu[None,:,keep]
 cc=y[:,None,1]-(mu[None,:,1]-28.)+torch.einsum('hnd,d->hn',res,Q[1,keep])/Q[1,1]
 inv=torch.linalg.inv(cov[keep][:,keep]);quad=torch.einsum('hnd,dk,hnk->hn',res,inv,res)
 sd=torch.rsqrt(Q[1,1]);return -.5*quad+torch.log(sd*np.sqrt(2*np.pi)/12.)+logdiffcdf((22.-cc)/sd,(34.-cc)/sd)
def accumulate(state,a,ll,z):
 m=ll.max(1).values;w=torch.exp(ll-m[:,None]);s=w.sum(1);s2=(w*w).sum(1);v=torch.einsum('hn,nqd->hqd',w,z)
 if a in state:
  old=state[a];mm=torch.maximum(old[0],m);f=torch.exp(old[0]-mm);g=torch.exp(m-mm)
  state[a]=(mm,old[1]*f+s*g,old[2]*f*f+s2*g*g,old[3]*f[:,None,None]+v*g[:,None,None])
 else:state[a]=(m,s,s2,v)
def validate():
 from engine import observation_means
 rr=np.random.default_rng(seed('nuisance_validation'));x=tensor(rr.random((4096,5)));theta=decode(x[:,:4]);b=x[:,4]*.006;errs=[];base=None
 for d in [.022,.028,.034]:
  tt=theta.clone();tt[:,3]=d;pred=batch(tt,b,'corner',SHIFTS)['velocity'];obs=torch.cat([observation_means(tt,b,'corner',a) for a in ['factual','left','right','center']],1)
  value=torch.cat([pred.reshape(-1),obs.reshape(-1)])
  if base is None:base=value
  else:errs.append(float((value-base).abs().max()))
 assert max(errs)<1e-10,errs
 xx=x[:16].clone();xx[:,3]=.5;mu,cov=Model().mean_cov(decode(xx[:,:4]),xx[:,4]*.006,'view');noise=tensor(rr.normal(size=(16,3)));ys=mu+noise@torch.linalg.cholesky(cov).T;ana=marginal_ll(ys,mu,cov).diag()
 nodes,weights=np.polynomial.legendre.leggauss(128);Q=torch.linalg.inv(cov);numeric=[]
 for i in range(16):
  r=ys[i,[0,2]]-mu[i,[0,2]];center=float(ys[i,1]-(mu[i,1]-28.)+(Q[1,[0,2]]*r).sum()/Q[1,1]);sd=float(torch.rsqrt(Q[1,1]));lo=max(22.,center-12*sd);hi=min(34.,center+12*sd);assert hi>lo
  dd=tensor((nodes+1)*(hi-lo)/2+lo);mm=mu[i].repeat(128,1);mm[:,1]+=dd-28.;delta=ys[i]-mm;ll=-.5*torch.einsum('nd,dk,nk->n',delta,Q,delta)
  numeric.append(float(torch.logsumexp(ll+torch.log(tensor(weights)*(hi-lo)/24.),0)))
 err=float(np.max(np.abs(ana.cpu().numpy()-numeric)));assert err<1e-7,err
 save(ROOT/'validation/nuisance.json',dict(passed=True,d_invariance_max_error=max(errs),quadrature_loglik_max_error=err,prior_samples=4096,**meta()))
 print('NUISANCE_VALIDATION_PASSED',flush=True)
def main():
 validate()
 for c in range(N_WORLDS):
  f=ROOT/'nuisance'/f'case_{c:03d}';f.mkdir(parents=True,exist_ok=True)
  if (f/'complete.json').exists():continue
  w=ROOT/'worlds'/f'case_{c:03d}';p=json.loads((w/'public.json').read_text());outer=np.load(w/'outer.npz');ys={a:tensor(outer[a]) for a in ACTIONS};model=Model();A={};B={};seconds=0.;stats=[]
  for stream,index in [('A',0),('B',0),('A',1),('B',1)]:
   x,z,mus,st=bank_features(p['factual'],4096,seed('stream',c,stream,index));seconds+=st['seconds'];stats.append(st);state=A if stream=='A' else B
   sync();t=time.perf_counter();xx=x.clone();xx[:,3]=.5;view_mu,view_cov=model.mean_cov(decode(xx[:,:4]),xx[:,4]*.006,'view')
   for a in ACTIONS:
    if a=='view':ll=marginal_ll(ys[a],view_mu,view_cov)
    else:
     _,cov=model.mean_cov(decode(x[:1,:4]),x[:1,4]*.006,a);delta=ys[a][:,None,:]-mus[a][None,:,:];ll=-.5*torch.einsum('hnd,dk,hnk->hn',delta,torch.linalg.inv(cov),delta)
    accumulate(state,a,ll,z)
   sync();seconds+=time.perf_counter()-t
  sync();t=time.perf_counter();packed=pack(A,B,tensor(p['current']));sync();seconds+=time.perf_counter()-t
  original=np.load(ROOT/'acquisition'/f'case_{c:03d}'/'fixed16384.npz');error=float(np.max(np.abs(packed['mean'][...,1:]-original['mean'][...,1:])))
  assert error<1e-8,(c,error)
  np.savez_compressed(f/'samples.npz',**packed,seconds=seconds,n=16384)
  policy=dict(case=c,choices={k:packed[k].mean(0).argmax(1).tolist() for k in ['standard','crossfit']},seconds=seconds,dynamic_mean_replay_error=error,**meta());save(f/'policy.json',policy);save(f/'bank_stats.json',stats)
  save(f/'complete.json',dict(policy_sha256=hashlib.sha256((f/'policy.json').read_bytes()).hexdigest(),**meta()));print(json.dumps(dict(phase='nuisance',case=c,seconds=seconds,**meta())),flush=True)
 print('NUISANCE_COMPLETE',flush=True)
if __name__=='__main__':main()
