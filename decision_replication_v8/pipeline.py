"""Fixed protocol replication. Compute preparation and acquisition never read private/."""
import json,time,hashlib,os
from pathlib import Path
import numpy as np
import torch
from experiment import *

def bank_features(yf,n,s):
 sync();start=time.perf_counter();before=SIM_COUNTS['trajectories']
 x,stat=factual(yf,n,s);theta=decode(x[:,:4]);b=x[:,4]*.006
 pred=batch(theta,b,'corner',SHIFTS);assert not bool(pred['overflow'].any())
 z=pred['velocity'].permute(1,0,2).contiguous();model=Model();mus={}
 for a in ACTIONS:mus[a]=model.mean_cov(theta,b,a)[0]
 sync();elapsed=time.perf_counter()-start
 return x,z,mus,dict(n=n,seed=s,seconds=elapsed,trajectories=SIM_COUNTS['trajectories']-before,smc=stat)

def update(states,z,mus,ys,inv):
 for a in ACTIONS:
  delta=ys[a][:,None,:]-mus[a][None,:,:]
  ll=-.5*torch.einsum('hnd,dk,hnk->hn',delta,inv[a],delta);m=ll.max(1).values;w=torch.exp(ll-m[:,None])
  s=w.sum(1);s2=(w*w).sum(1);v=torch.einsum('hn,nqd->hqd',w,z)
  if a in states:
   old=states[a];mm=torch.maximum(old[0],m);f=torch.exp(old[0]-mm);g=torch.exp(m-mm)
   states[a]=(mm,old[1]*f+s*g,old[2]*f*f+s2*g*g,old[3]*f[:,None,None]+v*g[:,None,None])
  else:states[a]=(m,s,s2,v)

def pack(A,B,current):
 pooled=[];cross=[];pm=[];ess=[]
 for a in ACTIONS:
  aa=A[a];bb=B[a];ma=aa[3]/aa[1][:,None,None];mb=bb[3]/bb[1][:,None,None]
  mm=torch.maximum(aa[0],bb[0]);fa=torch.exp(aa[0]-mm);fb=torch.exp(bb[0]-mm);den=aa[1]*fa+bb[1]*fb
  mean=(aa[3]*fa[:,None,None]+bb[3]*fb[:,None,None])/den[:,None,None]
  pooled.append(((mean-current)**2).sum(-1)/2);cross.append(((ma-current)*(mb-current)).sum(-1)/2);pm.append(mean)
  ess.append(den*den/(aa[2]*fa*fa+bb[2]*fb*fb))
 return dict(standard=torch.stack(pooled,-1).cpu().numpy(),crossfit=torch.stack(cross,-1).cpu().numpy(),mean=torch.stack(pm,-1).cpu().numpy(),ess=torch.stack(ess,-1).cpu().numpy())

def one_pack(A,current):
 mm=torch.stack([A[a][3]/A[a][1][:,None,None] for a in ACTIONS],-1)
 return dict(standard=((mm-current[:,:,None])**2).sum(2).cpu().numpy()/2,mean=mm.cpu().numpy(),ess=torch.stack([A[a][1]**2/A[a][2] for a in ACTIONS],-1).cpu().numpy())

def conditional(case,ys,yf,method,n,moves,reps,tag,initial=None):
 folder=ROOT/tag/f'case_{case:03d}';folder.mkdir(parents=True,exist_ok=True);means=[];total=0.;stats=[]
 for a in ACTIONS:
  arr=[]
  for r in range(reps):
   stem=folder/f'{a}_{r}';sync();t=time.perf_counter()
   x,_,st=sample(physics_likelihood(Model(),tensor(yf),a,tensor(ys[a])),len(ys[a]),n,method,seed(tag,case,a,r),initial=initial,moves=moves)
   su=summaries(x,'corner');sync();elapsed=time.perf_counter()-t;total+=elapsed
   final=np.zeros(len(ys[a]))
   for row in st['history']:final[row['populations']]=row['beta']
   st.update(final_beta=final.tolist(),compute_seconds=elapsed,case=case,action=a,rep=r,**meta());stats.append(st)
   np.savez_compressed(stem.with_suffix('.npz'),**su);save(stem.with_suffix('.json'),st);arr.append(su['mean'])
  means.append(np.stack(arr))
 return np.stack(means,-1),total,stats

def acquire(case):
 world=ROOT/'worlds'/f'case_{case:03d}';folder=ROOT/'acquisition'/world.name;folder.mkdir(parents=True,exist_ok=True)
 pub=json.loads((world/'public.json').read_text());yf=pub['factual'];current=tensor(pub['current']);outer=np.load(world/'outer.npz');ys={a:tensor(outer[a]) for a in ACTIONS}
 model=Model();inv={a:torch.linalg.inv(model.mean_cov(decode(tensor([[.5]*4])),tensor([.003]),a)[1]) for a in ACTIONS}
 # First A bank is shared across comparison arms, but its full standalone cost is charged to both.
 x,z,mu,first=bank_features(yf,4096,seed('stream',case,'A',0));A={};B={}
 sync();t=time.perf_counter();update(A,z,mu,ys,inv);small=one_pack(A,current);sync();update_seconds=time.perf_counter()-t
 used=first['seconds']+update_seconds;bankstats=[dict(stream='A',index=0,**first)];totaln=4096
 np.savez_compressed(folder/'standard4096.npz',**small,seconds=used,n=4096)
 means,bridge_seconds,bridge_stats=conditional(case,{a:outer[a] for a in ACTIONS},yf,'bridge',1024,12,2,'bridge',x)
 t=time.perf_counter();g=((means[0]-current.cpu().numpy()[:,:,None])*(means[1]-current.cpu().numpy()[:,:,None])).sum(2)/2
 bridge_seconds+=time.perf_counter()-t;budget=first['seconds']+bridge_seconds
 np.savez_compressed(folder/'bridge.npz',samples=g,mean=means.mean(0),seconds=budget,n=1024)
 latest=None;latestn=0;latesttime=0.;checkpoints=[];pair_times=[]
 def checkpoint(label):
  nonlocal used,latest,latestn,latesttime
  sync();t=time.perf_counter();p=pack(A,B,current);sync();used+=time.perf_counter()-t
  checkpoints.append(dict(label=label,n=totaln,seconds=used,fraction=used/budget))
  if label:np.savez_compressed(folder/f'{label}.npz',**p,seconds=used,n=totaln)
  if used<=budget:latest=p;latestn=totaln;latesttime=used
 def add(stream,index,n):
  nonlocal used,totaln
  _,zz,mm,st=bank_features(yf,n,seed('stream',case,stream,index));sync();t=time.perf_counter();update(A if stream=='A' else B,zz,mm,ys,inv);sync();ut=time.perf_counter()-t
  used+=st['seconds']+ut;totaln+=n;bankstats.append(dict(stream=stream,index=index,accumulate_seconds=ut,**st))
 add('B',0,4096);checkpoint('fixed8192')
 t0=used;add('A',1,4096);add('B',1,4096);checkpoint('fixed16384');pair_times.append((4096,used-t0))
 index=2
 while used<.95*budget and index<128:
  remaining=budget-used
  # Conservative chunk-size prediction; complete checkpoints are selected only if <= budget.
  prevn,prevtime=pair_times[-1]
  if remaining<max(1.,prevtime*.65*min(1,256/prevn)):break
  n=int(np.clip(prevn*remaining/prevtime*.45,256,65536));n=max(256,(n//256)*256)
  t0=used;add('A',index,n);add('B',index,n);checkpoint(None);pair_times.append((n,used-t0));index+=1
  if used>budget:break
 if latest is None:raise RuntimeError('Fixed SNIS baseline already exceeds bridge budget; protocol cannot claim budget match')
 np.savez_compressed(folder/'budget.npz',**latest,seconds=latesttime,n=latestn)
 choices={};samples={'bridge':g,'standard4096':small['standard']}
 for label in ['fixed8192','fixed16384','budget']:
  p=np.load(folder/f'{label}.npz')
  for estimator in ['standard','crossfit']:samples[f'{estimator}_{label}']=p[estimator]
 for name,v in samples.items():choices[name]=v.mean(0).argmax(1).tolist()
 policy=dict(case=case,choices=choices,bridge_budget=budget,snis_budget_seconds=latesttime,snis_budget_n=latestn,budget_utilization=latesttime/budget,all_snis_compute_seconds=used,checkpoints=checkpoints,**meta())
 save(folder/'bank_stats.json',bankstats);save(folder/'policy.json',policy)
 save(folder/'complete.json',dict(policy_sha256=hashlib.sha256((folder/'policy.json').read_bytes()).hexdigest(),**meta()))
 print(json.dumps(dict(phase='acquisition',case=case,budget=budget,used=latesttime,n=latestn,**meta())),flush=True)

def evaluate(case,n=2048,tag='evaluation'):
 world=ROOT/'worlds'/f'case_{case:03d}';folder=ROOT/tag/world.name
 marker=ROOT/'acquisition'/world.name/'complete.json';assert marker.exists()
 policy=ROOT/'acquisition'/world.name/'policy.json';assert hashlib.sha256(policy.read_bytes()).hexdigest()==json.loads(marker.read_text())['policy_sha256']
 pub=json.loads((world/'public.json').read_text());ys=np.load(world/'private/actual.npz')
 mm,seconds,stats=conditional(case,{a:ys[a] for a in ACTIONS},pub['factual'],'full',n,18,1,tag)
 np.savez_compressed(folder/'predictions.npz',mean=mm[0]);save(folder/'complete.json',dict(seconds=seconds,n=n,policy_sha256=hashlib.sha256(policy.read_bytes()).hexdigest(),**meta()))
 print(json.dumps(dict(phase=tag,case=case,seconds=seconds,**meta())),flush=True)

def reference(case):
 world=ROOT/'worlds'/f'case_{case:03d}';pub=json.loads((world/'public.json').read_text());ys=np.load(world/'outer.npz')
 mm,seconds,stats=conditional(case,{a:ys[a][:128] for a in ACTIONS},pub['factual'],'full',2048,18,2,'reference')
 curr=np.array(pub['current'])[:,:,None];g=((mm[0]-curr)*(mm[1]-curr)).sum(2)/2;folder=ROOT/'reference'/world.name
 np.savez_compressed(folder/'samples.npz',samples=g,mean=mm.mean(0));save(folder/'complete.json',dict(seconds=seconds,**meta()))
 print(json.dumps(dict(phase='reference',case=case,seconds=seconds,**meta())),flush=True)

def main():
 for c in range(N_WORLDS):
  w=ROOT/'worlds'/f'case_{c:03d}'
  if not (w/'complete.json').exists():
   if (ROOT/'worlds'/f'claim_{c:03d}').exists():raise RuntimeError('Preparation claim incomplete; inspect owner before resume')
   make_world(c)
 # All worlds are prepared before primary acquisition starts. All policies freeze before actual decoding.
 for c in range(N_WORLDS):
  if not (ROOT/'acquisition'/f'case_{c:03d}'/'complete.json').exists():acquire(c)
 for c in range(N_WORLDS):
  if not (ROOT/'evaluation'/f'case_{c:03d}'/'complete.json').exists():evaluate(c)
 for c in [0,1,2,3,8,9,10,11]:
  if not (ROOT/'reference'/f'case_{c:03d}'/'complete.json').exists():reference(c)
 for c in range(4):
  if not (ROOT/'decoder_audit'/f'case_{c:03d}'/'complete.json').exists():evaluate(c,4096,'decoder_audit')
 print('PIPELINE_COMPLETE',flush=True)
if __name__=='__main__':main()
