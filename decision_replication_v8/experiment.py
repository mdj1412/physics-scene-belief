"""Independent corner replication; policy code never reads private evidence."""
import hashlib,json,os,socket,time
from pathlib import Path
import numpy as np
import torch
from engine import tensor,decode,batch,SHIFTS,SIM_COUNTS
from bridge import sample,physics_likelihood,summaries
from belief import Belief
ROOT=Path('results');N_WORLDS=64;H=512;R=32
ACTIONS=['view','left','right','center']
class Model(Belief):
 def __init__(self):super().__init__('corner','pixel_only','boundary')
 def mean_cov(self,theta,b,action):
  mu,cov=super().mean_cov(theta,b,action)
  return mu,cov*.25**2 if action=='view' else cov

def save(path,obj):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj));tmp.replace(p)
def seed(*parts):return int(hashlib.sha256(json.dumps(['replication_v8',*parts]).encode()).hexdigest()[:8],16)
def meta():return dict(job=os.environ.get('SLURM_JOB_ID'),host=socket.gethostname(),pid=os.getpid(),time=time.time())
def sync():torch.cuda.synchronize()
def factual(yf,n,s):
 model=Model()
 def like(x,ids):
  flat=x.reshape(-1,5);mu,_=model.mean_cov(decode(flat[:,:4]),flat[:,4]*.006,'factual');ll=-.5*((mu[:,0]-float(yf[0]))/.025)**2
  return ll.reshape(x.shape[:2]),torch.zeros(x.shape[:2],device=x.device,dtype=x.dtype)
 x,_,stat=sample(like,1,n,'full',s,moves=18)
 return x[0],stat

def make_world(case):
 folder=ROOT/'worlds'/f'case_{case:03d}';folder.mkdir(parents=True,exist_ok=True)
 rng=np.random.default_rng(seed('truth',case));x=tensor(rng.random((1,5)));model=Model();theta=decode(x[:,:4]);b=x[:,4]*.006
 mf,_=model.mean_cov(theta,b,'factual');yf=(mf.cpu().numpy()[0]+rng.normal(0,.025,1)).tolist()
 truth=batch(theta,b,'corner',SHIFTS);assert not bool(truth['overflow'].any())
 save(folder/'private/truth.json',dict(x=x.cpu().tolist(),velocity=truth['velocity'][:,0].cpu().tolist(),world_seed=seed('truth',case)))
 noise=tensor(np.random.default_rng(seed('actual_noise',case)).normal(size=(R,3)));actual={}
 for a in ACTIONS:
  mu,cov=model.mean_cov(theta,b,a);actual[a]=(mu+noise[:,:mu.shape[1]]@torch.linalg.cholesky(cov).T).cpu().numpy()
 np.savez_compressed(folder/'private/actual.npz',**actual)
 gen,gs=factual(yf,16384,seed('generator',case));current=summaries(gen[None],'corner')['mean'][0]
 rr=np.random.default_rng(seed('outer',case));xx=gen[torch.as_tensor(rr.integers(len(gen),size=H),device='cuda')];noise=tensor(rr.normal(size=(H,3)));outer={}
 for a in ACTIONS:
  mu,cov=model.mean_cov(decode(xx[:,:4]),xx[:,4]*.006,a);outer[a]=(mu+noise[:,:mu.shape[1]]@torch.linalg.cholesky(cov).T).cpu().numpy()
 np.savez_compressed(folder/'outer.npz',**outer);save(folder/'generator_stat.json',gs)
 save(folder/'public.json',dict(case=case,family='corner',factual=yf,current=current.tolist(),world_seed=seed('truth',case)))
 save(folder/'complete.json',meta());print(json.dumps(dict(phase='world',case=case,**meta())),flush=True)

def prepare(seconds):
 deadline=time.monotonic()+seconds;out=ROOT/'worlds';out.mkdir(exist_ok=True)
 for c in range(N_WORLDS):
  if time.monotonic()>deadline-60:break
  f=out/f'case_{c:03d}'
  if (f/'complete.json').exists():continue
  lock=out/f'claim_{c:03d}'
  try:lock.mkdir()
  except FileExistsError:continue
  save(lock/'worker.json',meta());make_world(c)
 print('PREPARATION_COMPLETE',flush=True)
if __name__=='__main__':
 import sys
 prepare(float(sys.argv[1]) if len(sys.argv)>1 else 300)
