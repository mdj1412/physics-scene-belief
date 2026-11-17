"""Matched four-coefficient thin-shell model: compare two source-fitted slip features."""
from pathlib import Path
import os, sys, json, platform
import numpy as np
import torch

D = Path(__file__).resolve().parent
PROJECT=D.parents[1]
sys.path.insert(0,str(PROJECT/'racket_external_v72/method'))
import bench

def features(x, ratio):
    normal=x[:,1].abs()
    slip=(x[:,0]-x[:,2]).abs()
    if ratio:
        if (normal < 1e-8).any():raise ValueError('zero normal speed')
        slip=slip/normal
    return torch.stack([normal,slip],1)

def transform(x,center,sd,ratio):
    z=(features(x,ratio)-center)/sd
    slip=x[:,0]-x[:,2]
    h=x.new_zeros(len(x),3,4)
    h[:,0,2]=-slip
    h[:,0,3]=-slip*z[:,1]
    h[:,1,0]=-x[:,1]
    h[:,1,1]=-x[:,1]*z[:,0]
    h[:,2,2]=1.5*slip
    h[:,2,3]=1.5*slip*z[:,1]
    return h,z

def prior(src,ratio):
    states=torch.cat([features(x,ratio) for x,_ in src])
    center=states.mean(0)
    sd=states.std(0,correction=0)
    coefs=[]; noises=[]
    for x,y in src:
        h,_=transform(x,center,sd,ratio)
        target=y-x*x.new_tensor([1,0,1])
        theta=torch.linalg.lstsq(h.reshape(-1,4),target.flatten(),driver='gels').solution
        residual=target-h@theta
        coefs.append(theta)
        noises.append(residual.T@residual/len(x))
    betas=torch.stack(coefs)
    precision=torch.linalg.inv(torch.cov(betas.T)+.02**2*torch.eye(4,device=betas.device))
    noise=torch.linalg.inv(torch.stack(noises).mean(0)+.05**2*torch.eye(3,device=betas.device))
    return center,sd,betas.mean(0),precision,noise

def predictions(src,x,y,tape,tx,ratio):
    center,sd,base,precision,noise=prior(src,ratio)
    h,_=transform(x,center,sd,ratio)
    _,z=transform(tx,center,sd,ratio)
    jt=h.mT@noise
    gram=jt@h
    rhs=jt@(y-x*x.new_tensor([1,0,1])-h@base).unsqueeze(-1)
    slip=tx[:,0]-tx[:,2]
    for k in (4,8):
        ids=tape[:,:k]
        beta=base+torch.linalg.solve(precision+gram[ids].sum(1),rhs[ids].sum(1)).squeeze(-1)
        e=(beta[:,0,None]+beta[:,1,None]*z[:,0]).clamp(0,1)
        a=(beta[:,2,None]+beta[:,3,None]*z[:,1]).clamp(0,.8)
        yield k,torch.stack([tx[:,0]-a*slip,-e*tx[:,1],tx[:,2]+1.5*a*slip],-1)

def main():
    gpu=bench.idle_device()
    torch.set_default_dtype(torch.float64)
    ref=PROJECT/'racket_video_v73/gpu/control'
    frozen=json.loads((ref/'run.json').read_text())
    data,_=bench.compare.old.load(bench.V70/'data/upstream')
    source=[bench.compare.xy(data,'train',bench.compare.GROUPS[g]) for g in frozen['source']]
    x,y=data['train',10][:2];tx,ty=data['test',10][:2]
    tape=torch.as_tensor(np.load(ref/'splits.npz')['balanced'],device=x.device)
    oldbank=np.load(ref/'predictions.npz')
    out={};checks={}
    for name,ratio in [('v73_absolute',False),('speed_ratio',True)]:
        for k,p in predictions(source,x,y,tape,tx,ratio):
            key=f'{name}_k{k}'
            if not ratio:
                err=float(abs(p-bench.as_cuda(oldbank[f'elastic_impulse_k{k}'][:,:len(tx)])).max())
                assert err<1e-9,err
                checks[key]=err
            out[key]={}
            group_fast=abs(tx[:,1])>=frozen['fast_cutoff_mps']
            for group,mask in [('all',torch.ones_like(group_fast)),('fast',group_fast)]:
                ok=mask & ((ty[:,0]-ty[:,2]).abs()>.2)
                sign=bench.release(p)[:,ok]==bench.release(ty)[ok]
                v,w=bench.errors(p,ty)
                out[key][group]=dict(n=int(mask.sum()),resolved=int(ok.sum()),accuracy=float(sign.double().mean()),velocity=float(v[:,mask].mean()),spin=float(w[:,mask].mean()))
    meta=dict(groups=out,checks=checks,host=platform.node(),gpu=gpu,torch=torch.__version__,job=os.environ.get('SLURM_JOB_ID'),step=os.environ.get('SLURM_STEP_ID'))
    (D/'result.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(meta)
if __name__=='__main__':main()
