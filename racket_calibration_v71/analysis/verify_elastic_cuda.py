"""Independent CUDA replay of the source-prior elastic response and all paired draws."""
import csv
import json
import os
import platform
import subprocess
import numpy as np
import torch
from elastic import HERE, GROUPS, BUDGETS, read, train_tape
def run():
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
    procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    assert len(gpu.splitlines())==1 and not procs and int(gpu.split(',')[-1])==0 and int(gpu.split(',')[-2])<50
    t=lambda x: torch.as_tensor(x,device='cuda',dtype=torch.float64)
    def getdata(cfgs,split):return tuple(map(t,read(cfgs,split)))
    def coords(x):return torch.stack([x[:,1].abs(),(x[:,0]-x[:,2]).abs()],dim=1)
    def hmat(x,center,scale):
        z=(coords(x)-center)/scale;s=x[:,0]-x[:,2]
        H=x.new_zeros((len(x),3,4))
        H[:,0,2]=-s;H[:,0,3]=-s*z[:,1]
        H[:,2,2]=1.5*s;H[:,2,3]=1.5*s*z[:,1]
        H[:,1,0]=-x[:,1];H[:,1,1]=-x[:,1]*z[:,0]
        return H
    with (HERE/'elastic_draws.csv').open() as f:
        original={(r['rubber'],int(r['budget']),int(r['draw'])):(float(r['velocity']),float(r['spin'])) for r in csv.DictReader(f)}
    worst=np.zeros(2); count=0
    for g,spec in GROUPS.items():
        src=[getdata(s['held'],'train') for n,s in GROUPS.items() if n!=g]
        c=torch.cat([coords(x) for x,y in src]); center=c.mean(0);scale=c.std(0,unbiased=False)
        mask=t([1,0,1]);theta=[];sig=[]
        for x,y in src:
            h=hmat(x,center,scale);u=y-x*mask
            b=torch.linalg.lstsq(h.reshape(-1,4),u.reshape(-1,1)).solution[:,0]
            e=u-h@b; theta.append(b);sig.append(e.T@e/len(e))
        pop=torch.stack(theta); mu=pop.mean(0);s0=torch.cov(pop.T)+.0004*t(np.eye(4))
        vn=torch.stack(sig).mean(0)+.0025*t(np.eye(3))
        x,y=getdata(spec['held'],'train');xx,yy=getdata(spec['held'],'test')
        h=hmat(x,center,scale);j=h.mT@torch.linalg.inv(vn)
        a=j@h;v=j@(y-x*mask-h@mu).unsqueeze(-1)
        tape=torch.tensor(train_tape(len(x),min(spec['held'])),device='cuda')
        zz=(coords(xx)-center)/scale;slip=xx[:,0]-xx[:,2]
        for k in BUDGETS:
            idx=tape[:,:k]
            b=(torch.linalg.solve(torch.linalg.inv(s0)+a[idx].sum(1),v[idx].sum(1))).squeeze(-1)+mu
            alpha=(b[:,2,None]+b[:,3,None]*zz[:,1]).clamp(0,.8); e=(b[:,0,None]+b[:,1,None]*zz[:,0]).clamp(0,1)
            velocity=torch.hypot(xx[:,0]-alpha*slip-yy[:,0],-e*xx[:,1]-yy[:,1]).mean(1)
            spin=(xx[:,2]+1.5*alpha*slip-yy[:,2]).abs().mean(1)/.02
            res=torch.stack([velocity,spin],dim=1).cpu().numpy()
            err=abs(res-np.array([original[g,k,d] for d in range(256)])).max(0)
            worst=np.maximum(worst,err); count+=len(res)
    assert worst.max()<1e-8
    (HERE/'elastic_cuda.json').write_text(json.dumps(dict(max_abs_error=worst.tolist(),draw_budget_product_count=count,node=platform.node(),job=os.environ.get('SLURM_JOB_ID'),gpu_before=gpu,compute_process_before=procs),indent=2))
if __name__=='__main__':run()
