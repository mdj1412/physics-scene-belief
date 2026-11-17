"""Zero-shot transfer to a held-out racket rubber. All fitting on an idle CUDA device."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
GROUPS = {'hexer_duro':(1,3,9), 'power_3':(2,4), 'grass_dtecs':(5,6),
          'diamant':(7,), 'blowfish':(8,), 'abs_ii_soft':(10,)}
R = .02
K = 2/3
MUS = [i*.025 for i in range(41)]+[1.1,1.25,1.5,2.]
DT = torch.float64


def tensor(x):
    return torch.as_tensor(x,device='cuda',dtype=DT)


def load(root):
    folds = {}
    hashes = {}
    for split in ('train','test'):
        for config in range(1,11):
            path=root/split/f'racket_{config:02}.csv'
            with path.open(newline='') as f:
                rows=list(csv.DictReader(f))
            def state(keys):
                return tensor([[float(row[k])*scale for k,scale in keys] for row in rows])
            x=state([('v_in_x',1),('v_in_z',1),('spin_before_y',R)])
            y=state([('v_out_x',1),('v_out_z',1),('spin_after_y',R)])
            assert torch.isfinite(x).all() and torch.isfinite(y).all()
            assert all(float(r['v_in_y'])==0 and float(r['v_out_y'])==0 for r in rows)
            folds[split,config]=(x,y,[(r['racket_angle'],r['traj_dir']) for r in rows])
            hashes[f'{split}/{path.name}']=hashlib.sha256(path.read_bytes()).hexdigest()
    return folds,hashes


def design(z):
    return torch.column_stack([torch.ones_like(z[:,0]),z])


def ols(a,b):
    return torch.linalg.lstsq(a,b,driver='gels').solution


def empirical(x,y,tx):
    mean,sd=x.mean(0),x.std(0,correction=0)
    a=design((x-mean)/sd)
    ta=design((tx-mean)/sd)
    side=y[:,0]-y[:,2] > 0
    beta=torch.zeros(4,device=x.device,dtype=x.dtype)
    penalty=torch.diag(tensor([0,1,1,1]))
    for _ in range(35):
        p=torch.sigmoid((a@beta).clamp(-35,35))
        hess=a.T@((p*(1-p))[:,None]*a)+penalty
        step=torch.linalg.solve(hess,a.T@(p-side.double())+penalty@beta)
        beta-=step
        if float(step.norm())<1e-9: break
    branch=ta@beta>0
    experts=[ols(a[side==b],y[side==b]) for b in (False,True)]
    two=torch.where(branch[:,None],ta@experts[1],ta@experts[0])
    coef=ols(a,y)
    return {'two_regime':two,'single_affine':ta@coef},dict(
        mean=mean.tolist(),std=sd.tolist(),logistic=beta.tolist(),
        expert_minus=experts[0].tolist(),expert_plus=experts[1].tolist(),single=coef.tolist(),
        train_positive=int(side.sum()),target_positive=int(branch.sum()))


def rotate(x,deg):
    a=deg*math.pi/180
    c,s=math.cos(a),math.sin(a)
    return c*x[:,0]+s*x[:,1],-s*x[:,0]+c*x[:,1]


def grid_predictions(x,deg,mu,e):
    vt,vn=rotate(x,deg)
    slip=vt-x[:,2]
    # J/m is along the racket tangent and produces the opposing rim velocity change.
    cap=mu[:,None]*(1+e)*(-vn).clamp_min(0)
    impulse=torch.clamp(-slip/(1+1/K),min=-cap,max=cap)
    vtf=vt+impulse
    vnf=torch.where(vn<0,-e*vn,vn)
    angle=deg*math.pi/180
    c,s=math.cos(angle),math.sin(angle)
    vals=torch.stack([c*vtf-s*vnf,s*vtf+c*vnf,x[:,2]-impulse/K],-1)
    return vals


def mechanical(x,y,tx):
    mus=tensor(MUS)
    best={}
    for deg in range(-35,36):
        _,ni=rotate(x,deg)
        if torch.any(ni>=0): continue
        _,no=rotate(y,deg)
        e=-(ni@no)/(ni@ni)
        e=float(e.clamp(0,1))
        yp=grid_predictions(x,deg,mus,e)
        risk=(yp[:,:,[0,2]]-y[:,[0,2]]).square().mean(dim=(1,2))
        i=int(risk.argmin())
        value=(float(risk[i]),abs(deg),deg,MUS[i],e)
        for name,ok in [('inclined',deg>=0),('flat',deg==0),('inclined_signed',True)]:
            if ok and (name not in best or value<best[name]): best[name]=value
    pred,pars={},{}
    for name,(risk,_,deg,mu,e) in best.items():
        yp=grid_predictions(tx,deg,tensor([mu]),e)[0]
        pred[name]=yp
        pars[name]=dict(deg=deg,mu=mu,e=e,train_risk=risk,nonapproach=int((rotate(tx,deg)[1]>=0).sum()))
    # Constant elastic impulse, fitted by the same pair of tangential losses, no rolling clamp.
    s=x[:,0]-x[:,2]
    alpha=float(((-s*(y[:,0]-x[:,0])+s/K*(y[:,2]-x[:,2])).sum()/((1+1/K**2)*(s@s))).clamp_min(0))
    e=pars['flat']['e']; ts=tx[:,0]-tx[:,2]
    pred['elastic_impulse']=torch.stack((tx[:,0]-alpha*ts,-e*tx[:,1],tx[:,2]+alpha*ts/K),1)
    pars['elastic_impulse']=dict(alpha=alpha,e=e)
    return pred,pars


def records(x,y,yp):
    err=yp-y
    v_err=torch.linalg.vector_norm(err[:,:2],dim=1); w_err=abs(err[:,2])/R
    data=[yp[:,0],yp[:,1],yp[:,2]/R,y[:,0],y[:,1],y[:,2]/R,v_err,w_err]
    return torch.column_stack(data).cpu().numpy(),dict(
      n=len(x),velocity_mae=float(v_err.mean()),spin_mae=float(w_err.mean()),
      velocity_rmse=float(v_err.square().mean().sqrt()),spin_rmse=float(w_err.square().mean().sqrt()),
      tangential_mae=float(abs(err[:,0]).mean()),normal_mae=float(abs(err[:,1]).mean()))


def benchmark(data_dir,output):
    assert torch.cuda.is_available()
    data,hashes=load(data_dir)
    report=dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,host=platform.node(),
                slurm=os.getenv('SLURM_JOB_ID'),hashes=hashes,folds={})
    fields=['rubber','config','split','row','traj_dir','racket_angle','method','vx_pred','vz_pred',
            'wy_pred','vx_obs','vz_obs','wy_obs','error_v','error_w']
    with (output/'predictions.csv').open('w') as f:
        writer=csv.writer(f); writer.writerow(fields)
        for rubber,held in GROUPS.items():
            source=[i for i in range(1,11) if i not in held]
            x=torch.cat([data['train',i][0] for i in source])
            y=torch.cat([data['train',i][1] for i in source])
            parts=[(split,i,data[split,i]) for split in ('test','train') for i in held]
            tx=torch.cat([d[0] for _,_,d in parts]); ty=torch.cat([d[1] for _,_,d in parts])
            emp,ep=empirical(x,y,tx); mech,mp=mechanical(x,y,tx)
            spec=tx.clone(); spec[:,1]=-spec[:,1]
            pred={**emp,**mech,'mean':y.mean(0).expand_as(ty),'specular':spec}
            entry=dict(train=len(x),source=source,held=held,physical=mp,empirical=ep,metrics={})
            for m,yp in pred.items():
                ntest=sum(len(d[0]) for split,_,d in parts if split=='test')
                _,a=records(tx[:ntest],ty[:ntest],yp[:ntest]); _,allm=records(tx,ty,yp)
                entry['metrics'][m]=dict(test=a,combined=allm)
                pos=0
                for split,i,(vx,vy,meta) in parts:
                    values,_=records(vx,vy,yp[pos:pos+len(vx)])
                    for row,((angle,traj),v) in enumerate(zip(meta,values)):
                        writer.writerow([rubber,i,split,row,traj,angle,m,*v])
                    pos+=len(vx)
                assert pos==len(yp)
            report['folds'][rubber]=entry
    totals={}
    for part in ('test','combined'):
        totals[part]={}
        for model in next(iter(report['folds'].values()))['metrics']:
            values=[v['metrics'][model][part] for v in report['folds'].values()]
            out={}
            for key in ('velocity_mae','spin_mae','normal_mae','tangential_mae'):
                out[key+'_macro']=float(np.mean([v[key] for v in values]))
                out[key+'_micro']=float(np.average([v[key] for v in values],weights=[v['n'] for v in values]))
            out['n']=sum(v['n'] for v in values); totals[part][model]=out
    report['totals']=totals
    (output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(totals['test'],indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,default=HERE.parent/'data'/'upstream')
    args=p.parse_args()
    benchmark(args.data,HERE)
