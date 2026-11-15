"""Fresh evaluation with identical input dimension and decoding capacity on two scalar representations."""
from pathlib import Path
import os,json,time
import numpy as np
from scipy.special import logsumexp
from diag34_generate import FAMILIES as FAMS,SCHED,OUT,block,torch
from diag34_model import fit,predict
from diag34_stats import transform,simultaneous,dump


def scalar(x,y,d):
    fy=np.linspace(-.010,.010,9)
    slope=.1*x[:,:,:9]@fy/(fy@fy)
    jx=1.-y[:,:,0]
    jy=1.10*d-y[:,:,1]
    mu=d*(jy+slope*jx)/np.maximum(jx-slope*jy,.35)
    centers=np.array([.112,.240,.262,.389])
    marginal=[]
    for sign in (1,-1):
        expected=centers[None,None,:]*(1+sign*d[:,:,None]*.20)
        marginal.append(logsumexp(-.5*((mu[:,:,None]-expected)/.04)**2,axis=-1))
    return (marginal[0]-marginal[1]).sum(1)


def dataset(split,f,s,j):
    k=(2,4)[s]
    rows=[]; score=[]; labels=[]; starts=[]
    for b in range(8):
        with np.load(OUT/f'heads_{split}_{FAMS[f]}_{b:03d}.npz') as z:
            di=z['d'][:,j,:k]
            for o in range(2):
                rows.extend(scalar(z['x'][:,:k],z['y'][:,o,j,:k],di))
                score.extend(z['scores'][:,o,j,s]);starts.extend(di[:,0]);labels.extend([1-o]*8)
    return np.array([rows,score]).T,np.array(starts),np.array(labels)


def main():
    assert os.environ.get('SLURM_JOB_ID') and not torch.cuda.is_available();torch.set_num_threads(1)
    from detector34 import ActionConditionedDetector
    detector=ActionConditionedDetector(seed=3003913,bank_size=8192)
    t=time.perf_counter()
    with torch.inference_mode():
        for f,fam in enumerate(FAMS):
            for b in range(8):block(f,b,'scalar_test',detector)
            print(fam,round(time.perf_counter()-t,1),flush=True)
    loss=np.zeros((4,8,2,2,2));acc=np.zeros_like(loss)
    for f in range(4):
        for j in range(2):
            for s in range(2):
                tr,dt,yt=dataset('train',3-f,s,j)
                te,de,ye=dataset('scalar_test',f,s,j)
                for h in range(2):
                    xx,ba=transform(tr[:,h:h+1],dt,nonlinear=True)
                    zz,_=transform(te[:,h:h+1],de,basis=ba,nonlinear=True)
                    assert xx.shape[1]==zz.shape[1]==13
                    p=predict(fit(xx,yt),zz)
                    loss[f,:,j,s,h]=((p-ye)**2).reshape(8,2,8).mean((1,2))
                    acc[f,:,j,s,h]=((p>.5)==ye).reshape(8,2,8).mean((1,2))
    main_delta=np.diff(loss.mean((0,1)),axis=-1)[...,0]
    ids=np.random.default_rng(340313).integers(8,size=(16000,4,8))
    draws=np.diff(loss[np.arange(4)[None,:,None],ids].mean((1,2)),axis=-1)[...,0]
    band=simultaneous(main_delta.ravel(),draws.reshape(len(draws),-1)).reshape(2,2,2)
    dump(OUT/'equal_scalar_analysis.json',dict(order=['physics_proxy','frozen_gain'],schedules=SCHED,stages=[2,4],models=13,
        mean=loss.mean((0,1)),acc=acc.mean((0,1)),delta=main_delta,ci=band,elapsed_seconds=time.perf_counter()-t))
if __name__=='__main__':main()
