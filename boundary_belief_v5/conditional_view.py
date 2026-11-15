"""Conditional importance proposal for a sharp image observation.

Do not approximate a narrow geometry likelihood by selecting one atom from a
factual bank. Sample geometry conditional on the hypothetical image feature,
sample material from its prior, then weight by factual likelihood. Standard
importance integration, not a new inference algorithm.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from engine import tensor,decode,observation_means,batch,SHIFTS,SIM_COUNTS
from belief import Belief,predict

def truncated(mean,sd,lo,hi):
    c0=torch.special.ndtr((lo-mean)/sd);c1=torch.special.ndtr((hi-mean)/sd)
    mass=(c1-c0).clamp_min(1e-15)
    u=(c0+torch.rand_like(mean)*mass).clamp(1e-14,1-1e-14)
    return mean+sd*torch.special.ndtri(u),mass.log()

def factual_evidence(family,y,seed,n=32768):
    torch.manual_seed(seed);x=torch.rand((n,5),device='cuda',dtype=torch.float64)
    v=observation_means(decode(x[:,:4]),x[:,4]*.006,family,'factual')[:,0]
    return torch.logsumexp(-.5*((v-y)/.025)**2,0)-np.log(n)

def conditional(model,atoms,y_factual,logzf,outer=64,inner=512,seed=0,details=False):
    torch.manual_seed(seed);start=time.perf_counter();before=SIM_COUNTS['trajectories'];theta,b,w=atoms
    mu,cov=model.mean_cov(theta,b,'view');dims=mu.shape[1]
    ids=torch.multinomial(w,outer,replacement=True)
    yy=mu[ids]+torch.randn((outer,dims),device=theta.device,dtype=theta.dtype)@torch.linalg.cholesky(cov).T
    latent=torch.rand((outer,inner,5),device=theta.device,dtype=theta.dtype)
    bias=model.curve[0,:2];logzq=torch.zeros(outer,device=theta.device,dtype=theta.dtype)
    if dims==3:
        grid=(torch.arange(512,device=theta.device,dtype=theta.dtype)+.5)*(.006/512)
        dummy=torch.zeros((512,4),device=theta.device,dtype=theta.dtype)
        curve=model.mean_cov(dummy,grid,'view')[0][:,2]
        llq=-.5*(yy[:,2,None]-curve[None,:])**2/cov[2,2]-.5*torch.log(2*np.pi*cov[2,2])
        logzq=torch.logsumexp(llq,1)-np.log(len(grid))
        bi=torch.multinomial(torch.softmax(llq,1),inner,replacement=True);bb=grid[bi]
        mean_gd=yy[:,None,:2]-bias-(yy[:,2,None]-curve[bi])[:,:,None]*(cov[:2,2]/cov[2,2])
        conditional_cov=cov[:2,:2]-cov[:2,2,None]@cov[None,2,:2]/cov[2,2]
    else:
        bb=latent[:,:,4]*.006;mean_gd=(yy[:,:2]-bias)[:,None,:].expand(-1,inner,-1);conditional_cov=cov[:2,:2]
    gm,logzg=truncated(mean_gd[:,:,0],torch.sqrt(conditional_cov[0,0]),-8.,8.)
    dm_mean=mean_gd[:,:,1]+conditional_cov[1,0]/conditional_cov[0,0]*(gm-mean_gd[:,:,0])
    dm_sd=torch.sqrt(conditional_cov[1,1]-conditional_cov[1,0]**2/conditional_cov[0,0])
    dm,logzd=truncated(dm_mean,dm_sd,22.,34.)
    latent[:,:,0]=(gm+8)/16;latent[:,:,3]=(dm-22)/12;latent[:,:,4]=bb/.006
    tt=decode(latent.reshape(-1,5)[:,:4]);flat_b=bb.reshape(-1)
    vf=observation_means(tt,flat_b,model.family,'factual')[:,0].reshape(outer,inner)
    logw=-.5*((vf-y_factual)/.025)**2+logzg+logzd
    ww=torch.softmax(logw,1);ess=1/(ww*ww).sum(1)
    simulation=batch(tt,flat_b,model.family,SHIFTS)
    if bool(simulation['overflow'].any()):raise RuntimeError('conditional rollout overflow')
    z=simulation['velocity'].reshape(6,outer,inner,2)
    mm=torch.einsum('ok,qokd->qod',ww,z)
    initial=predict(atoms,model.family)['velocity'];current=torch.einsum('qnd,n->qd',initial,w)
    raw=((mm-current[:,None,:])**2).sum(2).mean(1)
    # First-order SNIS Monte Carlo variance correction for squared posterior mean.
    mcvar=(((z-mm[:,:,None,:])**2).sum(3)*(ww**2)[None,:,:]).sum(2).mean(1)
    corrected=(raw-mcvar).clamp_min(0)
    delta=yy-mu[ids]
    true_ll=-.5*torch.einsum('od,dk,ok->o',delta,torch.linalg.inv(cov),delta)-.5*(dims*np.log(2*np.pi)+torch.logdet(cov))
    logpy=logzq-np.log(16*12)+torch.logsumexp(logw,1)-np.log(inner)-logzf
    eig=(true_ll-logpy).mean()
    torch.cuda.synchronize()
    result=dict(gain=corrected.cpu().tolist(),raw_gain=raw.cpu().tolist(),inner_variance=mcvar.cpu().tolist(),eig=float(eig),
        ess_median=float(ess.median()),ess_min=float(ess.min()),outer=outer,inner=inner,seed=seed,
        seconds=time.perf_counter()-start,trajectory_rollouts=SIM_COUNTS['trajectories']-before)
    if details:result['details']=dict(observations=yy.cpu().tolist(),ess=ess.cpu().tolist(),prediction_mean=mm.cpu().tolist())
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',type=int,default=0);p.add_argument('--cases',type=int,nargs='+');p.add_argument('--qualities',nargs='+',default=['high','low','pixel_only']);p.add_argument('--source',default='results/main');p.add_argument('--out',default='results/conditional_view');p.add_argument('--outer',type=int,default=64);p.add_argument('--inner',type=int,default=512);p.add_argument('--wait-inputs',action='store_true');args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    for case in args.cases or range(args.worker,72,2):
        dest=out/f'case_{case:03d}.json'
        if dest.exists():continue
        while args.wait_inputs and not (Path(args.source)/dest.name).exists():time.sleep(2)
        d=json.loads((Path(args.source)/dest.name).read_text());old=np.load(Path(args.source)/f'factual_{case:03d}.npz')
        atoms=tuple(tensor(old[k]) for k in ['theta','rounding','weight']);f=d['family'];yf=d['observations']['factual'][0]
        logzf=factual_evidence(f,yf,44000+case);records={}
        for quality in args.qualities:
            records[quality]={}
            for mode in ['bbox','boundary']:
                records[quality][mode]=conditional(Belief(f,quality,mode),atoms,yf,logzf,args.outer,args.inner,45000+case)
        dest.write_text(json.dumps(dict(case=case,records=records,log_factual_evidence=float(logzf)),indent=2));print('conditional',case,'seconds',sum(r['seconds'] for q in records.values() for r in q.values()),flush=True)
