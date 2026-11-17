"""Generate sensor potentials on CPU; only masked arrays reach the policy."""
import argparse
import hashlib
import itertools
import os
import sys
from pathlib import Path
import numpy as np

H=Path(__file__).resolve().parent
P=H.parent
LOCK=P/'invariance_audit_v66'/'locked'
sys.path[:0]=[str(LOCK),str(P/'blind_decoder_v43')]
import locked_run
import world43 as w
from mechanics68 import simulate

FAMS=('wendland','dispersive')
SURF=('inlay','woven')
LAWS=('strengthening','weakening')
VAR=('base','hollow','bulge')
HGT=(19,31)

def panel(k):return f'panel_{k:02d}'
def name(f,s,l,v,h):return f'{f}_{s}_{l}_{v}_h{h}.npz'

def base(k):
    with np.load(LOCK/'results'/'worlds'/(panel(k)+'.npz')) as z:
        theta=z['theta']; signs=z['signs']; regs=z['regimes']
        alphas=z['alphas'];seed=int(z['seed']);support=int(z['support_panel'])
    caltheta=[];eps_cal_x=[];eps_cal_y=[];eps_x=[];eps_q=[];eps_p=[];directions=[]
    for e in range(4):
        cs=seed+100+e;ds=seed+200+e
        ids=w.w.stream(cs,98).permutation(1024)[:4]
        ctx=('wendland','base',.019,'inlay','strengthening',int(signs[e]),support)
        caltheta.append(w.draw_object(1024,cs,10,alphas[e],ctx)[ids].numpy())
        eps_cal_x.append(w.w.stream(cs,11).normal(size=(1024,11))[ids])
        eps_cal_y.append(w.w.stream(cs,99).normal(size=(4,2)))
        eps_x.append(w.w.stream(ds,21).normal(size=(128,11)))
        eps_q.append(w.w.stream(ds,40+w.Q).normal(size=(128,4,7)))
        eps_p.append(w.w.stream(ds,40+w.P).normal(size=(128,4,2)))
        directions.append(w.w.directions('alternating',(-1,1)[k%2],4));directions[-1][-1]=1
    return (theta,np.array(caltheta),signs,regs,*map(np.array,(eps_cal_x,eps_cal_y,eps_x,eps_q,eps_p,directions)))

def build(k,backend):
    # Only generator receives names / prototype; output is exposed to predictors by field.
    theta,ct,signs,regs,ecx,ecy,ex,eq,ep,dirs=base(k)
    sx,sq,sp=(w.w.sigmas()[i].cpu().numpy() for i in (0,w.Q,w.P))
    root=H/'tapes'/backend/panel(k);root.mkdir(exist_ok=True,parents=True)
    for f,v in itertools.product(FAMS,VAR):
        ctx=(f,v,.019,'inlay','strengthening',1,980+k)
        vis=lambda ts,hs: w.contour(w.t(ts.reshape(-1,6)),hs,ctx).cpu().numpy().reshape(*ts.shape[:-1],-1)
        fcal=vis(ct,w.w.FY);fdep=vis(theta,w.w.FY)
        views={h:vis(theta,w.w.view_heights(h/1000))[:,:,None]+eq*sq for h in HGT}
        for s,l in itertools.product(SURF,LAWS):
            x=np.empty((4,128,11));prior=np.empty((4,4,11))
            measured=np.empty((4,4,2));probe=np.empty((4,128,4,2))
            for e in range(4):
                def collide(th,h,speed,reg):
                    return simulate(th,h,speed,f,v,s,l,int(signs[e]),reg,radius_mm=.25,backend=backend)[0]
                vel=collide(theta[e],0,w.w.FACTUAL_INCOMING,'global_null')
                x[e]=np.column_stack([fdep[e],vel])+ex[e]*sx
                vcal=collide(ct[e],0,w.w.FACTUAL_INCOMING,'global_null')
                prior[e]=np.column_stack([fcal[e],vcal])+ecx[e]*sx
                measured[e,:3]=collide(ct[e,:3],0,dirs[e,:3]*abs(w.w.PROBE_INCOMING),str(regs[e]))
                measured[e,3:]=collide(ct[e,3:],w.w.MATCH_HEIGHT*1000,w.w.MATCH_SPEED,str(regs[e]))
                measured[e]+=ecy[e]*sp
                pv=collide(theta[e],0,w.w.PROBE_INCOMING,str(regs[e]))
                probe[e]=pv[:,None]+ep[e]*sp
            for h,view in views.items():
                # Geometry-only fields must be bitwise shared across the new laws and match archive.
                with np.load(LOCK/'results'/panel(k)/name(f,s,'rational',v,h)) as z:
                    np.testing.assert_allclose(z['xx'][...,:9],x[...,:9],rtol=0,atol=2e-15)
                    archived=z['query_paid'];finite=np.isfinite(archived)
                    np.testing.assert_allclose(archived[finite],view[finite],rtol=0,atol=2e-15)
                np.savez(root/name(f,s,l,v,h),xx=x,query=view,probe=probe,cx=prior,cy=measured,dirs=dirs)
    print('tape',backend,k,flush=True)

if __name__=='__main__':
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    locked_run.configure();w.torch.set_num_threads(1)
    pa=argparse.ArgumentParser()
    pa.add_argument('backend',choices=['prescribed','compliant']);pa.add_argument('panels',nargs='+',type=int)
    args=pa.parse_args()
    for k in args.panels:build(k,args.backend)
