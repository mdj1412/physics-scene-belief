"""Replay the purchased sensor calibration and factual histories using external PyBullet."""
import argparse
from itertools import product
from pathlib import Path
import json, os, sys, time
import numpy as np
B=Path(__file__).resolve().parent
H=B.parent
sys.path.insert(0,str(H))
import bullet68 as ext
import tapes68 as tape
sim=ext.simulator
pb=sim.b
LAWS=tape.LAWS
SURF=tape.SURF
CELLS=tuple(product(SURF,LAWS))
ENG=dict(horizon=sim.HORIZON,step=2e-5,mesh_spacing_mm=.1,radius_mm=.25)
class SensorReader:
    """Capture the two terminal velocity coordinates already read by v44 before body removal."""
    def __init__(self): self.rows=[]
    def __getattr__(self,name): return getattr(pb,name)
    def getBaseVelocity(self,id):
        ans=pb.getBaseVelocity(id)
        self.rows.append(ans[0])
        return ans

def mu(theta,vt,h,surface,law,sign,reg):
    near=np.select([reg=='strong',reg=='weak'],[1.,.3],default=0.)
    env=near+(1-near)*np.cos(.5*np.pi*np.minimum(abs(h)/23.,1.))**2
    env=np.where(reg=='global_null',0.,env)
    k=np.exp(-.5*((h-22.)/5.5)**2)
    scale,orientation=(1+.079*k,1-1.45*k) if surface=='inlay' else (1+.047*np.sin(2*np.pi*h/24),.38+.62*np.cos(2*np.pi*h/24))
    rate=1+(1 if law=='strengthening' else -1)*.22*np.tanh((abs(vt)-.40)/.19)
    asym=.145*vt**3/(abs(vt)**3+.61**3)
    return theta[:,4]*scale*(1+sign*env*orientation*asym)*rate

def impacts(th,height,speeds,signs,regimes,f,v,diag):
    n=len(th)
    vel=np.empty((4,n,2))
    for speed in np.unique(speeds):
        group=np.flatnonzero(speeds==speed)
        for a in range(0,len(group),64):
            ids=group[a:a+64]; t=th[ids]
            world=sim.Surface(t,height,f,v,ENG['step'],ENG['mesh_spacing_mm'])
            _,site,g=world.support(.25,0.)
            vt=(speed+g)/np.hypot(1,g)
            for ci,(surface,law) in enumerate(CELLS):
                material=mu(t,vt,height+site,surface,law,signs[ids],regimes[ids])
                if not diag['checked_formula']:
                    matched=np.array([ext.new_mu(t[z:z+1,4],np.array([[vt[z]]]),np.array([height+site[z]]),surface,law,signs[ids[z:z+1]],regimes[ids[z:z+1]]).item() for z in range(len(ids))])
                    np.testing.assert_allclose(material,matched,atol=1e-13,rtol=1e-13)
                    diag['checked_formula']=True
                sim.b.rows.clear()
                raw=world.simulate(.25,0.,material,t[:,5],velocity=float(speed))
                xyz=np.asarray(sim.b.rows)/sim.SCALE
                assert xyz.shape==(len(ids),3) and np.isfinite(xyz).all()
                # A genuine missed contact still yields its ballistic outgoing velocity;
                # no rows are silently dropped from calibration or observation selection.
                diag['shots']+=len(ids);diag['unloaded']+=int(np.sum(~raw['valid']))
                diag['loaded_end']+=int(np.sum(raw['loaded_end']))
                vel[ci,ids]=np.column_stack((-xyz[:,2],xyz[:,0]))
    return vel

def build(k):
    then=time.monotonic()
    th,ct,signs,regs,ecx,ecy,ex,eq,ep,dirs=tape.base(k)
    sx,sq,sp=(tape.w.w.sigmas()[i].numpy() for i in (0,tape.w.Q,tape.w.P))
    out=B/'tapes'/tape.panel(k);out.mkdir(exist_ok=True,parents=True)
    diag=dict(checked_formula=False,shots=0,unloaded=0,loaded_end=0)
    def prefix(t,scale):return np.repeat(t,scale)
    hardware={}
    for f,v in product(tape.FAMS,tape.VAR):
        # Cache origin contacts only when all vertices, the commanded height and the
        # sampled contour on the old mesh are identical for the full 10 mm scene.
        probe_grid=np.linspace(-5.,5.,101)
        shape=np.array(ext.profile(probe_grid,f,v)[0],dtype='f8')
        key=shape.tobytes()
        if key not in hardware:
            factual_th=np.concatenate([th.reshape(-1,6),ct.reshape(-1,6)])
            sg=np.r_[prefix(signs,128),prefix(signs,4)]
            vfact=impacts(factual_th,0.,np.full(len(sg),tape.w.w.FACTUAL_INCOMING),sg,np.full(len(sg),'global_null'),f,v,diag)
            probe=impacts(th[:,:8].reshape(-1,6),0.,np.full(32,tape.w.w.PROBE_INCOMING),prefix(signs,8),prefix(regs,8),f,v,diag)
            far=impacts(ct[:,:3].reshape(-1,6),0.,dirs[:,:3].reshape(-1)*abs(tape.w.w.PROBE_INCOMING),prefix(signs,3),prefix(regs,3),f,v,diag)
            hardware[key]=vfact,probe,far
        vfact,probe,far=hardware[key]
        near=impacts(ct[:,3].reshape(-1,6),tape.w.w.MATCH_HEIGHT*1000,np.full(4,tape.w.w.MATCH_SPEED),signs,regs,f,v,diag)
        for ci,(s,law) in enumerate(CELLS):
            fn=tape.name(f,s,law,v,19)
            # Prescribed tapes supply only the shared geometric channels, never any
            # contact velocity. New additive measurement errors use original tapes' seeds.
            with np.load(H/'tapes'/'prescribed'/tape.panel(k)/fn) as z:
                x=z['xx'];cx=z['cx'];np.testing.assert_array_equal(z['dirs'],dirs)
            x[...,9:]=vfact[ci,:512].reshape(4,128,2)+ex[...,9:]*sx[9:]
            cx[...,9:]=vfact[ci,512:].reshape(4,4,2)+ecx[...,9:]*sx[9:]
            cy=np.concatenate((far[ci].reshape(4,3,2),near[ci].reshape(4,1,2)),axis=1)+ecy*sp
            paid=probe[ci].reshape(4,8,1,2)+ep[:,:8]*sp
            np.savez(out/fn,xx=x,cx=cx,cy=cy,dirs=dirs,probe8=paid)
    diag.update(unique_origin_meshes=len(hardware),seconds=time.monotonic()-then,horizon=ENG['horizon'],objects_per_selection=128,objects_per_score=8)
    (out/'diagnostics.json').write_text(json.dumps(diag,indent=2))
    print(k,diag,flush=True)
if __name__=='__main__':
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    tape.locked_run.configure();tape.w.torch.set_num_threads(1)
    pb.connect(pb.DIRECT);sim.b=SensorReader()
    for k in map(int,sys.argv[1:]):build(k)
    pb.disconnect()
