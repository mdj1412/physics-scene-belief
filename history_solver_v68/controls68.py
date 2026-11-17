"""Prelabel tests: closed-form plane inclines, sign symmetry, sensor coordinates, convergence."""
import json
import os
from pathlib import Path
import numpy as np
from mechanics68 import simulate
H=Path(__file__).resolve().parent

def main():
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    rng=np.random.default_rng(680631)
    th=np.zeros((320,6))
    th[:,2]=rng.uniform(-.28,.28,len(th)); th[:,4]=np.tile([0.,.035,1.,.3],len(th)//4)
    th[:,5]=rng.uniform(.28,.90,len(th)); vel=rng.uniform(-.85,.85,len(th))
    g=th[:,2]; norm=np.hypot(1.,g); vn=(1.-g*vel)/norm; vt=(vel+g)/norm
    post_n=th[:,5]*vn
    post_t=np.sign(vt)*np.maximum(abs(vt)-th[:,4]*(1.+th[:,5])*vn,0.)
    expected=np.stack([-(post_n-g*post_t)/norm,(post_t+g*post_n)/norm],-1)
    out={}
    for backend in ('prescribed','compliant'):
        for fine in (False,True):
            z,sl,d=simulate(th,19,vel,'flat','base','flat','constant',0,'strong',2,backend,fine,True)
            inv=abs((z[:,1]+g*z[:,0])/norm)
            np.testing.assert_allclose(inv,sl,atol=1e-13)
            reverse=th.copy();reverse[:,2]*=-1
            rz,rsl=simulate(reverse,19,-vel,'flat','base','flat','constant',0,'strong',2,backend,fine)
            mirror_err=max(np.max(abs(rz-z*np.array([1,-1]))),np.max(abs(rsl-sl)))
            energy=.5*(z*z).sum(1)-.5*(vel*vel+1)
            record=dict(velocity_max=float(abs(z-expected).max()),velocity_mean=float(abs(z-expected).mean()),
               slip_max=float(abs(sl-abs(post_t)).max()),relative_normal_impulse_max=float(abs(d['normal_impulse']/((1.+th[:,5])*vn)-1).max()),
               mirror_max=float(mirror_err),max_kinetic_energy_change=float(energy.max()),
               perfect_stick_max_slip=float(np.max(sl[th[:,4]==1.])),
               frictionless_tangent_error=float(np.max(abs(sl[th[:,4]==0.]-abs(vt[th[:,4]==0.])))))
            assert mirror_err<1e-12 and energy.max()<0
            assert record['velocity_max']<.035 and record['perfect_stick_max_slip']<1e-10
            assert record['frictionless_tangent_error']<1e-12
            out[backend+('_fine' if fine else '')]=record
    assert out['compliant_fine']['velocity_mean']<out['compliant']['velocity_mean']
    import sys
    sys.path.insert(0,str(H.parent/'law_axis_v67'))
    import score67 as ref
    from locked_geometry import profile
    nr=48
    slopes=rng.normal(0,.09,nr); bends=rng.uniform(-.18,.25,nr)
    t=np.column_stack([np.zeros(nr),np.zeros(nr),slopes,bends,rng.uniform(.12,.65,nr),rng.uniform(.30,.85,nr)])
    sign=rng.choice([-1,1],nr); reg=rng.choice(['null','weak','strong'],nr)
    err=0.
    for law in ('strengthening','weakening'):
        for r in (.25,2.):
            for f in ('wendland','dispersive'):
                h=19.5
                x,w=ref.phys.quadrature(r,12,6);xx=x+h
                g=t[:,2,None]+t[:,3,None]*xx/25.+profile(xx,f,'bulge')[1]
                ctr=t[:,2]+t[:,3]*h/25.+profile(h,f,'bulge')[1]
                drive,resist,_,vt=ref.phys.impulse_geometry(g,ctr,t[:,5],w)
                mu=ref.law(t[:,4],vt,xx,'inlay',law,sign,reg)
                original=ref.phys.residual(drive,resist,mu)
                _,sl=simulate(t,h,.425,f,'bulge','inlay',law,sign,reg,r,'prescribed')
                err=max(err,float(abs(original-sl).max()))
    assert err<5e-14
    out['random_toy_prescribed_v67_max_slip_error']=err
    out['plane_cases']=len(th);out['job']=os.environ['SLURM_JOB_ID']
    (H/'controls68.json').write_text(json.dumps(out,indent=2)+'\n')
    print(out,flush=True)

if __name__=='__main__': main()
