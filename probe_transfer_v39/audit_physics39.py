"""Independent CPU reproduction of catalog identity, contact outcomes and paid sensor rows."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import time

import numpy as np
import physics39 as w

ROOT = Path(__file__).resolve().parent
CELLS = [s+"_"+f for s in ("shelf","ridge") for f in ("weakening","strengthening")]
TOL = 3.e-12


def rng(seed,lane):
    return np.random.default_rng(np.random.SeedSequence([seed,270000,lane]))


def geom(y,shape):
    u = np.asarray(y)/.025
    if shape == "shelf":
        v = np.pi*u/2
        # cosc identity, and independent integral of the slope.
        return .086*np.sin(v/2)**2, .043*.025*(u-2*np.sin(v)/np.pi)
    return -.153*u/(1+u*u)**2, -.0019125*u*u/(1+u*u)


def contour(th,y,shape):
    return th[:,1,None]+th[:,2,None]*y+th[:,3,None]*y*y/.05+geom(y,shape)[1]


def impact(th,h,speed,shape,surface,orientation,regime):
    slope = th[:,2]+th[:,3]*h/.025+geom(h,shape)[0]
    denom = np.hypot(1,slope)
    vn,vt = (1-speed*slope)/denom,(speed+slope)/denom
    s = abs(vt)
    near = {"strong":1., "weak":.3, "null":0.}[regime]
    amp = near+(1-near)*np.cos(.5*np.pi*min(abs(h)/.023,1.))**2
    if surface=="weakening":
        direction = .19*np.sign(vt)*(1-np.exp(-s/.47))
        stat,kin = 1+.04/(1+(s/.4)**2),.89+.08*np.exp(-s/.48)
    else:
        direction = .384*.75*vt/(.75**2+vt*vt)
        stat,kin = 1+.05*s/(.45+s),.88+.11*s/(.50+s)
    mu = th[:,4]*(1+orientation*amp*direction)
    jn = (1+th[:,5])*np.maximum(vn,0.)
    stick = s <= mu*stat*jn
    # Resolve outgoing velocity in normal/tangential coordinates, then rotate into world axes.
    out_n = vn-jn
    out_t = np.where(stick,0.,vt-np.sign(vt)*mu*kin*jn)
    if np.min(vn) <= 0 or np.any((kin <= 0)|(kin>stat)):
        raise ValueError("Inconsistent contact support")
    if np.any(vt*out_t < -1.e-14) or np.any(out_n*out_n+out_t*out_t > 1+speed*speed+1.e-12):
        raise ValueError("Dissipativity violation")
    return np.column_stack(((out_n+slope*out_t)/denom, (out_t-slope*out_n)/denom)), stick


def checked(errors,name,actual,stored):
    actual,stored = np.asarray(actual),np.asarray(stored)
    if actual.shape!=stored.shape:
        raise ValueError((name,actual.shape,stored.shape))
    err=float(abs(actual.astype(float)-stored.astype(float)).max())
    if not np.isfinite(err) or err>TOL:
        raise ValueError((name,err))
    errors[name]=max(err,errors.get(name,0.))


def supports(errors):
    packed=[]
    for cell in CELLS:
        sh,surf=cell.split("_")
        for c in range(12):
            for a in w.ALPHAS:
                for lane in (10,20):
                    th=w.latent_support(a,(sh,surf,1,c),lane)
                    theta=th.cpu().numpy()
                    packed.append(np.ascontiguousarray(theta,dtype="<f8").view(np.dtype((np.void,48))).ravel())
                    for h in (w.FY,w.VYS[0]):
                        checked(errors,"profile",contour(theta,h,sh),w.contour(th,h,(sh,surf,1,c)).cpu().numpy())
                    # Height-zero velocity is independent of near-field strength. At the
                    # query and fourth-probe heights test both signs and every local regime.
                    for sign in (1,-1):
                        for h,speed,reg in [(0.,s,"strong") for s in (0.,-1.1,1.1)]+[
                                (h,s,r) for h,s in ((.023,1.1),(.025,.425))
                                for r in ("strong","weak","null")]:
                            new,st=impact(theta,h,speed,sh,surf,sign,reg)
                            old,state=w.impact(th,h,speed,(sh,surf,sign,c),reg)
                            checked(errors,"all_catalog_velocity",new,old.cpu().numpy())
                            checked(errors,"all_catalog_state",st,state.cpu().numpy())
                            if h==0 and not np.all(st==(speed==0)):
                                raise ValueError("Factual/conventional state mismatch")
    vectors=np.concatenate(packed)
    n=len(np.unique(vectors))
    if n!=len(vectors):
        raise ValueError("Expected distinct prototype vectors")
    return dict(unique_prototype_vectors=n,n_catalog_splits=len(packed),
                duplicate_vectors=0,constitutive_cases=n*18)


def audit_block(meta_file,visits,errors):
    m=json.loads(meta_file.read_text())
    cell=m["cell"]; sh,surf=cell.split("_")
    k,b=m["catalog"],m["block"]
    seed=390_000_000+10_000_000*CELLS.index(cell)+100_000*k+10_000*b
    assert m["base_seed"]==seed and m["start"]==(-1,1)[b]
    alphas=np.asarray(w.ALPHAS)[np.random.default_rng(seed+401).permutation(8)]
    path=meta_file.with_suffix(".npz")
    assert hashlib.sha256(path.read_bytes()).hexdigest()==m["sha256"]
    with np.load(path,allow_pickle=False) as f:
        raw={key:f[key] for key in ("alpha","labels","cal_x","ids","dirs","fourth_y","fourth_stick","first3_y")}
    checked(errors,"alpha_schedule",alphas,raw["alpha"])
    for e,al in enumerate(alphas):
        sc,sd=seed+100+e,seed+200+e
        assert sc==m["calibration_seeds"][e] and sd==m["deployment_seeds"][e]
        ci=rng(sc,3910).integers(2048,size=1024); di=rng(sd,3920).integers(2048,size=128)
        ids=rng(sc,98).permutation(1024)[:4]
        dirs=np.array([1,-1,1])*m["start"]
        noise=rng(sc,99).normal(size=(4,2))*.027
        th=w.latent_support(al,(sh,surf,1,k),10).cpu().numpy()[ci[ids]]
        td=w.latent_support(al,(sh,surf,1,k),20).cpu().numpy()[di]
        key=(cell,k,tuple(al))
        bucket=visits.setdefault(key,[set(),set(),set()])
        for seen,vals in zip(bucket,(ci,ci[ids],di)):
            seen.update(map(int,vals))
        for j,sgn in enumerate((1,-1)):
            checked(errors,"paid_ids",ids,raw["ids"][j,e])
            checked(errors,"far_directions",dirs,raw["dirs"][j,e])
            xf=np.column_stack((contour(th,w.FY,sh),impact(th,0.,0.,sh,surf,sgn,"strong")[0]))
            x=xf+rng(sc,11).normal(size=(1024,11))[ids]*np.array([.0016]*9+[.095]*2)
            checked(errors,"calibration_factual",x,raw["cal_x"][j,e])
            first,_=impact(th[:3],0.,dirs*1.10,sh,surf,sgn,"strong")
            checked(errors,"first_three",first+noise[:3],raw["first3_y"][j,e])
            for r,reg in enumerate(("strong","weak","null")):
                vel,state=impact(th[3:],.023,1.10,sh,surf,sgn,reg)
                checked(errors,"last_velocity",vel[0]+noise[3],raw["fourth_y"][j,r,e])
                checked(errors,"last_state",state[0],raw["fourth_stick"][j,r,e])
                checked(errors,"query_label",impact(td,.025,.425,sh,surf,sgn,reg)[1],raw["labels"][j,r,e])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--follow",action="store_true")
    args=parser.parse_args()
    if os.environ.get("BELIEF_DEVICE")!="cpu" or not (ROOT/"freeze39.sha256").exists():
        raise RuntimeError("Physics audit requires CPU and frozen v39 design")
    w.torch.set_num_threads(1)
    errors={};visits={};done=set()
    info=supports(errors)
    while True:
        for meta in sorted((ROOT/"results").glob("*/catalog_*/block_*.json")):
            if meta not in done:
                audit_block(meta,visits,errors)
                done.add(meta)
        if len(done)==96 or not args.follow: break
        time.sleep(5)
    if len(done)!=96: raise RuntimeError("Some blocks missing")
    nc,npd,nd=(sum(len(v[i]) for v in visits.values()) for i in range(3))
    info.update(complete_blocks=len(done),unique_calibration_factual=nc,unique_calibration_paid=npd,
                unique_deployment_prototypes=nd,deployment_draws=len(done)*8*128,
                error_max=errors,tolerance=TOL,host=socket.gethostname(),job=os.getenv("SLURM_JOB_ID"))
    (ROOT/"physical_summary39.json").write_text(json.dumps(info,indent=2)+"\n")
    print(json.dumps(info),flush=True)


if __name__=="__main__":
    main()
