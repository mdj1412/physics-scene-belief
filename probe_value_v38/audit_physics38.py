"""Independent CPU checks of catalog physics and reconstruction of the recorded target trials."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import time

import numpy as np

import physics38 as w

ROOT = Path(__file__).resolve().parent
CELLS = [f"{shape}_{surface}" for shape in w.OBJECTS for surface in w.SURFACES]
REGIMES = ("strong", "weak", "null")
SIZES = {"friction observations": 0}
MAXIMA = {}
TOL = 3.e-12


def array(x):
    return x.detach().cpu().numpy()


def compare(name, a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        raise ValueError((name, a.shape, b.shape))
    err = float(np.max(np.abs(a.astype(float) - b.astype(float))))
    MAXIMA[name] = max(MAXIMA.get(name, 0.), err)
    if err > TOL:
        raise ValueError((name, err))


def geom(h, shape):
    u = np.asarray(h) / .025
    if shape == "shoulder":
        return .072 * u * u / (1 + u*u), .0018 * (u - np.arctan(u))
    e = np.exp(-u*u / 1.28)
    return -.078 * u * e, -.078 * .016 * (1-e)


def contour(th, h, shape):
    h = np.asarray(h)
    return th[:,1,None] + th[:,2,None]*h + th[:,3,None]*h*h/.05 + geom(h,shape)[1]


def independent(th, h, speed, law, regime, audit=False):
    shape, surface, sign = law
    slope = th[:,2] + th[:,3]*h/.025 + geom(h,shape)[0]
    norm = np.hypot(1,slope)
    vn, vt = (1-speed*slope)/norm, (speed+slope)/norm
    s = np.abs(vt)
    height_phase = min(abs(h)/.023,1.) * np.pi
    local = {"strong":1., "weak":.3, "null":0.}[regime]
    amplitude = .5*(1+np.cos(height_phase)) + local*.5*(1-np.cos(height_phase))
    if surface == "weakening":
        d = .19 * np.sign(vt)*(1-np.exp(-s/.47))
        stat = 1 + .04/(1+(s/.4)**2)
        kin = .89 + .08*np.exp(-s/.48)
    else:
        d = .384*.75*vt/(.75**2+vt**2)
        stat = 1 + .05*s/(s+.45)
        kin = .88 + .11*s/(s+.50)
    mu = th[:,4] * (1+sign*amplitude*d)
    jn = (1+th[:,5])*np.maximum(vn,0.)
    stick = s <= mu*stat*jn
    jt = np.where(stick, vt, np.sign(vt)*mu*kin*jn)
    # Explicitly rotate from local contact coordinates back to world coordinates.
    normal_out = vn-jn
    tangent_out = vt-jt
    out = np.column_stack(((normal_out+slope*tangent_out)/norm,
                           (tangent_out-slope*normal_out)/norm))
    if audit:
        if not ((mu>0).all() and (kin>0).all() and (stat>kin).all() and (vn>0).all()):
            raise ValueError("outside constitutive bounds")
        compare("normal restitution", normal_out, -th[:,5]*vn)
        if np.any(np.abs(tangent_out[stick])>TOL) or np.any(vt[~stick]*tangent_out[~stick]<-TOL):
            raise ValueError("inconsistent tangential contact state")
        if np.any(np.square(out).sum(1) > 1+speed**2+TOL):
            raise ValueError("kinetic energy increased")
        if np.any((s > mu*stat*jn) != (~stick)):
            raise ValueError("inconsistent static threshold")
        # Rebuild the imparted normal and tangential impulses from Cartesian velocity differences.
        di, dj = 1-out[:,0], speed-out[:,1]
        compare("momentum projection normal", (di-slope*dj)/norm, jn)
        compare("momentum projection tangential", (slope*di+dj)/norm, jt)
        SIZES["friction observations"] += len(th)
    return out, stick


def pack_rows(a):
    a = np.ascontiguousarray(a, dtype="<f8")
    return a.view(np.dtype((np.void,a.dtype.itemsize*a.shape[1]))).reshape(-1)


def static_checks():
    all_rows = []
    for cell in CELLS:
        shape, surface = cell.split("_")
        for al in w.ALPHAS:
            for lane in (10,20):
                support = w.latent_support(al,(shape,surface,1),lane)
                th = array(support)
                compare("catalog sign invariance", th, array(w.latent_support(al,(shape,surface,-1),lane)))
                all_rows.append(pack_rows(th))
                for heights in (w.FY,w.VYS[0]):
                    compare("catalog contour",contour(th,heights,shape),array(w.contour(support,heights,(shape,surface,1))))
                for sign in (1,-1):
                    law = (shape,surface,sign)
                    for reg in REGIMES:
                        for h,speed in ((0.,0.),(0.,-1.1),(0.,1.1),(.023,1.1),(.025,.425)):
                            out, stick = independent(th,h,speed,law,reg,True)
                            actual, target_stick = w.impact(support,h,speed,law,reg)
                            compare("catalog velocity",out,array(actual))
                            compare("catalog contact state",stick,array(target_stick))
                            if h==0 and not np.all(stick==(speed==0)):
                                raise ValueError("invalid factual/far contact state")
    unique = len(np.unique(np.concatenate(all_rows)))
    if unique!=len(all_rows)*w.SUPPORT_SIZE:
        raise ValueError("duplicate prototypes between or within catalogs")
    return dict(catalogs=len(all_rows),size_per_catalog=w.SUPPORT_SIZE,unique_total=unique,
                catalog_intersections=0,exhaustive_chosen_constitutive_states=SIZES["friction observations"])


def rng(seed,lane):
    return np.random.default_rng(np.random.SeedSequence([seed,270000,lane]))


def raw_checks(folder, visits):
    meta = json.loads((folder/"meta.json").read_text())
    shape,surface = meta["cell"].split("_")
    for tag,sgn in (("plus",1),("minus",-1)):
        law=(shape,surface,sgn)
        with np.load(folder/f"shared_{tag}.npz",allow_pickle=False) as shared:
            alpha, actions = shared["alpha"],shared["actions"]
            xs, ids, y0 = shared["far_x"], shared["ids"], shared["first3_y"]
            dirs=shared["dirs"]
            xx, qry, pr = shared["xx"],shared["query_obs"],shared["probe_obs"]
        arms={}
        for reg in REGIMES:
            with np.load(folder/f"{reg}_{tag}.npz",allow_pickle=False) as f:
                arms[reg] = (f["fourth_y"],f["fourth_stick"],f["labels"])
        for e,al in enumerate(alpha):
            sc,sd = meta["calibration_seeds"][e],meta["deployment_seeds"][e]
            ci = rng(sc,3810).integers(2048,size=1024)
            di = rng(sd,3820).integers(2048,size=128)
            paid = rng(sc,98).permutation(1024)[:4]
            compare("paid row indices",paid,ids[e])
            compare("first command directions",dirs[e],np.array([1,-1,1])*meta["start"])
            catalog_key=(meta["cell"], tuple(al.tolist()))
            if sgn==1:
                visits.setdefault(catalog_key, [set(),set(),set()])
                for bucket,rows in zip(visits[catalog_key],(ci,ci[paid],di)):
                    bucket.update(map(int,rows))
            tc = array(w.latent_support(al,law,10))[ci[paid]]
            td = array(w.latent_support(al,law,20))[di]
            far = independent(tc[:3],0.,1.1*dirs[e],law,"strong")[0]
            factual_c = np.column_stack((contour(tc,w.FY,shape),independent(tc,0.,0.,law,"strong")[0]))
            factual_d = np.column_stack((contour(td,w.FY,shape),independent(td,0.,0.,law,"strong")[0]))
            factual_sigma=np.array([.0016]*9+[.095]*2)
            xc=factual_c+rng(sc,11).normal(size=(1024,11))[paid]*factual_sigma
            xd=factual_d+rng(sd,21).normal(size=(128,11))*factual_sigma
            compare("archived calibration factual", xc, xs[e])
            compare("archived deployment factual", xd, xx[e])
            noise = rng(sc,99).normal(size=(4,2))*.027
            compare("archived first three paid",far+noise[:3],y0[e])
            for reg,(obs,stick,labels) in arms.items():
                m,k = independent(tc[-1:],.023,1.1,law,reg)
                compare("archived fourth paid", m[0]+noise[3],obs[e])
                compare("archived fourth slip/stick",k[0],stick[e])
                compare("archived held-out labels",independent(td,.025,.425,law,reg)[1],labels[e])
            for a,observed,mu,sigma in ((w.Q,qry,contour(td,w.VYS[0],shape),.00030),
                                        (w.P,pr,independent(td,0.,-1.1,law,"strong")[0],.027)):
                keep=actions[e]==a
                predicted = mu[keep,None] + rng(sd,40+a).normal(size=(128,32,mu.shape[1]))[keep]*sigma
                compare("archived selected sensor",predicted,observed[e])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--timeout-seconds",type=int,default=3600)
    args=parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or w.torch.cuda.is_available() or os.environ.get("BELIEF_DEVICE")!="cpu":
        raise RuntimeError("Use a CPU-only Slurm step outside GPU timing")
    w.torch.set_num_threads(1)
    start=time.monotonic()
    visited={}
    with w.torch.no_grad():
        result=static_checks()
        print("catalog checked",result,flush=True)
        completed=set()
        while len(completed)<32:
            for meta in sorted((ROOT/"results").glob("*/block_*/meta.json")):
                if meta.parent not in completed:
                    raw_checks(meta.parent,visited)
                    completed.add(meta.parent)
                    print("archive checked",meta.parent.parent.name,meta.parent.name,len(completed),flush=True)
            if time.monotonic()-start>args.timeout_seconds:
                raise RuntimeError("timed out before full results")
            if len(completed)<32:
                time.sleep(8)
    buckets=np.array([[len(b) for b in v] for v in visited.values()])
    result.update(completed_blocks=len(completed),checks_max_abs=MAXIMA,tolerance=TOL,
                  unique_calibration_factual_prototypes=int(buckets[:,0].sum()),
                  unique_calibration_paid_prototypes=int(buckets[:,1].sum()),
                  unique_deployment_prototypes=int(buckets[:,2].sum()),deployment_draws=32768,
                  host=socket.gethostname(),job=os.environ["SLURM_JOB_ID"],
                  physics_sha256=hashlib.sha256((ROOT/"physics38.py").read_bytes()).hexdigest())
    tmp=ROOT/"physical_summary.tmp.json"
    tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    tmp.replace(ROOT/"physical_summary.json")
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    main()
