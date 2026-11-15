"""Independent power experiment for the identifiable profile parameter; CPU/Slurm only.

Under the exact v25 polynomial profile and Gaussian calibration, the mean M and
the least-squares curvature coefficient T have independent noise. Factual velocity
depends only on slope and restitution; integrating out the independent friction
width makes (M,T) sufficient for the two population alignment parameters, with
zero information on the second one. No contact labels are used here.
"""
import json
import os
from pathlib import Path
import sys

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import ndtr
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "proxy_shift_v26"))
from physics26 import CALIBRATION, change_proxy, parameters, t, torch
from physics25 import FY, profile

Y = np.asarray(FY)
B = Y**2/.05
B0 = B - B.mean()
SIGMA = CALIBRATION[0]
SD_M = SIGMA/np.sqrt(len(Y))
SD_T = SIGMA/np.sqrt(B0@B0)
OFFSET = .00115
OFFSET_HALF = .0017
CURVES = np.array([.022,.340])
CURVE_HALF = .012
T2 = SD_T**2 + np.mean(CURVES**2) + CURVE_HALF**2/3.
SCALE = OFFSET*(CURVES[1]**2-CURVES[0]**2)/2.
SIZES = np.array([64,256,1024,4096,16384])
ALPHAS = np.array([-.8,0.,.8])
REPS = 96
SEED = 2797033


def profile_sensor(units, flip, eps):
    wide = units[:, 0] >= .5
    curve = np.where(units[:, 2]>=.5,1.,-1.)*CURVES[wide.astype(int)] + 2*CURVE_HALF*(units[:,4]-.5)
    offset = OFFSET*np.where(wide^flip,1.,-1.) + 2*OFFSET_HALF*(units[:,6]-.5)
    slope = .070*(units[:,7]-.5)
    return offset[:, None]+slope[:,None]*Y+curve[:,None]*B+eps*SIGMA


def density_components(m, tc, points=6):
    """Exact uniform offset integral, Gauss–Legendre on each signed curvature interval.

    Returns two equally normalized densities: original and swapped offset/width.
    Constant Gaussian/uniform normalizations common to both are omitted.
    """
    m, tc = np.broadcast_arrays(m,tc)
    out = np.zeros(m.shape+(2,))
    nodes, ws = leggauss(points)
    for group, width in enumerate(CURVES):
        for sign in (-1,1):
            for node,w in zip(nodes,ws):
                c = sign*width+CURVE_HALF*node
                marginal_t = w*np.exp(-.5*((tc-c)/SD_T)**2)
                for swap in (0,1):
                    offset = OFFSET*(1 if group^swap else -1)
                    # Symmetrize around the center so both CDF arguments use the
                    # lower tail at extreme m; avoids subtracting two values near one.
                    d = abs(m-B.mean()*c-offset)
                    z1 = (OFFSET_HALF-d)/SD_M
                    z0 = (-OFFSET_HALF-d)/SD_M
                    out[...,swap] += marginal_t*(ndtr(z1)-ndtr(z0))
    return out


def density_score(m,tc,points):
    ds = density_components(m,tc,points=points)
    return (ds[...,0]-ds[...,1])/(ds[...,0]+ds[...,1])


class ScoreTable:
    def __init__(self):
        self.mm = np.linspace(-.008,.008,321)
        self.tt = np.linspace(-5.,5.,401)
        self.table = density_score(self.mm[:,None],self.tt[None,:],6)

    def __call__(self,m,tc):
        ix = (m-self.mm[0])/(self.mm[1]-self.mm[0])
        iy = (tc-self.tt[0])/(self.tt[1]-self.tt[0])
        i = np.clip(ix.astype(int),0,len(self.mm)-2)
        j = np.clip(iy.astype(int),0,len(self.tt)-2)
        ex,ey = ix-i,iy-j
        a = self.table
        s = ((1-ex)*((1-ey)*a[i,j]+ey*a[i,j+1])
             + ex*((1-ey)*a[i+1,j]+ey*a[i+1,j+1]))
        outside=(ix<0)|(iy<0)|(ix>len(self.mm)-1)|(iy>len(self.tt)-1)
        if outside.any():
            s[outside]=density_score(m[outside],tc[outside],12)
        return s


def mle(score):
    """Concave one-parameter marginal log likelihood, derivative sum s/(1+alpha*s)."""
    if np.sum(score/(1-score)) <= 0:
        return -1.
    if np.sum(score/(1+score)) >= 0:
        return 1.
    lo,hi=-1.,1.
    for _ in range(17):
        mid=(lo+hi)*.5
        if np.sum(score/(1+mid*score)) >= 0:
            lo=mid
        else:
            hi=mid
    return (lo+hi)*.5


def publish_stats(alpha, vals, null_reject):
    vals=np.asarray(vals)
    sd=float(vals.std(ddof=1))
    return dict(mean=float(vals.mean()),bias=float(vals.mean()-alpha),sd=sd,
                bias_mc_se=sd/np.sqrt(len(vals)),rmse=float(np.sqrt(np.mean((vals-alpha)**2))),
                sign_accuracy=float(np.mean(vals*alpha>0)) if alpha!=0 else None,
                reject_zero=float(np.mean(null_reject)))


def main():
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("BELIEF_DEVICE")!="cpu":
        raise RuntimeError("Run on the allocated CPU with BELIEF_DEVICE=cpu.")
    torch.set_num_threads(1)
    check=np.random.default_rng(27730224)
    units=check.random((128,9));flip=check.random(128)<.5;noise=check.normal(size=(128,9))
    source=parameters(t(units));changed=change_proxy(source,torch.as_tensor(2*flip,dtype=torch.uint8))
    reference=profile(changed,FY).cpu().numpy()+SIGMA*noise
    model_error=float(np.max(np.abs(profile_sensor(units,flip,noise)-reference)))
    assert model_error<1e-14
    assert abs(Y.mean())<1e-14 and abs(B0@Y)<1e-14 and abs(B0.mean())<1e-14
    scoretab=ScoreTable()
    parts=[]
    max_interp=0.;max_quadrature=0.
    for ia,alpha in enumerate(ALPHAS):
        measurements=np.empty((REPS,len(SIZES),5))
        for rep in range(REPS):
            rng=np.random.default_rng(np.random.SeedSequence([SEED,ia,rep]))
            n=int(SIZES[-1]);unit=rng.random((n,9))
            flipped=rng.random(n)<(1-alpha)/2.
            xx=profile_sensor(unit,flipped,rng.normal(size=(n,len(Y))))
            m=xx.mean(axis=1);tc=xx@B0/(B0@B0)
            moment=m*(tc**2-T2)/SCALE
            score=scoretab(m,tc)
            if rep%12==0:
                direct=density_score(m[:80],tc[:80],16)
                max_interp=max(max_interp,float(np.max(np.abs(direct-score[:80]))))
                max_quadrature=max(max_quadrature,float(np.max(abs(direct-density_score(m[:80],tc[:80],6)))))
            for j,n in enumerate(SIZES):
                v=moment[:n];s=score[:n]
                z_m=v.mean()/(v.std(ddof=1)/np.sqrt(n))
                z_s=s.sum()/np.sqrt(s@s)
                measurements[rep,j]=[v.mean(),mle(s),abs(z_m)>norm.ppf(.975),abs(z_s)>norm.ppf(.975),np.mean(s*s)]
        for j,n in enumerate(SIZES):
            m,est,rej_m,rej_s,observed_info=measurements[:,j].T
            parts.append(dict(alpha_c=float(alpha),n=int(n),reps=REPS,
                moment=publish_stats(alpha,m,rej_m),projected_mle=publish_stats(alpha,est,rej_s),
                mean_squared_score_at_null=float(observed_info.mean())))
    assert max_interp < .001 and max_quadrature < 1e-10, (max_interp,max_quadrature)
    out=dict(scope="Independent diagnostic, no policy or held-out loss was computed.",
        profile_unit_test_max_error=model_error,max_bilinear_score_error=max_interp,
        max_6_vs_16_point_quadrature_score_error=max_quadrature,alpha_f_information=0.,
        seed=SEED,noise_sd_M=SD_M,noise_sd_T=SD_T,population_ET2=T2,
        moment_denominator=SCALE,draws_per_alpha=REPS*int(SIZES[-1]),
        n_notes="Independent replicas per alpha, common prefixes across sizes; iid width sampling.",
        test_notes="Two-sided normal-reference moment and null score tests, nominal 5%; MLE is bounded, moment unbounded.",
        rows=parts)
    folder=Path(__file__).resolve().parent/"results";folder.mkdir(exist_ok=True)
    tmp=folder/"identifiable_power.tmp";tmp.write_text(json.dumps(out,indent=2,allow_nan=False))
    tmp.replace(folder/"identifiable_power.json")
    for row in parts:
        print(row["alpha_c"],row["n"],row["moment"]["sd"],row["projected_mle"]["sign_accuracy"],row["projected_mle"]["reject_zero"])

if __name__=="__main__":
    main()
