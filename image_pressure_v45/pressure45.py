"""Prescribed, centered finite pressure impulse with a single translational slip mode."""
import argparse
from functools import lru_cache
import itertools
import json
from pathlib import Path
import os
import sys
import numpy as np
from scipy.special import erf, roots_legendre

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "blind_decoder_v43"))
from physics_check43 import friction, slope

RADII = (0., .25, 2.)  # Peak radius of circular footprint in projected millimetres.
OFFSETS = (-.5, 0., .5)
INCOMING = .425
THRESHOLD = .020
SURFACES = ("inlay", "woven")
LAWS = ("rational", "gaussian_rate")
SHAPES = ("fold", "ripple")
VARIANTS = ("base", "hollow", "bulge")
HEIGHTS = (19, 31)


@lru_cache(None)
def quadrature(radius, nx, nt):
    """Normal impulse weights integrated over projected y and the symmetric loading cycle."""
    if radius == 0:
        return np.zeros(1), np.ones(1)
    q, w = roots_legendre(nx)
    time, wt = roots_legendre(nt)
    # Sine envelope over cumulative load fraction [0,1] is symmetric about its midpoint.
    scale = np.sin(np.pi * (time + 1) / 4)
    wx = w * .75 * (1 - q*q)
    x = radius * scale[:, None] * q
    weight = .5 * wt[:, None] * wx
    assert abs(weight.sum()-1.) < 5e-14 and abs(np.sum(weight*x)) < 1e-13
    return x.ravel(), weight.ravel()


@lru_cache(None)
def slope_field(center, radius, shape, var, nx, nt):
    offset, weight = quadrature(radius, nx, nt)
    h = center + offset
    additive = np.array([slope(z,shape,var) for z in h])
    return h, additive, weight


def material(mu, vt, h, sf, law, signs, regimes):
    near = np.array([{"strong":1.,"weak":.3,"null":0.,"global_null":0.}[a] for a in regimes])
    envelope = near[:,None] + (1.-near[:,None]) * np.cos(.5*np.pi*np.minimum(abs(h)/23.,1.))**2
    envelope[regimes=="global_null"] = 0.
    if sf == "inlay":
        k = np.exp(-.5*((h-22.)/5.5)**2)
        scale, orient = 1.+.079*k, 1.-1.45*k
    else:
        phase = 2*np.pi*h/24.
        scale, orient = 1.+.047*np.sin(phase), .38+.62*np.cos(phase)
    v = abs(vt)
    if law == "rational":
        rate = v*v/(v*v+.58**2)
        asym = .145*vt**3/(v**3+.61**3)
        stat = 1.045-.046*rate
    else:
        band = np.exp(-.5*((v-.64)/.25)**2)
        asym = .16*erf(vt/.34)*(1.-1.55*band)
        stat = 1.007+.043*band
    return mu[:,None] * scale * (1+signs[:,None]*envelope*orient*asym)*stat


def impulse_geometry(g, center_slope, restitution, weights):
    """Normal impulse projected on the central tangent and local Coulomb projection factors."""
    norm = np.hypot(1.,g)
    norm0 = np.hypot(1.,center_slope)
    normal_impulse = (1.+restitution)*np.maximum(1.-INCOMING*center_slope, 0.)/norm0
    slip0 = (INCOMING+center_slope)/norm0
    normal_tangent = (g-center_slope[:,None])/(norm*norm0[:,None])
    tangent_tangent = (1.+g*center_slope[:,None])/(norm*norm0[:,None])
    drive = slip0 + normal_impulse*(normal_tangent@weights)
    resistance_per_mu = normal_impulse[:,None]*abs(tangent_tangent)*weights
    return drive, resistance_per_mu, normal_impulse, (INCOMING+g)/norm


def residual(drive, resistance_per_mu, mu):
    return np.maximum(abs(drive)-(resistance_per_mu*mu).sum(axis=-1),0.)


def spatial_area(center, radius, theta, shape, var, nx):
    """Effective loaded surface area at peak load. p is a density per projected area."""
    if radius == 0:
        return np.zeros(len(theta))
    # Chebyshev angle quadrature removes the endpoint power from the density of p².
    a, w = roots_legendre(nx)
    angle = (a+1)*np.pi/2
    q = np.cos(angle)
    h = center + radius*q
    deformed = np.array([slope(z,shape,var) for z in h])
    g = theta[:,2,None]+theta[:,3,None]*h/25.+deformed
    # Marginal y-integral of squared projected pressure: density 8/(3pi)*(1-q²)^(3/2).
    density_w = 4./3.*w*np.sin(angle)**4
    stretch_inv = (1./np.hypot(1.,g)) @ density_w
    return (8*np.pi*radius**2/9.)/stretch_inv


def output_folder(high):
    return ROOT / ("labels45_pressure_fine" if high else "labels45")


def run_group(panel, shape, var, height, high):
    folder = output_folder(high)/f"panel_{panel:02d}"
    folder.mkdir(exist_ok=True,parents=True)
    paths = {(sf,law):folder/f"{shape}_{sf}_{law}_{var}_h{height}.npz" for sf,law in itertools.product(SURFACES,LAWS)}
    if all(path.exists() for path in paths.values()):
        return
    with np.load(ROOT / "results" / "worlds" / f"panel_{panel:02d}.npz") as f:
        poses = f["theta"]
        E,N,P = poses.shape
        theta = poses.reshape(E*N,P)
        signs = np.repeat(f["signs"],N)
        regimes = np.repeat(f["regimes"],N)
    assert (E,N,P)==(4,128,6)
    arr = {k:np.empty((3,3,E,N),np.float32) for k in paths}
    count = len(theta)
    eff_area = np.empty((3,3,E,N),np.float32)
    impulse = np.empty((3,E,N),np.float32)
    nx,nt = (24,12) if high else (12,6)
    for d,off in enumerate(OFFSETS):
        center = height + off
        g0 = theta[:,2]+theta[:,3]*(center/25.)+slope(center,shape,var)
        for a,r in enumerate(RADII):
            h, add, weight = slope_field(center,r,shape,var,nx,nt)
            g = theta[:,2,None]+theta[:,3,None]*h/25.+add
            drive,resist,normal,vt=impulse_geometry(g,g0,theta[:,5],weight)
            assert normal.min()>0
            impulse[d]=normal.reshape(E,N)
            eff_area[a,d] = spatial_area(center,r,theta,shape,var,24 if high else 16).reshape(E,N)
            for sf,law in paths:
                mu=material(theta[:,4],vt,h,sf,law,signs,regimes)
                speed=residual(drive,resist,mu)
                assert speed.shape == (count,) and np.isfinite(speed).all() and np.all(mu>0.)
                arr[sf,law][a,d] = speed.reshape(E,N)
    for key,path in paths.items():
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, labels=arr[key]<=THRESHOLD, slip_speed=arr[key], surface_effective_area_mm2=eff_area,
            normal_impulse_Ns=impulse, centers_mm=height+np.array(OFFSETS), radii_mm=RADII,
            area_projected_mm2=np.pi*np.square(RADII), effective_projected_area_mm2=8./9.*np.pi*np.square(RADII))
        os.replace(tmp,path)


def neutral():
    # Constant slope and material have one closed form for every load footprint.
    rng = np.random.default_rng(450123)
    g0 = rng.uniform(-.22,.18,200)
    mu0 = rng.uniform(.05,.65,200)
    e = rng.uniform(.05,.95,200)
    max_error=0.; max_centroid=0.; max_load=0.; max_point=0.; max_mu=0.; max_area_err=0.
    for nx,nt in ((12,6),(24,12)):
        for r in RADII:
            x,weight=quadrature(r,nx,nt)
            max_centroid=max(max_centroid,abs(x@weight))
            max_load=max(max_load,abs(weight.sum()-1.))
            g=np.broadcast_to(g0[:,None],(len(g0),len(weight)))
            drive,resist,j,vt=impulse_geometry(g,g0,e,weight)
            speed=residual(drive,resist,mu0[:,None])
            truth=np.maximum((abs(INCOMING+g0)-mu0*(1.+e)*(1.-INCOMING*g0))/np.hypot(1.,g0),0.)
            max_error=max(max_error,float(np.max(abs(speed-truth))))
            # A constant tilted plane stretches the effective area by sqrt(1+s²).
            if r:
                a,w=roots_legendre(24)
                factor=(4./3.*w*np.sin((a+1)*np.pi/2)**4).sum()
                max_area_err=max(max_area_err,abs(factor-1.))
            for sf,law in itertools.product(SURFACES,LAWS):
                h=np.array([19.,31.,30.5])
                vt=np.array([[-.2,.451,.6],[.1,.01,-.05]])
                signs=np.array([-1,1]); regs=np.array(["weak","strong"])
                mu=material(mu0[:2],vt,h,sf,law,signs,regs)
                for i,k in itertools.product(range(2),range(3)):
                    m=friction(mu0[i],vt[i,k],h[k]/1000.,sf,law,signs[i],regs[i])[0]
                    max_mu=max(max_mu,abs(mu[i,k]-m))
            if not r: max_point=max(max_point,float(np.max(abs(speed-truth))))
    assert max(max_error,max_centroid,max_load,max_mu,max_point,max_area_err)<1e-12
    # A curved friction field µ=µ0+0.1*x² gives a nonzero size effect even on a flat substrate.
    demo=[]
    for r in RADII:
        x,w=quadrature(r,12,6)
        drive,resist,j,vt=impulse_geometry(np.zeros((1,len(w))),np.zeros(1),np.array([.5]),w)
        speed=residual(drive,resist,.12+.10*x[None]**2).item()
        if abs(speed-max(.425-1.5*(.12+.01*r**2),0))>2e-10:
            raise ValueError("Quadratic-field fixture failed")
        demo.append(speed)
    result=dict(flat_constant_slip_error_mps=max_error,unit_scalar_load_error=max_load,
      first_contact_and_centroid_max_error_mm=max_centroid,material_scalar_replay_error=max_mu,
      point_closed_form_error_mps=max_point,peak_pressure_area_normalization_error=max_area_err,
      friction_quadratic_flat_radii_mm=RADII,friction_quadratic_expected_residual_mps=demo)
    (ROOT/"pressure_neutral45.json").write_text(json.dumps(result,indent=2)+"\n")
    print(result)


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--neutral", action="store_true")
    p.add_argument("--fine",action="store_true")
    p.add_argument("--panels",nargs="+",type=int)
    args=p.parse_args()
    if args.neutral: neutral()
    else:
        if not (ROOT/"PRESSURE_GO").exists(): raise RuntimeError("Target scoring requires freeze release.")
        for panel in args.panels:
            for s,v,h in itertools.product(SHAPES,VARIANTS,HEIGHTS):
                run_group(panel,s,v,h,args.fine)
            print("completed",panel,"fine" if args.fine else "primary",flush=True)
