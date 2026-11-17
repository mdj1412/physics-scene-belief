"""Held-out contour, surface and deformation fixtures. Shape coordinates in mm."""
from functools import lru_cache
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent/'factorial_projection_v47'))
from world47 import ALIGNMENTS, COS2, HEIGHTS, OFFSETS, RADII, SIGN, VARIANTS, source_axis

FAMILIES = ('saddleback', 'terrace', 'swale', 'scallop')
SURFACES = ('mosaic', 'nap')
LAWS = ('rational', 'gaussian_rate')
STATES = (0, -1, 1)
STATE_WEIGHTS = (.50, .25, .25)
# Each term: center, width, amplitude, optional asymmetric shoulder.
WINGS = {
    'saddleback': ((13.5,2.3,-.70,.3),(23.5,1.75,.84,-.65),(26.8,1.45,.29,.45)),
    'terrace': ((13.1,1.9,.37,-.5),(24.6,2.8,.80,.82),(36.3,2.0,-.48,-.25)),
    'swale': ((13.7,2.2,-.26,.7),(24.5,2.6,-.88,-.50),(35.9,2.0,.60,.55)),
    'scallop': ((14.1,1.95,.75,.15),(23.45,1.55,-.62,-.7),(26.6,1.6,.74,.4),(36.7,2.2,-.34,-.3)),
}
# These can change global and query-profile geometry; the inert ±2.5-mm patches match across families.
STATIONS = ((18.75,4.05,.235,.38),(31.4,4.7,-.205,-.46))
OPTICAL = {'saddleback':(1,-2,2,0,-1,0,1),'terrace':(1,2,0,-1,-2,1,-1),
           'swale':(-2,1,1,0,2,-1,-1),'scallop':(0,-1,2,1,-2,1,-1)}
# Nominal deformation slope, quadratic drift, fundamental mode, second harmonic.
BENDING = {'saddleback':(.142,.30,.18,-.11),'terrace':(.161,-.24,.31,.18),
           'swale':(.126,.16,-.14,.21),'scallop':(.153,-.34,.22,-.24)}


def camera_shape(x_mm, family, variant='base'):
    if variant not in SIGN:
        raise ValueError(variant)
    x = np.asarray(x_mm, dtype=float)
    z,g = np.zeros_like(x), np.zeros_like(x)
    for at,width,amp,a in STATIONS+WINGS[family]:
        t=(x-at)/width
        q=np.maximum(1.-t*t,0.)
        z+=amp*q**3*(1.+a*t)
        g+=amp/width*(-6*t*q*q*(1.+a*t)+a*q**3)
    return z,g


def loaded_shape(x_mm, height_mm, family, state):
    if state not in STATES:
        raise ValueError(state)
    x=np.asarray(x_mm,dtype=float)
    u=(x-height_mm)/2.5
    if np.any(abs(u)>1.+1.e-10):
        raise ValueError('Loaded patch excludes points outside ±2.5 mm.')
    z,g=camera_shape(x,family)
    slope,a,b,c=BENDING[family]
    shape=(u+a*u*u/2+b*np.sin(np.pi*u)/np.pi+c*(1-np.cos(2*np.pi*u))/(2*np.pi))/(1+b)
    gradient=(1.+a*u+b*np.cos(np.pi*u)+c*np.sin(2*np.pi*u))/(1+b)
    return z+2.5*state*slope*shape,g+state*slope*gradient


def surface_field_mm(h,surface):
    if surface=='mosaic':
        # Two separated material deposits with distinct width, sign and stiffness.
        a=math.exp(-.5*((h-16.)/4.1)**2)
        b=math.exp(-.5*((h-32.5)/3.2)**2)
        return 1.+.082*a-.059*b, .7+.38*a-.51*b
    if surface=='nap':
        # Smooth transition between two oppositely oriented brushed patches.
        edge=math.tanh((h-25.)/5.2)
        return 1.015+.059*edge, .25-.79*edge
    raise ValueError(surface)


def sensor_delta_mm(family,height_mm,alignment,variant):
    axis=source_axis(height_mm)
    v=np.asarray(OPTICAL[family],dtype=float)
    v-=axis*(v@axis)
    v/=np.linalg.norm(v)
    a=COS2[alignment]
    return SIGN[variant]*1.2*(np.sqrt(a)*axis+np.sqrt(1.-a)*v)


def install_world43(world):
    if getattr(world,'_v50_installed',False):
        return
    base=world.perturbation
    sf=world.surface_field
    def geometry(y,family,variant):
        if family not in FAMILIES:
            return base(y,family,variant)
        y=world.t(y)
        z,g=camera_shape(y.detach().cpu().numpy()*1000.,family,variant)
        return (world.torch.as_tensor(g,device=y.device,dtype=y.dtype),
                world.torch.as_tensor(z*.001,device=y.device,dtype=y.dtype))
    def field(h,surface):
        return surface_field_mm(float(h)*1000.,surface) if surface in SURFACES else sf(h,surface)
    world.perturbation=geometry
    world.surface_field=field
    world.TARGET_SHAPES=FAMILIES
    world.TARGET_SURFACES=SURFACES
    world._v50_installed=True


@lru_cache(None)
def gauge_mm(family,site_mm,state,load_fraction=.2):
    x=np.array([site_mm-1.5,site_mm+1.5])
    z=loaded_shape(x,site_mm,family,state)[0]-camera_shape(x,family)[0]
    return float((z[1]-z[0])*load_fraction)


@lru_cache(32)
def noise_terms(seed,site):
    rng=np.random.default_rng(np.random.SeedSequence([seed,500714,site]))
    gain=rng.uniform(.9,1.1,(4,128,1))*rng.uniform(.93,1.07)
    bias=rng.uniform(-.012,.012)+rng.normal(0,.007,(4,128,1))
    noise=rng.normal(0,.028,(4,128,4))
    return gain,bias,noise


def readings(pop,family,site_mm,station_state):
    gain,bias,noise=noise_terms(int(pop['seed']),site_mm)
    return (gain*gauge_mm(family,site_mm,station_state)+bias+noise).astype(np.float32)
