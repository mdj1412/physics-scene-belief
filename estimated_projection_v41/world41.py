"""New test shapes, contact heights and unlabeled paired profile measurements."""
from functools import lru_cache
from pathlib import Path
import sys
import math

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT/'tangent_transfer_v40'))
import physics40 as v40
from physics40 import (ALPHAS, FACTUAL_INCOMING, FY, MATCH_HEIGHT, MATCH_SPEED, P,
                       PROBE_INCOMING, Q, QUERY_INCOMING, REGIMES, SUPPORT_SIZE,
                       SURFACES, CONTOURS, directions, draw_theta, profile, sigmas,
                       coefficients, stream, t, torch)

FAMILIES = ('flare', 'roll')
HEIGHTS = (.019, .031)
SUPPORT_SEED = 2041100000
PAIR_COUNT = 32
MAX_AMPLITUDE = .0009
LOBE_KNOTS = (.011, .019, .023, .031, .039)
LOBE_WEIGHTS = {'flare': (.80, -.85, .30, .95), 'roll': (-.45, .55, .95, -.55)}


def view_heights(height):
    return height + np.arange(-3,4,dtype=np.float64)*.002


def base_profile(y, family):
    u = t(y)/.025
    if family == 'flare':
        return (.068*(-torch.expm1(-u.square()/.90)),
                .068*.025*(u-.5*math.sqrt(.90*math.pi)*torch.erf(u/math.sqrt(.90))))
    if family == 'roll':
        den = torch.sqrt(1.+u.square())
        return -.072*u/den, -.072*.025*(u.square()/(1.+den))
    raise ValueError(family)


def deformation(y, family):
    y=t(y)
    slope = torch.zeros_like(y)
    value = torch.zeros_like(y)
    for left, right, weight in zip(LOBE_KNOTS[:-1], LOBE_KNOTS[1:], LOBE_WEIGHTS[family]):
        g, f = v40.lobe(y,left,right)
        value = value + MAX_AMPLITUDE*weight*f
        slope = slope + MAX_AMPLITUDE*weight*g
    return slope,value



def perturbation(y, family, variant):
    g,f = base_profile(y,family)
    if variant == 'base':
        return g,f
    sign = {'hollow':-1., 'bulge':1.}[variant]
    dg,df = deformation(y,family)
    return g+sign*dg, f+sign*df


def catalog_seed(panel, alpha_index, lane):
    if not 0 <= panel < 500 or lane not in (10,20) or not 0<=alpha_index<len(ALPHAS):
        raise ValueError('Invalid support key')
    return SUPPORT_SEED+panel*10000+alpha_index*100+lane


@lru_cache(maxsize=32)
def _catalog(panel, alpha_index, lane):
    return draw_theta(SUPPORT_SIZE, catalog_seed(panel,alpha_index,lane),lane,ALPHAS[alpha_index])


def latent_support(alpha, context, lane):
    return _catalog(context[-1], ALPHAS.index(list(map(float,alpha))),lane)


def draw_indices(n, seed, lane):
    return stream(seed,4100+lane).integers(SUPPORT_SIZE,size=n)


def draw_object(n, seed, lane, alpha, context):
    return latent_support(alpha,context,lane)[draw_indices(n,seed,lane)]


def contour(theta,y,context):
    return profile(theta,y)+perturbation(y,*context[:2])[1]


def impact(theta,height,speed,context,regime):
    family,variant,_,law,sign,_ = context
    _,_,slope,curve,mu,e = theta.unbind(-1)
    local=slope+curve*(height/.025)+perturbation(height,family,variant)[0]
    norm=torch.sqrt(1+local.square())
    vt=(speed+local)/norm
    jn=(1+e)*((1-local*speed)/norm).clamp_min(0.)
    static,kinetic=coefficients(mu,vt,height,law,sign,regime)
    stick=vt.abs()<=static*jn
    jt=torch.where(stick,vt,vt.sign()*kinetic*jn)
    return torch.stack((1-(jn+jt*local)/norm,speed+(jn*local-jt)/norm),-1),stick


def factual(theta, seed, lane, context):
    vel,stick=impact(theta,0.,FACTUAL_INCOMING,context,'global_null')
    if not bool(stick.all()):
        raise ValueError('Factual support violates sticking assumption')
    mean=torch.cat((contour(theta,FY,context),vel),-1)
    return (mean+t(stream(seed,lane).normal(size=tuple(mean.shape)))*sigmas()[0]).cpu().numpy()


def calibration(seed,alpha,context,regime,start):
    theta=draw_object(1024,seed,10,alpha,context)
    x=factual(theta,seed,11,context)
    ids=stream(seed,98).permutation(1024)[:4]
    dirs=directions('alternating',start,4)
    dirs[-1]=1
    far,far_stick=impact(theta[ids[:3]],0.,t(dirs[:3])*abs(PROBE_INCOMING),context,regime)
    if bool(far_stick.any()):
        raise ValueError('Conventional calibration probe stuck')
    near,near_stick=impact(theta[ids[3:]],MATCH_HEIGHT,MATCH_SPEED,context,regime)
    noise=t(stream(seed,99).normal(size=(4,2)))*sigmas()[P]
    y=(torch.cat((far,near),0)+noise).cpu().numpy()
    return (x,ids,y,dirs),theta[ids[3]].cpu().numpy(),near[0].cpu().numpy(),bool(near_stick[0])


def deployment(seed,alpha,context,regime,repeats=32):
    theta=draw_object(128,seed,20,alpha,context)
    x=factual(theta,seed,21,context)
    height=context[2]
    _,label=impact(theta,height,QUERY_INCOMING,context,regime)
    vel,probe_stick=impact(theta,0.,PROBE_INCOMING,context,regime)
    if bool(probe_stick.any()):
        raise ValueError('Conventional deployment probe stuck')
    means={Q:contour(theta,view_heights(height),context),P:vel}
    obs={}
    noise_scale=sigmas()
    for a,mean in means.items():
        noise=t(stream(seed,40+a).normal(size=(len(mean),repeats,mean.shape[-1])))
        obs[a]=(mean[:,None]+noise*noise_scale[a]).cpu().numpy()
    return x,label.cpu().numpy(),theta.cpu().numpy(),obs


def pair_seed(panel,family,height):
    return 1941010000+panel*1000+FAMILIES.index(family)*100+HEIGHTS.index(height)*10


def direction_pairs(panel,family,height,n=PAIR_COUNT):
    """Return registered profile pairs. Internal latent objects/amplitudes are not estimators' inputs."""
    if not 1<=n<=PAIR_COUNT:
        raise ValueError(n)
    seed=pair_seed(panel,family,height)
    heights=view_heights(height)
    # Eight latent populations equally represented, on a dedicated independent object stream.
    theta=torch.cat([draw_theta(4,seed+1000000+100000*j,73,alpha)
                     for j,alpha in enumerate(ALPHAS)])
    location=profile(theta,heights)+base_profile(heights,family)[1]
    offset=deformation(heights,family)[1]
    amplitude=t(stream(seed,91).uniform(-1.,1.,(PAIR_COUNT,2,1)))
    noise=t(stream(seed,92).normal(size=(PAIR_COUNT,2,7)))*sigmas()[Q]
    pairs=location[:,None,:]+amplitude*offset+noise
    order=stream(seed,93).permutation(PAIR_COUNT)
    return pairs[order[:n]].cpu().numpy()


def estimate_direction(pairs):
    """Whiten paired differences then calculate centered PCA. Output sign is arbitrary."""
    pairs=np.asarray(pairs,dtype=np.float64)
    if pairs.ndim!=3 or pairs.shape[1:]!=(2,7) or len(pairs)<3 or not np.isfinite(pairs).all():
        raise ValueError('Expected at least three finite two-view 7-coordinate pairs')
    # v35 calibrated query-view coordinate noise in metres, available to every method.
    difference=(pairs[:,1]-pairs[:,0])/.00030
    difference-=difference.mean(axis=0)
    _,_,vt=np.linalg.svd(difference,full_matrices=False)
    return vt[0].copy()


def oracle_direction(family,height):
    raw=deformation(view_heights(height),family)[1]/sigmas()[Q]
    return (raw/raw.norm()).cpu().numpy()
