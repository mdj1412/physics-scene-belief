"""Evaluator geometry shared by the held-out renderer, sensors and collision models.

x and f are in mm; g=df/dx is dimensionless. The common six latent parameters
from the original catalog are added outside shape_profile, exactly once.
"""
import numpy as np

SHAPES = ("terrace", "dimple")
VARIANTS = ("base", "hollow", "bulge")
_VARIATIONS = {"base": 0., "hollow": -1., "bulge": 1.}


def terrace(x, s):
    """Raised shelf with smooth entry and exit; change ramp width and center."""
    f=np.zeros_like(x,dtype=float)
    g=np.zeros_like(f)
    for center, width, amplitude, shift, width_change in (
        (18.3, 6.2, 0.90, 1.20, .18),
        (31.8, 5.0, -.70, -.85, -.22)
    ):
        c=center+s*shift
        w=width*(1.+width_change*s)
        u=np.clip((x-c)/w+.5,0.,1.)
        f+=amplitude*u**3*(10.-15.*u+6.*u*u)
        g+=(amplitude/w)*30.*u*u*(1.-u)**2
    return f,g


def dimple(x, s):
    """Two Gaussian depressions with different, perturbed widths and depths."""
    f=np.zeros_like(x,dtype=float)
    g=np.zeros_like(f)
    for center,width,depth,shift,width_change,depth_change in (
        (19.7,2.00,.68,-.65,.28,.35),
        (30.3,2.75,.95, .90,-.17,-.22)
    ):
        c=center+s*shift
        w=width*(1.+width_change*s)
        d=depth*(1.+depth_change*s)
        z=(x-c)/w
        basis=-d*np.exp(-.5*z*z)
        deriv=-z*basis/w
        ref=-d*np.exp(-.5*(c/w)**2)
        dref=c*ref/(w*w)
        f+=basis-ref-dref*x
        g+=deriv-dref
    return f,g


def shape_profile(x, shape, variant):
    x=np.asarray(x,dtype=np.float64)
    variation=_VARIATIONS[variant]
    if shape=="terrace":
        return terrace(x,variation)
    if shape=="dimple":
        return dimple(x,variation)
    raise ValueError(shape)


def install_world43(world):
    """Supply the new profile to the inherited evaluator-only sensor generator."""
    if getattr(world,"_v46_installed",False):
        return
    original=world.perturbation
    def perturbation(y,shape,variant):
        if shape not in SHAPES:
            return original(y,shape,variant)
        y=world.t(y)
        f,g=shape_profile(y.detach().cpu().numpy()*1000.,shape,variant)
        def as_tensor(a):
            return world.torch.as_tensor(a,device=y.device,dtype=y.dtype)
        return as_tensor(g),as_tensor(f*.001)
    world.perturbation=perturbation
    world.TARGET_SHAPES=SHAPES
    world._v46_installed=True
