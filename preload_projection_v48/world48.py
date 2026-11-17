"""Unloaded silhouettes, optical profiles, and loaded cross-sections. Lengths in mm."""
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "factorial_projection_v47"))
import world47 as source

SOURCE = source.FAMILIES
FAMILIES = ("ogee", "cove", "reed", "flute")
HEIGHTS = source.HEIGHTS
ALIGNMENTS = source.ALIGNMENTS
VARIANTS = source.VARIANTS
RADII = source.RADII
OFFSETS = source.OFFSETS
STATES = (0, -1, 1)
STATE_WEIGHTS = (.5, .25, .25)
source_axis = source.source_axis

# Compact C2 lobes outside both ±2.5-mm measurement windows.
WINGS = {
    "ogee": ((12.9,2.4,-.48), (23.5,1.8,.72), (26.3,1.9,-.61)),
    "cove": ((13.4,2.55,.25), (24.6,2.75,-.98), (36.8,2.9,.34)),
    "reed": ((13.7,2.15,.35), (23.6,1.35,.46), (25.5,1.1,.60), (37.1,2.75,-.55)),
    "flute": ((12.9,1.80,-.32), (24.0,2.4,.85), (27.0,1.1,-.32), (35.9,2.0,-.73)),
}
OPTICAL = {
    "ogee": (1,1,-1,-2,0,2,-1), "cove": (1,-1,0,2,-2,1,-1),
    "reed": (-1,2,-2,1,-1,2,0), "flute": (0,-2,1,2,-1,-1,1),
}
# The common unloaded contact stations are new; local neighborhoods match between families.
STATIONS = ((19.,4.45,.195), (31.,4.2,-.225))
SLOPE = {"ogee":.135, "cove":.158, "reed":.120, "flute":.143}


def camera_shape(x_mm, family, variant="base"):
    if family in SOURCE:
        return source.camera_shape(x_mm, family, variant)
    if variant not in ("base", *VARIANTS):
        raise ValueError(variant)
    x = np.asarray(x_mm, dtype=float)
    z, slope = np.zeros_like(x), np.zeros_like(x)
    for at, width, magnitude in (*STATIONS, *WINGS[family]):
        t = (x-at)/width
        q = np.maximum(1.-t*t, 0.)
        z += magnitude*q**3
        slope += -6*magnitude*t*q*q/width
    return z, slope


def sensor_delta_mm(family, height_mm, alignment, variant):
    if family in SOURCE:
        return source.sensor_delta_mm(family, height_mm, alignment, variant)
    u = source_axis(height_mm)
    w = np.array(OPTICAL[family], float)
    w -= (w@u)*u
    w /= np.linalg.norm(w)
    a = source.COS2[alignment]
    return source.SIGN[variant]*1.2*(np.sqrt(a)*u+np.sqrt(1.-a)*w)


def loaded_shape(x_mm, height_mm, family, signed_state):
    """Optical readout never enters the full-load fixture. Signed_state sets only bending."""
    if signed_state not in STATES:
        raise ValueError(signed_state)
    if family in SOURCE:
        return source.loaded_shape(x_mm, height_mm, family,
            "hollow" if signed_state == -1 else "bulge", "null" if signed_state == 0 else "strong")
    x = np.asarray(x_mm, dtype=float)
    z, slope = camera_shape(x, family)
    t = (x-height_mm)/2.5
    if np.any(abs(t)>1.+1e-10):
        raise ValueError("Outside contact neighborhood")
    if family == "ogee":
        mode, gradient = t+.28*t*t/2-.20*t**4/4, 1.+.28*t-.20*t**3
    elif family == "cove":
        mode, gradient = .78*t+.22*np.sin(np.pi*t)/np.pi, .78+.22*np.cos(np.pi*t)
    elif family == "reed":
        mode, gradient = t-.45*t**3/3+.16*t**5/5, 1.-.45*t*t+.16*t**4
    else:
        mode, gradient = t+.32*(1.-np.cos(2.*np.pi*t))/(2.*np.pi), 1.+.32*np.sin(2.*np.pi*t)
    return z+2.5*signed_state*SLOPE[family]*mode, slope+signed_state*SLOPE[family]*gradient


def install_world43(world):
    if getattr(world, "_v48_installed", False):
        return
    old = world.perturbation
    def perturbation(y, shape, variant):
        if shape not in FAMILIES:
            return old(y,shape,variant)
        y = world.t(y)
        f,g = camera_shape(y.detach().cpu().numpy()*1000.,shape,variant)
        return (world.torch.as_tensor(g, device=y.device, dtype=y.dtype),
                world.torch.as_tensor(f*.001, device=y.device, dtype=y.dtype))
    world.perturbation = perturbation
    world.TARGET_SHAPES = FAMILIES
    world._v48_installed = True


def render_clip(theta, family, variant, h_mm, offset_mm, seed, device="cpu"):
    sys.path.insert(0,str(ROOT.parent/"normal_variance_v46"))
    import camera46
    if not getattr(camera46, "_v48_installed", False):
        old = camera46.shape_function
        camera46.shape_function = lambda shape: camera_shape if shape in FAMILIES else old(shape)
        camera46._v48_installed = True
    return camera46.render_clip(theta,family,variant,h_mm,offset_mm,seed,device)
