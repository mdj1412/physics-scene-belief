"""Factorial intervention in optical readout and loaded contact. Coordinates in mm.

Only camera_shape and install_world43 participate in pre-impact geometry.
sensor_delta_mm modifies the paid range profile alone. loaded_shape is accessible
only to the evaluator. A matched object has the same pre-impact appearance and
all the same purchased observations for both values of mechanical coupling.
"""
from functools import lru_cache
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
FAMILIES = ("keel", "crown", "channel", "bead")
HEIGHTS = (19, 31)
OFFSETS = (-.5, 0., .5)
VARIANTS = ("hollow", "bulge")
ALIGNMENTS = ("low", "high")
COUPLINGS = ("null", "strong")
RADII = (.25, 2.)
SIGN = {"hollow": -1., "base": 0., "bulge": 1.}
COS2 = {"low": .1, "high": .9}
OPTICAL_NORM_MM = 1.20
LOAD_SLOPE = .14

# C2 compact segments with different signs, positions, widths and number of lobes.
# All wings have zero support on [16.5,21.5] and [28.5,33.5] mm.
WINGS = {
    "keel": ((25.0, 3.1, -.78),),
    "crown": ((14.0, 2.0, .48), (25.0, 3.1, .68), (36.0, 2.0, .38)),
    "channel": ((13.6, 2.4, -.72), (24.3, 2.5, .40), (36.0, 2.1, -.62)),
    "bead": ((13.4, 2.4, .56), (23.3, 1.55, -.40), (26.5, 1.45, .70), (36.2, 2.4, .42)),
}
# Four distinct PRESCRIBED responses under load; these parameters are not fitted.
# In the evaluated [-2.5,2.5] window the respective additive slopes are:
# constant, increasing quadratically in x, increasing linearly in x, and oscillatory.
# All four have the same nominal tangent response.
OPTICAL_REFERENCE = {
    "keel": [1, -2, 1, 0, -1, 2, -1],
    "crown": [-1, 2, -1, 2, -1, 0, 1],
    "channel": [1, 1, -2, 0, 2, -1, -1],
    "bead": [-1, 2, 0, -2, 2, -1, 0],
}


def camera_shape(x_mm, family, variant="base"):
    """Unloaded visible extruded contour. A neutral, common curvature crosses both sites."""
    x = np.asarray(x_mm, dtype=np.float64)
    if variant not in SIGN:
        raise ValueError(variant)
    # Zero value and derivative on the factual measurement interval [-10,+10].
    # Two local quartic stations have C2 joins; windows of the two heights are disjoint.
    f, g = np.zeros_like(x), np.zeros_like(x)
    for center, width, amplitude in ((19., 4.4, .22), (31., 4.1, -.18), *WINGS[family]):
        z = (x - center) / width
        q = np.maximum(1. - z*z, 0.)
        f += amplitude*q**3
        g += -6.*amplitude*z*q*q/width
    return f, g


@lru_cache(None)
def source_axis(height_mm):
    h = int(height_mm)
    if h != height_mm or h not in HEIGHTS:
        raise ValueError(height_mm)
    saved = json.loads((ROOT.parent / "image_pressure_v45/source_axis45.json").read_text())
    u = np.asarray(saved[str(h)]["axis"], dtype=np.float64)
    np.testing.assert_allclose(u@u, 1., atol=2e-12)
    return u


def sensor_delta_mm(family, height_mm, alignment, variant):
    """Seven equal-noise query channels, same norm at both assigned alignments."""
    u = source_axis(height_mm)
    v = np.asarray(OPTICAL_REFERENCE[family], dtype=np.float64)
    orthogonal = v - (v@u)*u
    orthogonal /= np.linalg.norm(orthogonal)
    a = COS2[alignment]
    return SIGN[variant] * OPTICAL_NORM_MM * (np.sqrt(a)*u + np.sqrt(1.-a)*orthogonal)


def loaded_shape(x_mm, height_mm, family, variant, coupling):
    """Deformation in a loaded cross-section referred to the nominal contact station.

    The shape is used within ±2.5 mm only. The imposed scalar load mode describes
    a prescribed deflection at load, independent of the range-sensor response;
    this is not a continuum elasticity solve.
    """
    x = np.asarray(x_mm, dtype=np.float64)
    b, g = camera_shape(x, family, variant)
    if coupling == "null":
        return b, g
    if coupling != "strong":
        raise ValueError(coupling)
    d = (x - height_mm) / 2.5
    if np.any(abs(d)>1.+1e-12):
        raise ValueError("Loaded evaluation is defined on the local contact patch")
    if family == "keel":
        integral, gradient = d, np.ones_like(d)
    elif family == "crown":
        integral, gradient = d + .40*d**3/3., 1.+.40*d*d
    elif family == "channel":
        integral, gradient = d + .35*d*d, 1.+.70*d
    else:
        integral = d + .40*(1.-np.cos(np.pi*d))/np.pi
        gradient = 1.+.40*np.sin(np.pi*d)
    amount = SIGN[variant] * LOAD_SLOPE
    return b + (2.5*amount)*integral, g + amount*gradient


def install_world43(world):
    """Install only the unloaded pre-impact shape in the sensor generator."""
    if getattr(world, "_v47_installed", False):
        return
    old = world.perturbation
    def perturbation(y, shape, variant):
        if shape not in FAMILIES:
            return old(y, shape, variant)
        y = world.t(y)
        f, g = camera_shape(y.detach().cpu().numpy()*1000., shape, variant)
        return (world.torch.as_tensor(g, device=y.device, dtype=y.dtype),
                world.torch.as_tensor(f*.001, device=y.device, dtype=y.dtype))
    world.perturbation = perturbation
    world.TARGET_SHAPES = FAMILIES
    world._v47_installed = True


def install_camera(camera46):
    if getattr(camera46, "_v47_installed", False):
        return
    old = camera46.shape_function
    camera46.shape_function = lambda shape: camera_shape if shape in FAMILIES else old(shape)
    camera46._v47_installed = True


def render_clip(theta, family, variant, h_mm, offset_mm, seed, device="cpu"):
    sys.path.insert(0, str(ROOT.parent / "normal_variance_v46"))
    import camera46
    install_camera(camera46)
    return camera46.render_clip(theta, family, variant, h_mm, offset_mm, seed, device)
