"""Source-trained conditional projection risk from a single paid profile and scalar preload."""
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
H = (19, 31)
SIGNED_CUTS = (-.060, -.025, .025, .060) # mm
UNSIGNED_CUTS = (.025, .060)
MODES = ("local", "remote", "unsigned", "none")
BINS = (5, 5, 3, 1)


def setup(raw, device="cpu"):
    tensors = ("residual", "axes", "residual_axes", "profile_map")
    a = {k:torch.as_tensor(raw[k],dtype=torch.float64,device=device) for k in tensors}
    a["signed_cuts"] = a["axes"].new_tensor(SIGNED_CUTS)
    a["unsigned_cuts"] = a["axes"].new_tensor(UNSIGNED_CUTS)
    i, j = torch.triu_indices(7, 7, device=device)
    a["i"], a["j"] = i, j
    if "coefficients" in raw:
        a["coef"] = {k:a["axes"].new_tensor(v) for k,v in raw["coefficients"].items()}
        a["fixed_height_mm"] = raw["fixed_height_mm"]
    return a


def load_policy(device="cpu"):
    return setup(json.loads((ROOT/"policy48.json").read_text()), device)


def profile_features(query_paid, height_mm, a):
    """36 bounded features: intercept, 7 main effects and their 28 quadratic combinations."""
    h = H.index(height_mm)
    q = torch.as_tensor(query_paid,dtype=torch.float64,device=a["axes"].device)
    if q.shape[-1]!=7:
        raise ValueError("Expected 7 paid profile channels in metres")
    paid = torch.isfinite(q).all(-1)
    mm = torch.nan_to_num(q) * 1000.
    poly = mm @ a["profile_map"]
    r = mm @ a["residual"]
    energy = r.square().sum(-1)
    axis_r = r @ a["residual_axes"][h]
    features = torch.stack((poly[...,0] / (4. if h==0 else 7.), poly[...,1]/2.5,
                            poly[...,2], (mm @ a["axes"][h]) / 3., axis_r,
                            (energy / 7.).sqrt() / .45 - 1.,
                            2 * axis_r.square()/energy.clamp_min(1e-15) - 1.),dim=-1).tanh()
    b = torch.cat((torch.ones_like(paid[...,None],dtype=features.dtype),features,
                   features[...,a["i"]] * features[...,a["j"]]),dim=-1)
    return b,paid


def decision(query_paid, displacement_mm, height_mm, mode, artifact):
    b,paid = profile_features(query_paid,height_mm,artifact)
    if mode in ("preload","fixed"):
        mode = "local" if mode=="preload" or height_mm==artifact["fixed_height_mm"] else "remote"
    if mode not in MODES:
        raise ValueError(mode)
    h = H.index(height_mm)
    risk = b @ artifact["coef"][mode][h].T
    if mode=="none":
        return paid & (risk[...,0]<0)
    d = torch.as_tensor(displacement_mm,device=b.device,dtype=b.dtype)
    if mode=="unsigned":
        d = d.abs()
        cuts = artifact["unsigned_cuts"]
    else:
        cuts = artifact["signed_cuts"]
    # Broadcast measurements over extra exposure axes (coupling state etc.) if needed.
    idx = torch.searchsorted(cuts,d.contiguous())
    target = torch.broadcast_shapes(idx.shape, paid.shape)
    picked = risk.expand(target+(risk.shape[-1],)).gather(-1,idx.expand(target)[...,None])[...,0]
    return paid & (picked < 0)
