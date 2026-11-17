"""Deployable route from pre-impact image statistics and one already-purchased query profile."""
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent


def load_policy(device="cpu"):
    raw = json.loads((ROOT / "policy47.json").read_text())
    names = ("residual_operator", "effective_axis", "x_grid", "variance_cuts",
             "legacy_risk", "alignment_cuts", "rms_cuts_mm", "risk")
    return {k: torch.tensor(raw[k], dtype=torch.float64, device=device) for k in names}


def feature_scores(query_paid, height_mm, artifact):
    """The residual is in mm. Masked/unpurchased query channels must contain NaN."""
    if height_mm not in (19, 31):
        raise ValueError(f"Unknown commanded height {height_mm}")
    y = torch.as_tensor(query_paid, dtype=torch.float64, device=artifact["risk"].device)
    if y.shape[-1] != 7:
        raise ValueError("Expected one seven-point paid query on the last dimension")
    valid = torch.isfinite(y).all(dim=-1)
    resid_mm = torch.nan_to_num(y) @ artifact["residual_operator"] * 1000.
    energy = resid_mm.square().sum(dim=-1)
    coordinate = resid_mm @ artifact["effective_axis"][int(height_mm == 31)]
    alignment = coordinate.square() / energy.clamp_min(1e-20)
    return alignment, (energy / 7).sqrt(), valid


def decision(query_paid, x_mm, variance, height_mm, artifact):
    """PyTorch-broadcastable leading dimensions; returns False for the physical-probe arm."""
    alignment, rms_mm, paid = feature_scores(query_paid, height_mm, artifact)
    x = torch.as_tensor(x_mm, device=paid.device)
    v = torch.as_tensor(variance, device=paid.device)
    b = (x[..., None] - artifact["x_grid"]).abs().argmin(-1)
    quartile = (v[..., None] > artifact["variance_cuts"][b]).sum(-1)
    legacy = (artifact["legacy_risk"][b, quartile] < 0).long()
    c = (alignment[..., None] > artifact["alignment_cuts"]).sum(-1)
    a = (rms_mm[..., None] > artifact["rms_cuts_mm"]).sum(-1)
    return paid & (artifact["risk"][b, legacy, c, a] < 0)
