"""Frozen v48 policies. Observable input is exactly one signed displacement (or none).

panel() is a label-free generator. f,h are indices; [M,weak,A,V,E,N,R].
The weak state indexes the observation at its OWN station; the evaluator integrates
its conditional law outside the policy. No hidden transfer parameters enter prediction.
"""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "preload_projection_v48"
sys.path.insert(0, str(SOURCE))
import route48
import sensor48
from world48 import FAMILIES, HEIGHTS, STATES

METHODS = ("original", "direct_none", "direct_local", "direct_fixed", "gate_local",
           "fixed_shared", "gate_fixed")


def update(prob, disp, weights):
    x = 2 * (prob - .5)
    z = np.clip(disp / .084, -2., 2.)
    a = abs(z)
    return np.clip(prob + weights[...,0] + weights[...,1]*x + weights[...,2]*x*x
                   + weights[...,3]*z + weights[...,4]*a + weights[...,5]*z*x
                   + weights[...,6]*a*x, 0., 1.)


class Policy:
    def __init__(self, device="cpu"):
        self.gate = route48.load_policy(device)
        src = json.loads((SOURCE/"direct_source48.json").read_text())
        self.coef = np.asarray(src["coefficients"])
        self.fixed = src["fixed_site"]
        assert self.fixed == self.gate["fixed_height_mm"] == 31

    def panel(self, idx):
        tag = f"panel_{idx:02d}.npz"
        with np.load(SOURCE/"worlds"/tag) as pop:
            seed = int(pop["seed"])
        with np.load(SOURCE/"forecast"/tag) as forecast:
            o, p, q = (forecast[k] for k in ("p_original", "p_projected", "query_paid"))
        for f, family in enumerate(FAMILIES):
            for h,height in enumerate(HEIGHTS):
                po, pp, paid = o[f,:,h], p[f,:,h], q[f,:,h]
                arm = np.isfinite(paid[...,0]).astype(np.intp)
                assert np.max(abs(pp[arm==0] - po[arm==0]), initial=0.) < 2e-10
                b,pay = route48.profile_features(paid,height,self.gate)
                risk = {mode: b @ self.gate["coef"][mode][h].T for mode in ("local","remote")}
                no = update(po,0.,self.coef[0,h][arm])
                out = np.empty((len(METHODS),3,*po.shape),dtype=np.float64)
                for j,state in enumerate(STATES):
                    sig = sensor48.readings({"seed":seed},family,height,state,height)
                    fix = (sig if height==self.fixed else
                           sensor48.readings({"seed":seed},family,self.fixed,state,self.fixed))
                    for m,reading,mode in ((4,sig,"local"),(6,fix,"local" if height==self.fixed else "remote")):
                        ids=np.searchsorted(route48.SIGNED_CUTS,reading)
                        # Only the paid view can trigger projection; gate remains false in other arm.
                        which = np.take_along_axis(risk[mode].cpu().numpy(),
                                                   np.broadcast_to(ids,po.shape)[...,None],axis=-1)[...,0]
                        out[m,j] = np.where(pay.cpu().numpy() & (which < 0.),pp,po)
                    out[0,j] = po
                    out[1,j] = no
                    out[2,j] = update(po,sig,self.coef[1,h][arm])
                    mode = 1 if height==self.fixed else 2
                    out[3,j] = update(po,fix,self.coef[mode,h][arm])
                    out[5,j] = update(po,fix,self.coef[1,h][arm])
                if height == self.fixed:
                    np.testing.assert_array_equal(out[2],out[3])
                    np.testing.assert_array_equal(out[4],out[6])
                assert np.isfinite(out).all() and out.min() >= -1e-12 and out.max() <= 1. + 1e-12
                yield f,h,out
