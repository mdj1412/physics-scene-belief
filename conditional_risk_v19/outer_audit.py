"""Exploratory test: keep the discrete scene support fixed; perturb only outer RQMC."""
from selector import Selector, SEEDS, TASKS, plans, action
from common import require, write, save
from model import t
from pathlib import Path
import torch
import numpy as np

require()
torch.set_num_threads(2)
root = Path(__file__).resolve().parent / "results"
public = np.load(root / "data/test/public.npz")
policy = np.load(root / "policy_0_1_0.npz")
selector = Selector(SEEDS[0])
streams = []
for k in range(8):
    draws = plans(510001 + k * 2000)
    streams.append((torch.stack([u for u, _ in draws]), torch.stack([e for _, e in draws])))
output = []
masses = []
pure = []
for i in range(512):
    state = selector.fit(t(public["factual"][i,0]), t(public["context"][i,0]), TASKS[1], False)
    w, z = state[:2]
    ix = torch.searchsorted(w.cumsum(0), selector.reps).clamp_max(len(w)-1)
    labels = z[ix,0]
    pure.append(bool(((labels.max(1).values == labels.min(1).values).all()).item()))
    masses.append(float(w @ (z[:,0] > 0).to(w.dtype)))
    if i >= 64:
        continue
    scene = []
    for u, eps in streams:
        selector.u, selector.eps = u, eps
        pair = [selector.mixture_risks(state, method, 4096, prefixes=True)
                for method in ("conditional", "self_label")]
        scene.append([[r[n].cpu().numpy() for n in (512, 4096)] for r in pair])
    output.append(scene)
risk = np.array(output)  # world, outer stream, method, size, action
gap = risk[...,2] - risk[...,0]
variance = gap.var(axis=1, ddof=1)
a = np.where(np.ptp(risk, axis=-1) <= 1e-12, 0, risk.argmin(-1))
consistency = np.mean([a[:,h] == a[:,j] for h in range(8) for j in range(h)], axis=(0,1))
tie = np.ptp(policy["risks"][:,2],axis=-1) <= 1e-12
masses = np.array(masses)
small = np.minimum(masses,1-masses).clip(min=0)
save(root/"outer_audit.npz", risks=risk, pure=pure, mass=masses)
write(root/"outer_audit.json", dict(worlds=64, streams=8, methods=["conditional","self_label"], M=[512,4096],
      rms_within_seed_sd=np.sqrt(variance.mean(0)).tolist(), pairwise_seed_agreement=consistency.tolist(),
      ties=int(tie.sum()), pure_both=int(np.sum(pure)), tied_and_pure=int(np.sum(tie & pure)),
      tie_minority_mass_quantiles=np.quantile(small[tie],[0,.5,.9,1]).tolist(),
      tied_mass_above_1pct=int(np.sum(tie & (small>.01))), tied_mass_above_1permille=int(np.sum(tie & (small>.001)))))
