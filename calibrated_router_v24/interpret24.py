"""Supplemental within-calibration scoring and cheap calibration-only control, with frozen test decisions."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/"cheap_tail_v24"))
from stats24 import boot_means
import numpy as np
import json,os
from scipy.stats import spearmanr
R=Path("results")
assert os.environ.get("SLURM_JOB_ID")
cfg=json.loads((R/"selected.json").read_text());world=np.load(R/"test/private.npz")
cell=world["cell"];a=np.stack([np.load(R/f"choices_{b}.npz")["acts"] for b in range(5)])-1
c=np.stack([np.load(R/f"eval_{b}.npz")["loss"].mean(-1) for b in range(5)])
fs=np.load(R/"choices_0.npz")["fire"];p=np.take_along_axis(c,a,axis=3)
# Majority route is calibrated on PUBLIC factual draws of the evaluated batch. Label-free transductive diagnostic only.
ref=np.where((fs.mean(0)>=.5)[None,None,:],p[...,3],p[...,2])
d=(p[...,0]-ref)[...,None]
boot=boot_means(d,cell,5000)
res={}
for j,cal in enumerate(("camera","probe","balanced")):
    unique=np.any(a[:,:,j,0]!=a[:,:,j,1],axis=0)
    fixed="high" if fs[:,j].mean()>=.5 else "low"
    vals=c[:,:,j].mean((0,1))
    res[cal]=dict(gate_actions=np.bincount(a[:,:,j,0].ravel(),minlength=3).tolist(),
        reference_actions=np.bincount(a[:,:,j,1].ravel(),minlength=3).tolist(),
        single_action_loss=vals.tolist(),majority=fixed,loss=float(ref[:,:,j].mean()),delta=float(d[:,:,j].mean()),
        ci=np.quantile(boot[:,j,0],[.025,.975]).tolist(),divergent_worlds=int(unique.sum()),
        majority_worlds=int((fs[:,j]==(fixed=="high")).sum()))
(R/"localization.json").write_text(json.dumps(res,indent=2))
