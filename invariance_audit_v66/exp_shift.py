"""Label-free covariate audit relative to source training normalization."""
import json
import os
from pathlib import Path

import numpy as np
import torch
from exp_readout import ROOT, SOURCE, TARGET, cells


def summarize(base, panels):
    state=torch.load(ROOT/"exp_fit_full_66101.pt",map_location="cpu",weights_only=True)
    mu=state["mean"][-7:].numpy()
    scale=state["scale"][-7:].numpy()
    count=0
    large3=large5=0
    zs=np.zeros(7)
    for p in panels:
        for _,h,stem in cells(base):
            with np.load(base/"results"/f"panel_{p:02d}"/stem) as a:
                q=a["query_paid"]*1000.
            if base==SOURCE and max(panels)<8:
                q=q[:,:,:2]
            q=q[np.isfinite(q).all(-1)]
            z=(q-mu)/scale
            zs+=z.sum(0)
            count+=len(z)
            large3+=int((abs(z)>3).sum())
            large5+=int((abs(z)>5).sum())
    return dict(count=count, frac_over3=large3/(count*7),frac_over5=large5/(count*7),
                average_z_per_channel=(zs/count).tolist())


if __name__=="__main__":
    assert os.getenv("SLURM_JOB_ID") and os.getenv("CUDA_VISIBLE_DEVICES")==""
    result=dict(train=summarize(SOURCE,range(8)),source_valid=summarize(SOURCE,range(8,12)),
                archived_target=summarize(TARGET,range(12)))
    (ROOT/"exp_shift.json").write_text(json.dumps(result,indent=2)+"\n")
