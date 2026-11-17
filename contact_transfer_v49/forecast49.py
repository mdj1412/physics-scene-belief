"""Freeze and persist paired forecasts for evaluator, blind to solver outcomes."""
from pathlib import Path
import os
import time

import numpy as np
import torch

from policy49 import METHODS, Policy, ROOT


if __name__ == "__main__":
    assert os.getenv("SLURM_JOB_ID") and torch.cuda.device_count()==1
    torch.set_num_threads(1)
    dest=ROOT/"forecast49"
    dest.mkdir(exist_ok=True)
    model=Policy("cuda")
    with torch.inference_mode():
        for k in range(12):
            out=np.empty((len(METHODS),4,2,3,2,2,4,128,4),dtype=np.float64)
            for f,h,p in model.panel(k):
                out[:,f,h]=p
            path=dest/f"panel_{k:02d}.npz"
            tmp=path.with_suffix(".tmp")
            with tmp.open("wb") as stream:
                np.savez_compressed(stream,p=out,methods=METHODS,job=os.environ["SLURM_JOB_ID"],utc=time.time())
            tmp.replace(path)
            print(k,flush=True)
