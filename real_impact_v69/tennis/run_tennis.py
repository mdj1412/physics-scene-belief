"""Cross two figure cases: fixed thin-shell parameter panel, no fitted outcomes."""
import csv
import itertools
import os
from pathlib import Path
import sys
import platform
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "solver"))
from solver import bullet, compliant, tarray
import numpy as np
import torch

def main():
    x = np.array([[5.8, -6.8, 0.], [21.1, -6.8, 0.]])
    k = 2 / 3
    rec = []
    for e, mu in itertools.product([.65, .75, .85], [.2, .5, .8]):
        rigid = bullet(x, e, mu, .00002, inertia_ratio=k)
        check = -(np.minimum(x[:, 0] / (1 + 1/k), -mu * (1 + e) * x[:, 1]))
        np.testing.assert_allclose(rigid[:, 0], x[:, 0]+check, atol=2.e-6)
        branches = [("pybullet", 0., rigid[:,0])]
        ratios = [0., .125, .5, 1.]
        comp = compliant(tarray(x), e, tarray([mu]*len(ratios)), tarray(ratios),
                         q=160, inertia_ratio=k).detach().cpu().numpy()
        assert abs(comp[0,:,0]-rigid[:,0]).max() < 2.e-6
        for ratio, p in zip(ratios,comp[:,:,0]):
            branches.append(("compliant_coulomb" if ratio==0 else "compliant_memory",ratio,p))
        for model, ratio, vx in branches:
            for case, v in zip("AB",vx):
                rec.append(dict(case=case,model=model,e=e,mu=mu,kt_ratio=ratio,com_vx_m_s=v))
    with (ROOT/"tennis"/"predictions.csv").open("w") as f:
        writer=csv.DictWriter(f,fieldnames=rec[0])
        writer.writeheader()
        writer.writerows(rec)
    out = dict(node=platform.node(),slurm_job=os.getenv("SLURM_JOB_ID"),gpu=torch.cuda.get_device_name(),
               n_forecasts=len(rec),moment_ratio=k)
    (ROOT/"tennis"/"run.json").write_text(json.dumps(out,indent=2)+"\n")

if __name__ == "__main__":
    main()
