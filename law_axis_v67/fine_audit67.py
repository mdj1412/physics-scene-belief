"""Recompute all new-law labels at doubled quadrature; re-score the sealed heads."""
import itertools
import json
import os
from pathlib import Path
import numpy as np
import score67 as scoring

H=Path(__file__).resolve().parent
S=scoring.SRC


def main():
    assert os.getenv("SLURM_JOB_ID") and os.environ.get("CUDA_VISIBLE_DEVICES")==""
    seal=json.loads((H/"forecast_seal.json").read_text())
    for fn,hs in seal["shards"].items():
        assert scoring.sha(H/"forecasts"/fn)==hs
    solver=scoring.phys
    solver.ROOT=S
    solver.output_folder=lambda high:H/"all_fine_check"
    solver.slope=lambda x,f,v:scoring.profile(x,f,v)[1]
    solver.material=scoring.law
    solver.LAWS=scoring.NEW
    channels=["soft_minus_full","restored_minus_dropped"]
    difference=np.zeros((12,2,2,4,2))
    label_errors=np.zeros((12,2,2))
    count=0
    for k in range(12):
        sub=scoring.panel(k)
        with np.load(H/"forecasts"/(sub+".npz")) as z:
            pp=z["p"]; names=z["cells"].tolist()
        for fi,f in enumerate(scoring.FAM):
            for v,h in itertools.product(scoring.VAR,scoring.HT):
                solver.run_group(k,f,v,h,True)
                with np.load(S/"routes"/sub/f"{f}_{v}_h{h}.npz") as z: route=z["route"]
                for s in scoring.SURF:
                    for lidx,l in enumerate(scoring.NEW):
                        fn=scoring.name(f,s,l,v,h)
                        with np.load(H/"all_fine_check"/sub/fn) as z: yfine=z["labels"][1:].astype(float)
                        with np.load(H/"labels"/sub/fn) as z: ybase=z["labels"][1:].astype(float)
                        dy=yfine-ybase
                        label_errors[k,fi,lidx]+=(dy!=0).mean()/12
                        count+=dy.size
                        if not dy.any(): continue
                        center=dy.mean(0)[...,None]
                        for acq in scoring.OLD:
                            row=pp[names.index(scoring.name(f,s,acq,v,h))].astype(float)
                            soft=(row[:,1]-row[:,0])[:,None]*route[None,...,None]
                            rescue=(row[:,4]-row[:,3])[:,None]
                            for j,changed in enumerate((soft,rescue)):
                                difference[k,fi,lidx,:,j]+=(-2*changed*center[None]).mean((1,2,3,4))/24
    res=dict(families=scoring.FAM,laws=scoring.NEW,contrasts=channels,new_label_comparisons=int(count),
           disagreement_by_family_law=label_errors.mean(0).tolist(),fine_minus_standard_brier_contrasts=difference.mean((0,3)).tolist(),
           max_individual_catalog_axis_abs=float(np.max(abs(difference))),job=os.environ["SLURM_JOB_ID"])
    (H/"fine_audit67.json").write_text(json.dumps(res,indent=2)+"\n")
if __name__=="__main__": main()
