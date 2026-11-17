"""Exact per-bank factorial accounting for final purchases under error switches."""
import json
import os
from pathlib import Path
import numpy as np

H = Path(__file__).resolve().parent
CATEGORIES = ("baseline_already_flips", "marker_alone_only", "delay_alone_only",
              "either_alone_suffices", "requires_both", "suppressed_baseline_flip")
# 0 identical, 1 slot1, 2 clean, 3 marker, 4 delay, 5 both, 6 corrected delay, 7 corrected both.
def main():
    assert os.environ.get("SLURM_JOB_ID") and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    counts = np.zeros((24,2,2,len(CATEGORIES)), dtype=np.int64)
    check = np.zeros((24,2,2),dtype=np.int64)
    ref_total=0
    for i in range(24):
        name=f"panel_{i:02d}.npz"
        with np.load(H/"blocks"/name) as z: routes=z["routes"]
        with np.load(H.parent/"reflective_tracking_v63"/"scored63"/name) as z: gate=z["gate"]
        mats=((np.arange(4)+i)%4)//2
        n_ref=np.array([np.count_nonzero(~gate[:,mats==m].astype(bool)) for m in (0,1)])
        n_all=np.array([gate[:,mats==m].size for m in (0,1)])
        assert (n_all==2560).all()
        ref_total += n_ref.sum()
        for d in range(4):
            for arm in (0,1):
                b=d^arm
                left=routes[d,1]
                disagrees=routes[b]!=left
                clean,marker=disagrees[2:4]
                for j,(yd,joint) in enumerate(((4,5),(6,7))):
                    delay=disagrees[yd]; both=disagrees[joint]
                    born=both & ~clean
                    tags=(both & clean, born&marker&~delay, born&delay&~marker,
                          born&marker&delay,born&~marker&~delay,clean&~both)
                    weights=n_ref[None,None,:,None]
                    counts[i,arm,j] += [np.sum(a*weights) for a in tags]
                    check[i,arm,j] += np.sum(both*weights)
    assert np.array_equal(counts[...,:5].sum(-1),check)
    primary=json.loads((H/'results67.json').read_text())
    for arm in range(2):
        for j,case in enumerate((5,7)):
            assert check[:,arm,j].sum()==primary["terminal_counts"][arm][case]
    denom=primary['bank_prefix_denominator_per_cell']
    # Paired leave-one diagnostics separately across two sources of variability.
    with np.load(H/"aggregate67.npz") as z:
        blocks=z["by_catalog_board"].mean((1,2)) # catalog,arm,case,board,metric
    paired=blocks[:,1,7,:,8]-blocks[:,0,7,:,8]
    omit_catalog=(paired.sum()-paired.sum(1))/((24-1)*8)
    omit_board=(paired.sum()-paired.sum(0))/(24*(8-1))
    totals=counts.sum(0)
    out=dict(categories=CATEGORIES,arms=["sham","two lights"],cells=["raw","corrected"],
        counts=totals.tolist(), rates_percent=(totals*100/denom).tolist(),
        fraction_within_joint_terminal=(totals[...,:5]/check.sum(0)[...,None]).tolist(),
        reference_prefixes=int(ref_total),denominator_per_arm=denom,
        primary_excess_leave_one_catalog=[float(omit_catalog.min()),float(omit_catalog.max())],
        primary_excess_leave_one_board=[float(omit_board.min()),float(omit_board.max())],
        estimand="paired potential purchases with common sensor/noise tape at imposed rates; categories nested on joint terminal flip",
        job=os.environ["SLURM_JOB_ID"])
    (H/"attribution67.json").write_text(json.dumps(out,indent=2)+"\n")
if __name__=="__main__":
    main()
