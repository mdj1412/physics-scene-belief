"""Exact finite-population gate-only replay, separated from deployable calibration."""
import csv
import json
import math
import os
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREV = ROOT.parent / "racket_transfer_v70"
BUDGETS = (0, 1, 2, 4, 8, 16, 32)
R = .02


def evaluate_row(row, fit):
    state = [float(row[k]) * s for k, s in
             (("v_in_x", 1), ("v_in_z", 1), ("spin_before_y", R))]
    a = [1.] + [(v-m)/s for v, m, s in zip(state, fit["mean"], fit["std"])]
    old = sum(v*b for v,b in zip(a,fit["logistic"])) > 0
    truth = [float(row[k]) * s for k,s in
             (("v_out_x",1),("v_out_z",1),("spin_after_y",R))]
    label = truth[0]-truth[2] > 0
    losses = []
    for route in (old, not old):
        coeff = fit["expert_plus" if route else "expert_minus"]
        out = [sum(a[i]*coeff[i][j] for i in range(4)) for j in range(3)]
        losses.append((math.dist(out[:2],truth[:2]), abs(out[2]-truth[2])/R))
    return old != label, losses


def exact_probability(total, mismatches, n, required):
    if required>n: return 0.
    lower = max(required,n-total+mismatches)
    upper = min(n,mismatches)
    if lower>upper: return 0.
    return sum(math.comb(mismatches,k)*math.comb(total-mismatches,n-k)
               for k in range(lower,upper+1)) / math.comb(total,n)


def critical(n, alpha):
    for k in range(n+1):
        if sum(math.comb(n,i) for i in range(k,n+1)) / 2**n <= alpha:
            return k
    return n+1


def run():
    fits=json.loads((PREV/"solver"/"metrics.json").read_text())["folds"]
    output={}
    for name,params in fits.items():
        configs={}
        by_split={"train":[], "test":[]}
        for config in params["held"]:
            conf={}
            for split in by_split:
                p=PREV/"data"/"upstream"/split/f"racket_{config:02}.csv"
                with p.open(newline="") as handle:
                    vals=[evaluate_row(row,params["empirical"]) for row in csv.DictReader(handle)]
                by_split[split].extend(vals)
                conf[split]={"n":len(vals),"mismatched":sum(flag for flag,_ in vals)}
            configs[str(config)]=conf
        train=by_split["train"]; test=by_split["test"]
        n=len(train); bad=sum(flag for flag,_ in train)
        frozen=[sum(loss[0][j] for _,loss in test)/len(test) for j in (0,1)]
        flipped=[sum(loss[1][j] for _,loss in test)/len(test) for j in (0,1)]
        ref={}
        for budget in BUDGETS:
            rules={"strict_majority":budget//2+1,
                   "binomial_one_sided_05":critical(budget,.05),
                   "binomial_one_sided_10":critical(budget,.10)}
            ref[budget]={}
            for rule, threshold in rules.items():
                p=exact_probability(n,bad,budget,threshold)
                ref[budget][rule]={"min_mismatches":threshold,"p_flip":p,
                     "expected_v":frozen[0]+p*(flipped[0]-frozen[0]),
                     "expected_w":frozen[1]+p*(flipped[1]-frozen[1])}
        output[name]={"n_pool":n,"mismatches":bad,"configs":configs,
                      "test_n":len(test),"test_disagreements":sum(f for f,_ in test),
                      "loss_original":frozen,"loss_flipped":flipped,"budgets":ref}
    meta={"node":platform.node(),"job":os.getenv("SLURM_JOB_ID"),
          "step":os.getenv("SLURM_STEP_ID"),"rubbers":output}
    (ROOT/"review"/"finite_reference.json").write_text(json.dumps(meta,indent=2)+"\n")

if __name__=="__main__":run()
