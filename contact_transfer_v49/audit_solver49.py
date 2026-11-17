"""Describe independent Bullet labels, paired refinement, and comparison with prior solver."""
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent
v48=ROOT.parent/"preload_projection_v48"
per_catalog=[]
fine_abs=[]; diff_at_flip=[]
minimum=float("inf")
for k in range(12):
    name=f"panel_{k:02d}.npz"
    with np.load(ROOT/"labels49"/name) as a, np.load(ROOT/"labels49_fine"/name) as b, np.load(v48/"labels"/name) as old:
        y=a["labels"]; f=b["labels"]; slots=b["object_slots"]; v=a["slip_speed"]
        for key in ("state","heights_mm","radii_mm","offsets_mm","families"):
            np.testing.assert_array_equal(a[key],old[key]); np.testing.assert_array_equal(a[key],b[key])
        np.testing.assert_array_equal(a["object_index"][:,slots],b["object_index"])
        matched=v[...,slots]
        fine_abs.append(np.abs(matched-b["slip_speed"]).ravel())
        diff=f!=y[...,slots]
        diff_at_flip.extend(np.abs(matched[diff]-.020).tolist())
        minimum=min(minimum,float(a["separation_speed"].min()))
        w=np.array([.5,.25,.25])
        def weighted(x):
            return float(w@x.mean(axis=(0,1,3,4,5,6)))
        per_catalog.append(dict(primary_stick_weighted=weighted(y),v48_stick_weighted=weighted(old["labels"]),
              new_vs_old_disagreement=weighted(y!=old["labels"]),refined_disagreement=weighted(diff),
              negative_positive_flip=float((y[:,:,1]!=y[:,:,2]).mean()),separating_fraction=float((a["separation_speed"]>0).mean()),
              near_threshold=weighted(np.abs(v-.020)<.002),refined_replays=int(f.size),stored_labels=int(y.size),
              stick_by_state=y.mean((0,1,3,4,5,6)).tolist(),flip_counts_raw=int(diff.sum())))
def ci(key):
    x=np.array([r[key] for r in per_catalog]); center=x.mean(axis=0); half=x.std(axis=0,ddof=1)*2.200985/np.sqrt(len(x))
    return dict(mean=center.tolist(),lo=(center-half).tolist(),hi=(center+half).tolist())
errors=np.concatenate(fine_abs)
out=dict(engine_version=version("pybullet"),metrics={k:ci(k) for k in (
        "primary_stick_weighted","v48_stick_weighted","new_vs_old_disagreement","refined_disagreement",
        "negative_positive_flip","separating_fraction","near_threshold","stick_by_state")},
         refinement=dict(abs_speed_error_median=float(np.median(errors)),abs_speed_error_p95=float(np.quantile(errors,.95)),
             abs_speed_error_max=float(errors.max()),distance_to_threshold_max_on_flip=float(max(diff_at_flip, default=0.)),
             raw_flips=sum(a["flip_counts_raw"] for a in per_catalog),raw_trials=sum(a["refined_replays"] for a in per_catalog)),
         min_outward_normal_speed=minimum,stored_labels=sum(a["stored_labels"] for a in per_catalog),
         unique_solver_calls=sum(a["stored_labels"] for a in per_catalog)*3//4,panels=per_catalog)
(ROOT/"solver49_audit.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps({k:v for k,v in out.items() if k!="panels"},indent=1))
