import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
records=[]
weight=np.array([.5,.25,.25])
def mean_state(x):
    return float(weight @ x.mean((0,1,3,4,5,6)))
for k in range(12):
    tag=f"panel_{k:02d}.npz"
    with np.load(ROOT/"labels49"/tag) as base, np.load(ROOT/"labels49_fine"/tag) as small, np.load(ROOT/"labels49_fine_all"/tag) as fine, np.load(ROOT/"labels49_ultra"/tag) as ultra:
        assert np.array_equal(small['object_slots'],ultra['object_slots'])
        slots=small['object_slots']; b=base["labels"]; f=fine["labels"]
        np.testing.assert_array_equal(small["labels"],f[...,slots])
        np.testing.assert_array_equal(small["slip_speed"],fine["slip_speed"][...,slots])
        record=dict(refined_stick=mean_state(f),all_refinement_disagreement=mean_state(b!=f),
            subset_fine_vs_ultra=mean_state(small['labels']!=ultra['labels']),
            subset_coarse_vs_ultra=mean_state(b[...,slots]!=ultra['labels']),
            refined_neg_pos_disagreement=float((f[:,:,1]!=f[:,:,2]).mean()))
        records.append(record)
out={"identical_saved_replay_vs_subset":True}
for key in records[0]:
    a=np.array([r[key] for r in records]); m=float(a.mean()); e=2.200985*a.std(ddof=1)/np.sqrt(12)
    out[key]=[m,m-e,m+e]
out["catalogs"]=records
(ROOT/"fullref_audit49.json").write_text(json.dumps(out,indent=2)+"\n")
print({k:v for k,v in out.items() if k!="catalogs"})
