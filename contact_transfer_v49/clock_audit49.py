import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
records=[]
weights=np.array([.5,.25,.25])
def avg(x):
    return float(weights@x.mean((0,1,3,4,5,6)))
for i in range(12):
    name=f"panel_{i:02d}.npz"
    row={}
    with np.load(ROOT/"labels49_ultra_all"/name) as all3, np.load(ROOT/"labels49_ultra"/name) as small, np.load(ROOT/"labels49_ultra_clock"/name) as clock, np.load(ROOT/"labels49_fine_all"/name) as all2, np.load(ROOT/"labels49"/name) as coarse:
        slots=small["object_slots"]
        np.testing.assert_array_equal(all3["slip_speed"][...,slots],small["slip_speed"])
        np.testing.assert_array_equal(slots,clock["object_slots"])
        row["fine_ultra_all_label_change"]=avg(all3['labels']!=all2['labels'])
        row["coarse_ultra_all_label_change"]=avg(all3['labels']!=coarse['labels'])
        row["ultra_all_stick"]=avg(all3["labels"])
        row["ultra_clock_only_subset_change"]=avg(small['labels']!=clock['labels'])
        row["ultra_clock_only_subset_max_speed_change"]=float(abs(small['slip_speed']-clock['slip_speed']).max())
        row["ultra_clock_vs_coarse_subset_change"]=avg(coarse['labels'][...,slots]!=clock['labels'])
        records.append(row)
res={}
for k in records[0]:
    v=np.array([r[k] for r in records]); m=v.mean(); e=2.200985*v.std(ddof=1)/np.sqrt(12)
    res[k]=[float(m),float(m-e),float(m+e)]
print(res)
(ROOT/"clock_audit49.json").write_text(json.dumps(dict(metrics=res,panels=records),indent=2)+"\n")
