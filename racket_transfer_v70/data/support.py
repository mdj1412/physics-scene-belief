"""Check flow of measured inlet states across distinct held-out rubber products."""
import csv
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree, ConvexHull

from audit import CONFIG, read, ROOT

def main():
    shot, rubber, features = [], [], []
    for rid, (g, _, _, _) in CONFIG.items():
        for i,row in enumerate(read("original",rid)[1]):
            shot.append((rid,i))
            rubber.append(g)
            features.append((row[8],row[10],row[3]))
    rubber=np.array(rubber); x=np.array(features)
    records=[]; stats={}
    for heldout in sorted(set(rubber)):
        source=rubber!=heldout; target=~source
        centered=(x-x[source].mean(0))/x[source].std(0)
        hull=ConvexHull(centered[source])
        inn=(centered[target] @ hull.equations[:,:-1].T + hull.equations[:,-1]).max(1)<=1.e-9
        nn=cKDTree(centered[source]).query(centered[target],k=5)[0][:,-1]
        # Reference for novelty is also cross-rubber, measured in the same scale.
        reference=[]
        for g in np.unique(rubber[source]):
            candidate=source & (rubber!=g)
            group=source & (rubber==g)
            reference.extend(cKDTree(centered[candidate]).query(centered[group],k=5)[0][:,-1])
        cutoff=float(np.quantile(reference,.95))
        for j,ok,d in zip(np.flatnonzero(target),inn,nn):
            records.append((*shot[j],heldout,bool(ok),d,cutoff,d<=cutoff))
        stats[heldout]=dict(n=int(target.sum()),outside_hull=int((~inn).sum()),
                  median_cross_rubber_knn5=float(np.median(nn)),q95_source_cross_rubber_knn5=cutoff,
                  above_source_q95=int(np.sum(nn>cutoff)))
    (ROOT/"support.json").write_text(json.dumps(dict(host=platform.node(),slurm=os.getenv("SLURM_JOB_ID"),
          columns=["vx_in m/s","vz_in m/s","omega_y_in rad/s"],rubbers=stats),indent=2)+"\n")
    with (ROOT/"support_rows.csv").open("w",newline="") as out:
        w=csv.writer(out)
        w.writerow(["configuration","original_row_0based","heldout_rubber","inside_source_hull","cross_rubber_knn5","source_cross_rubber_knn5_q95","inside_source_q95"])
        w.writerows(records)

if __name__=="__main__":
    main()
