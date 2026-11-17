"""Summarise exactly one paired benchmark value per product and calibration draw."""
from pathlib import Path
import csv
import json
import numpy as np
ROOT=Path(__file__).resolve().parent

def main():
    rubbers={}; counts={}
    with (ROOT/'draw_metrics.csv').open() as f:
        for r in csv.DictReader(f):
            key=(r['rubber'],r['method'],int(r['budget']))
            rubbers.setdefault(key,[]).append([float(r['velocity']),float(r['spin'])])
    folds={}
    for (g,m,k),a in rubbers.items():
        if len(a)!=256:raise ValueError((g,m,k,len(a)))
        folds.setdefault(g,{}).setdefault(m,{})[k]=np.mean(a,axis=0).tolist()
    methods=list(next(iter(folds.values())))
    sizes={g:json.loads((ROOT.parents[1]/'racket_transfer_v70'/'solver'/'metrics.json').read_text())['folds'][g]['metrics']['inclined']['test']['n'] for g in folds}
    macro={};micro={}
    for m in methods:
        for k in next(iter(folds.values()))[m]:
            v=[q[m][k] for q in folds.values()]
            macro.setdefault(m,{})[k]=np.mean(v,axis=0).tolist()
            micro.setdefault(m,{})[k]=np.average(v,axis=0,weights=list(sizes.values())).tolist()
    (ROOT/'summary.json').write_text(json.dumps(dict(folds=folds,macro=macro,micro=micro),indent=2)+'\n')
    print(json.dumps(macro,indent=2))
if __name__=='__main__':main()
