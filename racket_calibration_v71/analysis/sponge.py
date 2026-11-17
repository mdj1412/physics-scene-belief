"""Keep the deployment configuration fixed, change where the calibration shots were collected."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from elastic import GROUPS, read, HERE, BUDGETS
sys.path.insert(0,str(HERE.parent/'core'))
from adapt import adapt

def main():
    rubber='grass_dtecs'
    ep=GROUPS[rubber]['empirical']
    fit=json.loads((HERE.parent/'core'/'fit.json').read_text())
    penalty=fit['folds'][rubber]['penalty']
    outputs=[]
    for target in (5,6):
        xx,yy=map(torch.from_numpy,read([target],'test'))
        for calibration in (5,6):
            x,y=map(torch.from_numpy,read([calibration],'train'))
            tape=np.stack([np.random.default_rng(np.random.SeedSequence([710081,calibration,r])).choice(len(x),32,replace=False) for r in range(256)])
            for k in BUDGETS:
                idx=tape[:,:k]
                preds,diag=adapt(ep,x[idx],y[idx],xx,penalty)
                for method in ['gate_only','gate_expert','routed_expert_only']:
                    error=preds[method]-yy
                    vel=error[...,:2].norm(dim=-1).mean(-1).numpy()
                    spin=(error[...,2].abs().mean(-1)/.02).numpy()
                    outputs.append(dict(cal_cfg=calibration,test_cfg=target,budget=k,method=method,n_test=len(xx),n_cal_pool=len(x),n_draws=256,
                                        flip_rate=float(diag['flip'].double().mean()),velocity=float(np.mean(vel)),spin=float(np.mean(spin))))
    with (HERE/'sponge.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=outputs[0]);w.writeheader();w.writerows(outputs)
if __name__=='__main__':main()
