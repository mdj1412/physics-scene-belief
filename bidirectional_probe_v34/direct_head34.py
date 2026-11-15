"""Frozen stopping head, deployed from only k observed probes. No law sign is an input."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
FAMILIES=['rounded_fine','rounded_coarse','ridged_fine','ridged_coarse']
SCHEDULES=['same','alternating']
CUES=['neutral','estimated','oracle','placebo']
def load_artifact():
    return dict(head=json.loads((ROOT/'direct_models34.json').read_text()),
                direction=json.loads((ROOT/'diag_models.json').read_text()))
def cue_est(lock, idx, paid_x, paid_y, dirs, schedule):
    from diag34_model import raw_probability
    return 2*np.asarray(raw_probability(lock,FAMILIES[idx],paid_x,paid_y,dirs,schedule)).reshape(-1)-1
def design(features,start,alt,cue):
    x=np.asarray(features,dtype=float); cue=np.broadcast_to(np.asarray(cue,dtype=float),x.shape[:-1])
    start=np.broadcast_to(start,x.shape[:-1]);alt=np.broadcast_to(alt,x.shape[:-1])
    extra=np.concatenate([np.ones_like(cue)[...,None],x[...,[0,1,2,3,10,11,14]]],axis=-1)*cue[...,None]
    return np.concatenate([x,start[...,None],alt[...,None],extra],axis=-1)
def score_direct34(features,paid_x,paid_y,dirs,schedule,k,fold,artifact):
    x=design(features,dirs[0],float(schedule=='alternating'),
             cue_est(artifact['direction'],fold,paid_x[None],paid_y[None],dirs[None],schedule)[0])
    m=artifact['head']['folds'][str(fold)]['estimated'][str(k)]
    return float((x-np.array(m['mean']))/np.array(m['sd'])@np.array(m['beta'])+m['intercept'])
