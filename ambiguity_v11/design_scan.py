"""Pilot-only setting selection; never reads any test truth or outcomes."""
import os,json,time
from pathlib import Path
import numpy as np
import torch
import model
from model import Bank,draw_truth
assert os.environ.get('SLURM_JOB_ID');torch.set_num_threads(2)
records=[]
for bound in [.10,.18,.28]:
    model.AB=bound;b=Bank(25);rs=[]
    for i in range(32):
        public,_=draw_truth(51001+i*137)
        w=b.posterior(public);outer=b.outer(w,128,1000+i);jr,pr,_=b.risks(w,outer);rs.append(jr)
    rs=np.array(rs);record=dict(bound=bound,query_vin=model.QVIN,actions=np.bincount(rs.argmin(1),minlength=4).tolist(),risks=rs.tolist())
    records.append(record);Path('design_scan.json').write_text(json.dumps(records,indent=2));print(record,flush=True)
    del b;torch.cuda.empty_cache()
