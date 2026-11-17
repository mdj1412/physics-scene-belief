"""Nested source-product validation for a nonlinear reference and incoming-state support."""
import csv
import json
import os
import platform
from pathlib import Path
from itertools import product

import numpy as np
import torch

BASE = Path(__file__).resolve().parents[1]
GROUPS = {'Hexer Duro': [1,3,9], 'Power 3': [2,4], 'Grass D.TecS': [5,6],
          'Diamant': [7], 'Blowfish': [8], 'A-B-S II soft': [10]}
R=.02
CANDIDATES = list(product((.55,1.05),(.3,3.0)))


def data(split):
    x,y,ids,rows=[],[],[],[]
    for cfg in range(1,11):
        f=BASE/'data'/'upstream'/split/f'racket_{cfg:02d}.csv'
        a=np.genfromtxt(f,delimiter=',',names=True)
        def cols(names):
            return np.stack([a[k]*(R if k.startswith('spin') else 1.) for k in names],axis=1)
        x.append(cols(('v_in_x','v_in_z','spin_before_y')))
        y.append(cols(('v_out_x','v_out_z','spin_after_y')))
        ids.extend([cfg]*len(a));rows.extend(range(len(a)))
    return (torch.tensor(np.concatenate(x),device='cuda'),torch.tensor(np.concatenate(y),device='cuda'),
            np.array(ids), np.array(rows))


def design(x,centers,width):
    kernel=torch.exp(-torch.cdist(x,centers).square()/(2*width**2))
    return torch.cat([torch.ones_like(x[:,:1]), x, kernel], dim=1)


def prediction(x,y,xt,hyp):
    mean=x.mean(0); sd=x.std(0,unbiased=False).clamp(min=1e-7)
    levels=x.new_tensor([-2.25,-.75,.75,2.25])
    centers=torch.cartesian_prod(levels,levels,levels)
    train=design((x-mean)/sd,centers,hyp[0])
    test=design((xt-mean)/sd,centers,hyp[0])
    diag=x.new_full((train.shape[1],),hyp[1]);diag[:4]=1e-7
    weights=torch.linalg.solve(train.T@train+torch.diag(diag),train.T@y)
    return test@weights


def nearest(q,ref):
    return torch.cat([torch.cdist(block,ref).amin(1) for block in q.split(512)])


def main():
    torch.set_default_dtype(torch.float64)
    x,y,ids,_=data('train'); xt,yt,tid,rows=data('test')
    out=[]; meta={}
    for target,cfgs in GROUPS.items():
        tr=np.flatnonzero(~np.isin(ids,cfgs)); out_idx=np.flatnonzero(np.isin(tid,cfgs))
        same=np.flatnonzero(np.isin(ids,cfgs))
        score=np.zeros(len(CANDIDATES))
        distances=[]
        mean=x[tr].mean(0); sd=x[tr].std(0,unbiased=False)
        z=(x-mean)/sd
        for name,cc in GROUPS.items():
            if name==target: continue
            mask=np.isin(ids[tr],cc); a,b=tr[~mask],tr[mask]
            for j,h in enumerate(CANDIDATES):
                yp=prediction(x[a],y[a],x[b],h)
                score[j]+=float((yp-y[b]).square().mean())/5.
            distances.append(nearest(z[b],z[a]))
        h=CANDIDATES[int(np.argmin(score))]
        threshold=float(torch.quantile(torch.cat(distances),.95))
        d=nearest((xt[out_idx]-mean)/sd,z[tr]).cpu().numpy()
        rng=np.random.default_rng(7020+cfgs[0])
        matched=rng.choice(tr,len(same),replace=False)
        variants=[('pooled_rbf',tr),('matched_count_source_rbf',matched),('known_rubber_rbf',same)]
        meta[target]={'hyperparameters':h,'source_inner_mse':score.tolist(),'n_source':len(tr),
                      'n_target_train_diagnostic':len(same),'n_target_test':len(out_idx),'distance_threshold':threshold}
        for variant,idx in variants:
            predicted=prediction(x[idx],y[idx],xt[out_idx],h)
            ev=torch.linalg.vector_norm((predicted-yt[out_idx])[:,:2],dim=1)
            ew=abs((predicted-yt[out_idx])[:,2])/R
            result=torch.column_stack([predicted[:,:2],predicted[:,2]/R,ev,ew]).cpu().numpy()
            for row,(vx,vz,w,ve,we),distance in zip(out_idx,result,d):
                out.append(dict(rubber=target,config=int(tid[row]),row=int(rows[row]),method=variant,
                    vx_pred=vx,vz_pred=vz,wy_pred=w,error_v=ve,error_w=we,support_distance=distance,
                    within_support=int(distance<=threshold)))
    here=BASE/'analysis'
    with (here/'rbf_predictions.csv').open('w') as f:
        wr=csv.DictWriter(f,fieldnames=out[0]);wr.writeheader();wr.writerows(out)
    (here/'rbf_fit.json').write_text(json.dumps({'folds':meta,'gpu':torch.cuda.get_device_name(),'node':platform.node(),'job':os.environ['SLURM_JOB_ID']},indent=2))
if __name__=='__main__':main()
