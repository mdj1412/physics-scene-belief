"""Audit transfer of source conditional risks; leave out an entire target catalog for each score."""
import itertools
import json
import numpy as np
from score46 import ROOT,SHAPES,VARS,XGRID,ci

def table_bins(x, v, model):
    k=np.abs(x[...,None]-XGRID).argmin(-1)
    j=(v[...,None]>np.asarray(model["thresholds"])[k]).sum(-1)
    return 4*k+j

def switches(total,count):
    x=np.divide(total.sum(-1),count.sum(-1),out=np.zeros(6),where=count.sum(-1)>0)
    sub=np.divide(total,count,out=np.broadcast_to(x[:,None],(6,4)).copy(),where=count>0)
    return x<0,(.75*sub+.25*x[:,None])<0

def main():
    models=json.loads((ROOT/"router46.json").read_text())
    risk=np.zeros((2,12,2,6,4));num=np.zeros_like(risk)
    for pa,sh,vi,nH in itertools.product(range(12),range(2),range(3),(19,31)):
        sp=f"panel_{pa:02d}";stem=f"{SHAPES[sh]}_{VARS[vi]}_h{nH}.npz"
        with np.load(ROOT/"camera46"/sp/stem) as c:
            bins=[table_bins(c[x],c[v],models[md]).ravel() for md,x,v in
                  (("visual","x_hat_mm","variance_hat"),("oracle","true_x_mm","true_variance"))]
        D=0.
        for sf,law in itertools.product(("inlay","woven"),("rational","gaussian_rate")):
            pth=f"{SHAPES[sh]}_{sf}_{law}_{VARS[vi]}_h{nH}.npz"
            with np.load(ROOT/"results"/sp/pth) as p, np.load(ROOT/"labels46"/sp/pth) as l:
                y=l["labels"][1:].mean(0)
                o,g=p["pred"][:,0]
                D=D+((g*g-o*o).mean(-1)-2.*(g-o).mean(-1)*y)/4.
        for m,b in enumerate(bins):
            num[m,pa,sh]+=np.bincount(b,minlength=24).reshape(6,4)
            risk[m,pa,sh]+=np.bincount(b,weights=D.ravel(),minlength=24).reshape(6,4)
    result={}
    for m,name in enumerate(("visual","oracle")):
        tot=risk[m].sum((0,1));counts=num[m].sum((0,1))
        cv=[]
        for pa in range(12):
            rr=risk[m,pa].sum(0);nn=num[m,pa].sum(0)
            x,j=switches(tot-rr,counts-nn)
            cv.append([(rr*x[:,None]).sum()/nn.sum(),(rr*j).sum()/nn.sum(),(rr*(j.astype(float)-x[:,None])).sum()/nn.sum()])
        j=np.array(models[name]["risk_joint"])<0
        sx=np.array(models[name]["risk_x"])<0
        known=(risk[m]*(j.astype(float)-sx[:,None])).sum(axis=(1,2,3))/num[m].sum(axis=(1,2,3))
        result[name]=dict(cross_fitted_x_joint_increment=ci(cv),cv_panel=cv,
            cross_fitted_minus_frozen_increment=ci(np.array(cv)[:,2]-known),
            zero_shot_joint_minus_position=ci(known),count=counts,
            target_raw_D=np.divide(tot,counts,out=np.zeros_like(tot),where=counts>0),
            target_raw_D_by_shape=np.divide(risk[m].sum(0),num[m].sum(0),out=np.zeros_like(num[m,0]),where=num[m].sum(0)>0),
            per_cell_zero_shot_joint_minus_x=tot*(j.astype(float)-sx[:,None])/counts.sum(),
            target_clip_fraction=counts/counts.sum())
    (ROOT/"transfer46.json").write_text(json.dumps(result,default=lambda x:np.asarray(x).tolist(),allow_nan=False,indent=1)+"\n")
if __name__=="__main__":main()
