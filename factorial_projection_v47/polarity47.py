"""Exploratory outcome replay with reversed deformation polarity; no retraining."""
import json
import numpy as np
from score47 import ROOT, interval


def analyze(panel):
    tag=f"panel_{panel:02d}.npz"
    with np.load(ROOT/"forecast"/tag) as a, np.load(ROOT/"labels"/tag) as b:
        o,p=a["p_original"],a["p_projected"]
        y=b["labels"].astype(float)
        means=[]
        for labels in (y, y[:,:,::-1]):
            # Mechanical polarity exchange, conditional on the same object and all other controls.
            # Symmetric random polarity has exact risk equal to the mean of these two risks.
            dy=labels[:,:,:,1]-labels[:,:,:,0]
            cross=(p-o)[:,:,:,:,None,None,:,:,:]*dy[:,None,...,None]
            cross=-2*cross.mean(axis=(0,2,3,4,5,6,7,8))
            oo,pp=o[:,:,:,:,None,None,None,:,:,:],p[:,:,:,:,None,None,None,:,:,:]
            yy=labels[:,None,...,None]
            d=((pp-yy)**2-(oo-yy)**2).mean(axis=(0,2,3,5,6,7,8,9))
            np.testing.assert_allclose(d[:,1]-d[:,0],cross,atol=3e-13)
            means.append(d)
    return means


if __name__=="__main__":
    x=np.asarray([analyze(k) for k in range(12)])
    # per alignment/coupling cell, sign interaction with load-bearing deformation
    aligned,reverse=x[:,0],x[:,1]
    pooled=(aligned+reverse)/2
    def interaction(z):
        return z[:,1,1]-z[:,0,1]-(z[:,1,0]-z[:,0,0])
    d=dict(recorded=aligned.mean(0),reversed=reverse.mean(0),random_sign=pooled.mean(0),
           recorded_interaction=interval(interaction(aligned)),reversed_interaction=interval(interaction(reverse)),
           random_sign_interaction=interval(interaction(pooled)),mode="post-primary diagnostic, 12 paired catalogs")
    (ROOT/"polarity47.json").write_text(json.dumps(d,default=lambda z:np.asarray(z).tolist(),indent=1)+"\n")
    print(d)
