"""Separately count reversal of pre-impact slip direction and of the v70 gate's sign."""
import json
import numpy as np
from elastic import GROUPS, HERE, read, source_fit, rawfeat, design, base, train_tape, BUDGETS
def run():
    out={}
    for name,spec in GROUPS.items():
        ep=spec['empirical']
        x,y=read(spec['held'],'train');xx,yy=read(spec['held'],'test')
        m,sd,th,ip,ix=source_fit([read(s['held'],'train') for g,s in GROUPS.items() if g!=name])
        h=design(x,m,sd); ht=h.transpose(0,2,1)@ix
        hess=ht@h
        grad=(ht@((y-base(x))-h@th)[...,None])[...,0]
        tape=train_tape(len(x),min(spec['held']))
        sc0=xx[:,0]-xx[:,2]
        sc1=yy[:,0]-yy[:,2]
        gate=np.column_stack([np.ones(len(xx)),(xx-ep['mean'])/ep['std']])@ep['logistic']>0
        truth=sc1>0
        r={'n':len(xx),'in_to_out_same':int(np.sum(sc0*sc1>0)),'in_to_out_flip':int(np.sum(sc0*sc1<0)),
           'gate_disagrees':int(np.sum(truth!=gate)),'gate_flips_incoming':int(np.sum(gate!=(sc0>0))),'elastic_by_k':{}}
        z=(rawfeat(xx)-m)/sd
        for k in BUDGETS:
            inds=tape[:,:k]
            post=th+np.linalg.solve(ip+hess[inds].sum(1),grad[inds].sum(1)[...,None])[...,0]
            alpha=np.clip(post[:,2,None]+post[:,3,None]*z[:,1],0.,.8)
            pred=((1-2.5*alpha)*sc0)>0
            r['elastic_by_k'][k]=dict(sign_accuracy=float(np.mean(pred==truth)),predicted_fraction_reversing_incoming=float(np.mean(alpha>.4)))
        out[name]=r
    (HERE/'polarity.json').write_text(json.dumps(out,indent=2))
if __name__=='__main__':run()
