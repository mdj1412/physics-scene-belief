"""Sensitivity to the deployment frequency of nominal contacts, using matched panel risk."""
import json
import numpy as np
from stats42 import ROOT,N,NZ,Z

def describe(nominal,shifted,indices):
    a,b=nominal.mean(),shifted.mean()
    item={'nominal':float(a),'shifted':float(b),'difference':float(b-a),'break_even':None}
    if a*b<0:
        item['break_even']=float(-a/(b-a))
        aa=nominal[indices].mean(1); bb=shifted[indices].mean(1)
        ok=aa*bb<0
        item['resamples_with_internal_crossing']=float(ok.mean())
        if ok.all(): item['bootstrap_ci']=np.quantile(-aa/(bb-aa),[.025,.975]).tolist()
    return item

def main():
    src=json.loads((ROOT/'stats42.json').read_text()); p=np.array(src['per_panel_loss'])
    panel=np.array(src['per_panel_group_geometry'])
    g=p[:,:,2,0]-p[:,:,0,0]
    sel=p[:,:,2,3]-p[:,:,2,0]
    indices=np.random.default_rng(421921).integers(N,size=(16000,N))
    out={'projection':describe(g[:,Z],g[:,NZ].mean(1),indices),
         'selection':describe(sel[:,Z],sel[:,NZ].mean(1),indices),'group':{}}
    for i,f in enumerate(('flare','roll')):
        for j,h in enumerate(('h19','h31')):
            x=panel[:,i,j,:]
            out['group'][f+'_'+h]=describe(x[:,Z],x[:,NZ].mean(1),indices)
    (ROOT/'deployment_mix42.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
