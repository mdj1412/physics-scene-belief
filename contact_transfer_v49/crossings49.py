"""Exploratory transfer thresholds conditional on the specified mixture family."""
import os
import json
import numpy as np
from score49 import ROOT

assert os.getenv('SLURM_JOB_ID')
v=np.load(ROOT/'per_catalog49.npz')['loss']
delta=v[:,:,:,2,None]-v[:,:,:,[3,4,1]]
end=delta[:,:,[0,-1],:]
bootstrap=np.random.default_rng(490014).integers(0,len(v),(20000,len(v)))
def zeros(array):
    return array[...,0,:]/(array[...,0,:]-array[...,1,:])
roots=zeros(end.mean(0))
repeats=zeros(end[bootstrap].mean(axis=1))
both=np.maximum(repeats[:,:,0],repeats[:,:,1])
out=dict(methods=['fixed direct','local gate','no-extra direct'],root=roots,pointwise95=np.quantile(repeats,[.025,.975],axis=0),
         joint_win_point=np.maximum(roots[:,0],roots[:,1]),joint_win_crossing_pointwise95=np.quantile(both,[.025,.975],axis=0),
         note='Percentile bootstrap on 12 paired catalogs; interpolation of the imposed probability law, no empirical material cutpoint.')
(ROOT/'crossings49.json').write_text(json.dumps(out,default=lambda x:np.asarray(x).tolist(),indent=1))
