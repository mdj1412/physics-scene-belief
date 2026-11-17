"""Exact attribution on stored balanced branches; label-independent purchase paths."""
import json, os
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
METRICS=['clean_and_joint','clean_only_removed_by_joint','new_delay_only_sufficient',
         'new_marker_only_sufficient','new_either_sufficient','new_cooccurrence_required',
         'clean_and_joint_absent_in_both_single_arms','marker_read_coverage_only',
         'marker_read_polarity_only','marker_read_mixed','delay_read_coverage_only',
         'delay_read_polarity_only','delay_read_mixed']

def calc(c):
    with np.load(ROOT/'blocks'/f'panel_{c:02}.npz') as p:
        routes=p['routes']; reads=p['reads']; wt=p['gate_reference'][None,None,:,None]
    result=np.zeros((2,2,8,len(METRICS)))
    for width,first,other,corr in np.ndindex(2,2,2,2):
        d1=2*width+first;d2=2*width+(other^first)
        left=routes[d1,1]
        clean,marker,delay,joint=[left^routes[d2,i] for i in (2,3,4+2*corr,5+2*corr)]
        add=(~clean)&joint
        paths=[clean&joint,clean&~joint,add&delay&~marker,add&marker&~delay,
               add&marker&delay,add&~marker&~delay,clean&joint&~marker&~delay]
        np.testing.assert_array_equal(joint, np.logical_or.reduce([paths[j] for j in [0,2,3,4,5]]))
        for idx,flip in [(3,marker),(4+2*corr,delay)]:
            l,r=reads[d2,2],reads[d2,idx]
            old=(l!=0).all(-1); new=(r!=0).all(-1)
            coverage=(old!=new).any((-2,-1)); changed=(old&new&((l[...,0]!=l[...,1])!=(r[...,0]!=r[...,1]))).any((-2,-1))
            newly=(~clean)&flip
            assert np.all(~newly|coverage|changed)
            paths.extend([newly&coverage&~changed,newly&changed&~coverage,newly&coverage&changed])
        block=np.stack([p*wt for p in paths],-1).reshape(2,2,2,32,8,-1).mean((0,1,2,3))
        result[other,corr]+=block/4
    return result

if __name__=='__main__':
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    z=np.stack([calc(i) for i in range(24)])
    (ROOT/'paths67.json').write_text(json.dumps(dict(metrics=METRICS,
        fraction_of_all_bank_prefixes=z.mean((0,3)).tolist(),
        axis_order=['sham/two_light','raw/corrected','metric']),indent=2)+'\n')
