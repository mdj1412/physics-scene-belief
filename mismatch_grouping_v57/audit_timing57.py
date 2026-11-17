"""Count orders and audit the rare source-risk path with actual paid sign fixtures."""
from pathlib import Path
import json,os,sys
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'fresh_surface_v55'))
from bench55 import serial_alpha,terminal
from bench57 import fixture,independent_outer,outer,route,N,crossover
from policy54 import select54
from policy57 import choose

def main():
    assert os.environ.get('SLURM_JOB_ID')
    clocks=[]
    positions=np.zeros((5,5),int);previous=np.zeros((5,5),int)
    worst_error=0.;verify_n=0
    for c in range(N):
        raw,pilot,scen=fixture(c)
        with np.load(ROOT/'timing57'/f'panel_{c:02d}.npz') as t:
            assert np.array_equal(scen,t['scenario'])
            orders=t['orders']; assert np.array_equal(orders,crossover(c))
            for seq in orders:
                positions[seq,np.arange(5)]+=1
                np.add.at(previous,(seq[:-1],seq[1:]),1)
            clocks.append(t['ms'])
        # All rare source-risk switches should be processed like an actual I acquisition.
        # Select one paid pilot that triggers the rare action on each material, without labels.
        g=select54(raw[2][:,:3],raw[3][:,:3],mode='robust')
        endpoint=serial_alpha(raw)
        with np.load(ROOT/'pilots'/f'panel_{c:02d}.npz') as f:
            r,imp=f['pilot_ref'],f['pilot_imp']
        for regime in (0,1):
            sr=r[:,regime];si=imp[:,regime]
            positive=choose(sr[:,1],si[:,1],rule='risk')
            assert np.all(positive.any(axis=1))
            idx=positive.argmax(axis=1)
            pr=np.moveaxis(sr,2,1)[np.arange(2),idx]
            pi=np.moveaxis(si,2,1)[np.arange(2),idx]
            z=(pr,pi,pilot[2])
            np.testing.assert_array_equal(outer(2,pr,pi),independent_outer(pr,pi,2))
            mask=route(2,raw,z)
            assert np.all(mask) and np.all(outer(2,pr,pi))
            err=abs(terminal(raw,mask)-endpoint[1]).max()
            worst_error=max(worst_error,err)
            verify_n+=len(pr)
    assert (positions==N).all()
    assert (previous+np.eye(5,dtype=int)*N==N).all()
    arr=np.array(clocks)
    ids=np.random.default_rng(570141).integers(N,size=(16000,N))
    p90=np.quantile(arr,.9,axis=(0,1));boot=np.quantile(arr[ids],.9,axis=(1,2))
    ci=np.quantile(boot[:,[2]]-boot[:,[3,4]],[.0125,.9875],axis=0).T
    out=dict(physical_position=positions.tolist(),preceding=previous.tolist(),
             rare_trigger_verified_surface_cases=verify_n,rare_terminal_error=worst_error,
             exploratory_risk_minus_gate_fixed_ms=(p90[2]-p90[[3,4]]).tolist(),simultaneous95=ci.tolist())
    (ROOT/'timing_audit57.json').write_text(json.dumps(out,indent=2)+'\n')
    print(out)

if __name__=='__main__':main()
