"""Raw archive and timing replay audit; no contact physics or policy fitting in this program."""
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads(path.read_text())

def until(path):
    while not path.exists():time.sleep(4)

def verify(paths,run):
    error=0.
    oracle_diff=0.
    probe_diff=0.
    ref=None
    states_per_cell=0
    for path in paths:
        desc=read(path.with_suffix('.json'))
        assert desc['fingerprint']==run['fingerprint'] and desc['sha256']==digest(path)
        with np.load(path,allow_pickle=False) as data:
            pb=data['p_base']; labels=data['labels']; picks=data['choice']; risks=data['risk']
            z=data['z']; w=data['weights'];actions=data['actions']
            assert pb.shape==(3,2,8,128,8) and labels.shape==(9,2,3,8,128) and labels.dtype==bool
            assert risks.shape==(9,3,4,2,3,8) and picks.shape==(3,2,3,8,8)
            assert np.isfinite(pb).all() and pb.min()>-1e-11 and pb.max()<1+1e-11
            assert np.isin(picks,[0,1,2]).all() and np.isin(actions,[1,3]).all() and np.all((actions==3).sum(-1)==13)
            np.testing.assert_array_equal(data['offsets_mm'],run['offsets_mm'])
            np.testing.assert_array_equal(data['decoders'],run['decoders'])
            np.testing.assert_array_equal(data['methods'],run['methods'])
            for d,s,reg in np.ndindex(3,2,3):
                base=pb[d,s]
                gain=np.stack([np.zeros(8),np.full(8,.325),w[s,reg]])
                odds=np.exp(run['beta']*gain*z[s])[:,:,None,None]
                p=base[None]*odds/(1+(odds-1)*base[None])
                chosen=np.take_along_axis(p,picks[d,s,reg][None,:,None,:].astype(int),axis=0)[0]
                pred=np.concatenate([p,chosen[None]])
                # Mean(p^2) - 2 E[pY] + E[Y], averaged only over objects and held-out noise.
                target=labels[:,s,reg].astype(float)
                target_term=np.einsum('aen,len->lae',pred.mean(-1),target)/target.shape[-1]
                expected=(pred**2).mean((-1,-2))[None]-2*target_term+target.mean(-1)[:,None,:]
                err=np.max(np.abs(risks[:,d,:,s,reg]-expected))
                np.testing.assert_allclose(risks[:,d,:,s,reg],expected,atol=2.e-12,rtol=0)
                error=max(error,float(err))
                states_per_cell+=expected.size
            b=actions==3
            probe_diff=max(probe_diff,float(np.abs(pb[1:,b]-pb[0,b]).max()))
            if ref is None:
                ref=(labels[4],actions,pb[1],data['z'],data['weights'],data['first3_y'],data['fourth_y'])
            else:
                for item,old in zip((labels[4],actions,pb[1],z,w,data['first3_y'],data['fourth_y']),ref):
                    np.testing.assert_allclose(item,old,rtol=0,atol=3e-12)
                oracle_diff=max(oracle_diff,float(np.abs(pb[1]-ref[2]).max()))
    assert probe_diff<1.e-12
    return error,oracle_diff,probe_diff,states_per_cell


def main():
    until(OUT/'run.json'); run=read(OUT/'run.json')
    for name,h in run['hashes'].items(): assert digest(ROOT.parent/name)==h,name
    errors=np.zeros(3);counts=0;sha_checked=0; timing_err=0.
    for n in range(12):
        for fam,h,law in [(f,k,l) for f in ('flare','roll') for k in ('h19','h31') for l in ('saturating','crossover')]:
            files=[OUT/f'{fam}_{h}_{v}_{law}'/f'catalog_{n:02d}.npz' for v in ('base','hollow','bulge')]
            for p in files:until(p.with_suffix('.json'))
            err=verify(files,run)
            errors=np.maximum(errors,err[:3]);counts+=err[3];sha_checked+=len(files)
        path=OUT/'timing'/f'panel_{n:02d}.npz';until(path.with_suffix('.json'))
        side=read(path.with_suffix('.json'));assert digest(path)==side['sha256'] and side['fingerprint']==run['fingerprint']
        timing_err=max(timing_err,side['max_error'])
        with np.load(path) as t:
            assert t['full_ms'].shape==(4,5)
            assert np.all(t['full_ms']>0) and np.all(t['setup_ms']>=0)
            np.testing.assert_array_equal(t['methods'],run['timing_methods'])
            np.testing.assert_array_equal(np.sort(t['order'],axis=1),np.broadcast_to(np.arange(5),(4,5)))
            np.testing.assert_allclose(t['cold_ms'],t['setup_ms']+t['full_ms'],atol=1e-10)
        print('verified panel',n,flush=True)
    until(OUT/'completion.json')
    assert run['fingerprint']==read(OUT/'completion.json')['fingerprint']
    out=dict(checked_raw_archives=sha_checked,checked_timing_archives=12,environment_loss_values=counts,
             max_recomputed_brier_error=errors[0],max_oracle_contour_probability_error=errors[1],
             max_probe_decoder_probability_error=errors[2],max_online_probability_error=timing_err,
             checked_dependencies=len(run['hashes']),run_fingerprint=run['fingerprint'],job=os.environ.get('SLURM_JOB_ID'))
    (ROOT/'audit_runner42.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out),flush=True)
if __name__=='__main__':main()
