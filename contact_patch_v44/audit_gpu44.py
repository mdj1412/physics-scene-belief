"""Independently check sealed predicted probabilities, paid inputs and crossover measurements."""
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from scipy.special import expit, logit

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
SHAPES=('fold','ripple')
SURFACES=('inlay','woven')
LAWS=('rational','gaussian_rate')
VARIANTS=('base','hollow','bulge')
DEC=('original','privileged')
POLS=('zero','temperature')
T=json.loads((ROOT.parent/'blind_decoder_v43'/'frozen_model43.json').read_text())['temperature']


def checksum(file):
    actual=hashlib.sha256(file.read_bytes()).hexdigest()
    if actual != json.loads(file.with_suffix('.json').read_text())['sha256']:
        raise ValueError(file)


def main():
    shape=(2,2,4,128,4)
    keys=set()
    counts=np.zeros(2,dtype=int)
    max_err=np.zeros(3)
    unique=0
    target_leaks={'labels','y_target','stick','risk','offset_labels'}
    for n in range(12):
        with np.load(OUT/'worlds'/f'panel_{n:02d}.npz') as world:
            theta=world['theta']
            idx=world['object_index']
            assert theta.shape==(4,128,6) and world['support_panel']==400+n
            unique+=sum(len(np.unique(q)) for q in idx)
        for s,su,l,h in itertools.product(SHAPES,SURFACES,LAWS,(19,31)):
            for v in VARIANTS:
                f=OUT/f'panel_{n:02d}'/f'{s}_{su}_{l}_{v}_h{h}.npz'
                checksum(f); keys.add(f)
                with np.load(f) as a:
                    assert not (set(a.files)&target_leaks)
                    pred=a['pred']
                    act=a['selected']
                    np.testing.assert_array_equal(theta,a['theta'])
                    assert pred.shape==shape and np.isfinite(pred).all()
                    np.testing.assert_array_equal((act==3).sum(1),np.full(4,13))
                    for z,kind in [(a['query_paid'],1),(a['probe_paid'],3)]:
                        assert np.isfinite(z[act==kind]).all() and np.isnan(z[act!=kind]).all()
                    counts += [(act==1).sum(),(act==3).sum()]
                    for i,d in enumerate(DEC):
                        exp=expit(logit(np.clip(pred[i,0],1e-5,1-1e-5))/T[d])
                        max_err[i]=max(max_err[i],abs(exp-pred[i,1]).max())
                    max_err[2]=max(max_err[2],abs(pred[0,0][act==3]-pred[1,0][act==3]).max())
                    fixed=(a['xx'],a['calibration_x'],a['calibration_y'],act,a['probe_paid'])
                    if v=='base':
                        prev=fixed
                    else:
                        for x,y in zip(fixed,prev):
                            np.testing.assert_allclose(x,y,equal_nan=True,rtol=0,atol=0)
    assert len(keys)==576 and np.max(max_err)<2e-9
    vals=[]; cold=[]; terr=[]
    methods=[d+'_'+c for d,c in itertools.product(DEC,POLS)]
    for n in range(12):
        p=OUT/'timing'/f'panel_{n:02d}.npz'
        checksum(p)
        with np.load(p) as a:
            np.testing.assert_array_equal(a['methods'],methods)
            np.testing.assert_array_equal(np.sort(a['order'],axis=1),np.tile(np.arange(4),(3,1)))
            assert a['warm_ms'].shape==(3,4) and (a['cold_ms']>=a['warm_ms']).all()
            vals.append(a['warm_ms']);cold.append(a['cold_ms']);terr.append(a['max_error'].item())
    w,c=np.stack(vals),np.stack(cold)
    rng=np.random.default_rng(440014)
    sampled=rng.integers(0,12,(16000,12))
    def boot(x):
        val=np.quantile(x,.90,axis=(0,1))
        reps=np.quantile(x[sampled],.90,axis=(1,2))
        ci=np.quantile(reps,[.025,.975],axis=0).T
        return val,reps,ci
    q,b,ci=boot(w); qc,_,cci=boot(c)
    ratio=b[:,2]/b[:,0]
    out=dict(methods=methods, prediction_archives=len(keys),timing_archives=12,
            nonduplicated_prototype_count=unique,total_prototype_draws=12*4*128,
            first_eight_unique_prototype_count=sum(sum(len(np.unique(v[:8])) for v in np.load(z)['object_index']) for z in (OUT/'worlds').glob('panel_*.npz')),
            probabilities_including_noise_and_all_methods=int(np.prod(shape))*len(keys),
            query_probe_count_repeated_archive=counts.tolist(),temperature_and_probe_branch_replay_max=max_err.tolist(),
            warm_p90_ms=q.tolist(),warm_p90_ci=ci.tolist(),cold_p90_ms=qc.tolist(),cold_p90_ci=cci.tolist(),
            privileged_zero_over_original_zero=[q[2]/q[0],*np.quantile(ratio,[.025,.975])],max_timing_delta=max(terr))
    (ROOT/'audit_gpu44.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out),flush=True)


if __name__=='__main__':main()
