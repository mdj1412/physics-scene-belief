"""Validate every archived hash, cost measurement and independent CPU inference formula."""
from pathlib import Path
import json
import hashlib

import numpy as np
import model43

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
EXPECTED=[f'{d}_{p}' for d in ('original','robust','privileged') for p in ('zero','temperature','fixed','optimizer')]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    files=sorted(OUT.glob('panel_*/*.npz'))
    model=model43.load_model()
    assert len(files)==576
    meta=json.loads((OUT/'run_even.json').read_text())
    beta=json.loads((ROOT.parent/'height_offset_v42/results/run.json').read_text())['beta']
    fingerprint=meta['fingerprint']
    error=0.; count=0
    for path in files:
        md=json.loads(path.with_suffix('.json').read_text())
        assert md['sha256']==sha(path) and md['fingerprint']==fingerprint
        with np.load(path) as a:
            selection=a['selected']; sensors=[]
            for e in range(4):
                d={1:a['query_paid'][e],3:a['probe_paid'][e]}
                for action,sens in d.items():
                    assert np.isfinite(sens[selection[e]==action]).all()
                    assert np.isnan(sens[selection[e]!=action]).all()
                sensors.append(d)
            assert np.all((selection==3).sum(1)==13)
            pred=a['pred']
            base=pred[:,0]
            fitted=model43.robust_probability(model,base[0],a['xx'],selection,sensors,float(a['height']))
            error=max(error,float(np.max(np.abs(fitted-base[1]))))
            for di,d in enumerate(('original','robust','privileged')):
                calc=model43.temperature(model,base[di],d)
                error=max(error,float(np.max(np.abs(calc-pred[di,1]))))
                for pi,weight in ((2,.325),(3,a['weights'])):
                    # Odds rescaling written as p / (p + (1-p) exp(-delta)).
                    value=np.exp(np.clip(-beta*(a['z']*weight),-30,30))[:,None,None]
                    calc=base[di]/(base[di]+(1.-base[di])*value)
                    error=max(error,float(np.max(np.abs(calc-pred[di,pi]))))
            count+=pred.size
    assert error<5e-12,error
    times=[]; cold=[]; readout=[]
    for path in sorted((OUT/'timing').glob('*.npz')):
        md=json.loads(path.with_suffix('.json').read_text())
        assert md['sha256']==sha(path) and md['fingerprint']==fingerprint
        with np.load(path) as a:
            assert list(a['methods'])==EXPECTED
            assert np.all(np.sort(a['order'],axis=1)==np.arange(12))
            np.testing.assert_allclose(a['cold_ms'],a['setup_ms']+a['warm_ms'])
            times.append(a['warm_ms']);cold.append(a['cold_ms']);readout.append(float(a['max_error']))
    times=np.stack(times)
    assert times.shape==(12,3,12)
    # Leave-one-panel-out ranges quantify sensitivity of the percentile ratio.
    loo=[]
    for i in range(len(times)):
        b=np.quantile(np.delete(times,i,axis=0),.9,axis=(0,1))
        loo.append(b[7]/b[5])
    summary=dict(fingerprint=fingerprint,files=len(files),prediction_values=count,
                 max_policy_replay_error=error,max_latency_replay_error=max(readout),methods=EXPECTED,
                 warm_p90_ms=np.quantile(times,.9,axis=(0,1)).tolist(),
                 cold_p90_ms=np.quantile(cold,.9,axis=(0,1)).tolist(),
                 robust_optimizer_temperature_leave_one_panel_ratio_range=[min(loo),max(loo)])
    (ROOT/'audit_gpu43.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(summary)

if __name__=='__main__':
    main()
