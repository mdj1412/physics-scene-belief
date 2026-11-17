"""Rebuild every archived operational robust output and every correction using paid inputs."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import expit, logit
import model43

ROOT = Path(__file__).resolve().parent


def run():
    model = model43.load_model()
    fit = json.loads((ROOT/'source_validation43.json').read_text())
    files = sorted((ROOT/'results').glob('panel_*/*.npz'))
    assert len(files) == 576
    diffs = np.zeros(4)
    anti_leak = 0.
    paid_count = np.zeros(2,dtype=np.int64)
    for file in files:
        assert hashlib.sha256(file.read_bytes()).hexdigest() == json.loads(file.with_suffix('.json').read_text())['sha256']
        with np.load(file) as a:
            idx = a['selected']
            px,pr = a['query_paid'],a['probe_paid']
            for mat,act in ((px,1),(pr,3)):
                assert np.isfinite(mat[idx==act]).all()
                assert np.isnan(mat[idx!=act]).all()
            np.testing.assert_array_equal((idx==3).sum(axis=1),np.full(4,13))
            paid_count += [(idx==1).sum(),(idx==3).sum()]
            preds=a['pred']; low=preds[:,0]
            obs = [{1:x,3:y} for x,y in zip(px,pr)]
            base = model43.robust_probability(model,low[0],a['xx'],idx,obs,float(a['height']))
            diffs[0] = max(diffs[0],abs(base-low[1]).max())
            # Replace all unpurchased sensor values; family metadata and labels are never passed.
            alt = [{1:np.nan_to_num(xx,nan=1e4),3:np.nan_to_num(yy,nan=-1e4)} for xx,yy in zip(px,pr)]
            other = model43.robust_probability(model,low[0],a['xx'],idx,alt,float(a['height']))
            anti_leak = max(anti_leak, abs(other-base).max())
            for d,name in enumerate(('original','robust','privileged')):
                p=low[d]; z=a['z'][:,None,None]
                pred_t=expit(logit(np.clip(p,1e-5,1.-1e-5))/model['temperature'][name])
                diffs[1]=max(diffs[1],abs(pred_t-preds[d,1]).max())
                for c,w in ((2,.325),(3,a['weights'][:,None,None])):
                    factor=np.exp(fit['beta']*z*w)
                    b=p*factor/(1-p+p*factor)
                    diffs[c]=max(diffs[c],abs(b-preds[d,c]).max())
    assert max(diffs)<2e-9 and anti_leak==0., (diffs,anti_leak)
    out=dict(n_files=len(files),sha256_verified=len(files),max_abs_replay_error=diffs.tolist(),
             unselected_sensor_mutation_max_error=anti_leak,
             repeated_cell_query_probe_counts=paid_count.tolist())
    (ROOT/'audit43.json').write_text(json.dumps(out,indent=2)+'\n')


if __name__=='__main__':
    run()
