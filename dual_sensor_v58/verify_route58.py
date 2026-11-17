"""Exact finite-pilot policy operating curves, independent Monte Carlo check."""
from pathlib import Path
import json,os
import numpy as np
from route58 import TABLE,route,log_odds

def verify():
    assert os.environ.get('SLURM_JOB_ID')
    ordered=np.indices((25,25,25,25),dtype=np.uint8).reshape(4,-1).T
    methods=['dual4_map','dual8_map','dual4_risk','single4_naive','gate','fixed','single4_adjusted']
    action=np.stack([route(ordered,m) for m in methods])
    assert np.array_equal(action[3],action[6])
    assert np.all(np.isfinite(log_odds(ordered)))
    parent=np.arange(25).reshape(5,5).T.reshape(-1)
    np.testing.assert_allclose(log_odds(ordered),log_odds(parent[ordered]),rtol=0,atol=1e-12)
    for r,h in np.ndindex(2,2):
        for z in (0,1):
            group=np.exp(TABLE[z,r,h])/4
            np.testing.assert_allclose(group.sum(),1)
            marg=group.reshape(5,5).sum(0)
            np.testing.assert_allclose(marg,group.reshape(5,5).sum(1))
            other=np.exp(TABLE[1-z,r,h]).reshape(5,5).sum(0)/4
            np.testing.assert_allclose(marg,other)
    probs=np.zeros((2,2,2,len(methods)))
    for z,r,h in np.ndindex(2,2,2):
        mass=np.exp(TABLE[z,r,h][ordered].sum(1))/256
        np.testing.assert_allclose(mass.sum(),1)
        probs[z,r,h]=action@mass
    acc=(probs[0]+1-probs[1])/2
    out={'methods':methods,'route_z_r_h':probs.tolist(),'balanced_accuracy_r_h':acc.tolist(),
         'balanced_accuracy_all':acc.mean((0,1)).tolist(),'enumerated_ordered_outcomes':len(ordered)}
    (Path(__file__).parent/'exact_route58.json').write_text(json.dumps(out,indent=2)+'\n')
    print('exact accuracies',acc.mean((0,1)),flush=True)
if __name__=='__main__':verify()
