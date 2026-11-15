"""Value of paid contact conditional on seeing the same fresh object's factual measurement."""
import json
from pathlib import Path

import numpy as np
from stats31 import LAWS, ranking_and_repeat_noise, variance_components

ROOT=Path(__file__).resolve().parent
RESULTS=ROOT/'results'


def main():
    entries=[]
    for law in LAWS:
        path=RESULTS/law/'block_000'
        with np.load(path/'metrics.npz') as data:
            total=data['gain']
            fact=data['gain_factual_only']
            probe=data['gain_probe_only']
            np.testing.assert_allclose(total,fact+probe,atol=1e-12,rtol=0)
            fields=[]
            for gains in [total,probe,fact]:
                fields.append(dict(mean=gains.mean((1,2,3)).tolist(),mean_abs=np.abs(gains).mean((1,2,3)).tolist(),
                                   components=variance_components(gains).mean(1).tolist(),
                                   ranks=[ranking_and_repeat_noise(g,'both_new_1x1')[0] for g in gains]))
            entries.append(fields)
    result={}
    for var in ['mean','mean_abs','components','ranks']:
        bylaw=np.array([[field[var] for field in lawrow] for lawrow in entries])
        result[var]=dict(law=bylaw.tolist(),pooled=bylaw.mean(0).tolist())
    result.update(laws=LAWS,measure=['total','probe_given_factual','factual_only'],
                  design='All futures/batches both stages, only prespecified first block of each law. Descriptive conditional diagnostic without interval for law groups.')
    (RESULTS/'control31.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print('Average absolute changes by stage:',result['mean_abs']['pooled'])
if __name__=='__main__':
    main()
