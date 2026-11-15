"""Check all independently regenerated source, continuation, bank and target seed families."""
from pathlib import Path
import json
import numpy as np

ROOT=Path(__file__).resolve().parent

def add(catalog,unique_codes,label):
    duplicates=set(unique_codes)&catalog
    if len(set(unique_codes))!=len(unique_codes) or duplicates:
        raise ValueError(f'{label}: duplicate seed codes {duplicates}')
    catalog.update(unique_codes)


def main():
    dirs=[ROOT/x for x in json.loads((ROOT/'retrain_roots.json').read_text())['model_dirs']]
    occupied=set();stage_counts=[]
    for i,root in enumerate(dirs):
        here=0
        for arm in ('broad','repeated'):
            futures=1 if arm=='broad' else 2
            for p in sorted((root/arm).glob('block_*/meta.json')):
                md=json.loads(p.read_text())
                base=md['base_seed']
                code=list(md['bank_pair'])+[base+8903]
                with np.load(p.parent/'data.npz') as z:
                    spec=[]
                    for e,prefix in enumerate(z['prefix_seed']):
                        offset=base+1000*e
                        assert prefix==offset+1
                        ids=list(z['deployment_seeds'][e])+[int(prefix)]
                        spec=ids+[offset+100+20*s+j for s in range(2) for j in range(futures)]
                        code.extend(spec)
                add(occupied,code,f'{i}-{arm}-{base}')
                here+=len(code)
        stage_counts.append(here)
    target=[]
    for p in sorted((ROOT/'results_eval').glob('*/block_*/meta.json')):
        md=json.loads(p.read_text())
        base=md['calibration_seeds'][0]-100
        target+=md['bank_pair']+md['calibration_seeds']+md['deployment_seeds']+[base+401,base+501]
    assert len(target)==48*20
    add(occupied,target,'target')
    result=dict(training_nonshared_seeds_by_pair=stage_counts,target_nonshared_seed_count=len(target),
                total_unique_nonshared_seed_codes=len(occupied),n_target_blocks=48,
                shared_numerics='Feature detector and Sobol stratification constants held fixed intentionally.')
    (ROOT/'results_eval/seed_audit32.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
