import hashlib
import json
import re
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent


def main():
    high=ROOT/'precision3840'
    uuid=[]
    for suite in ['main','refined']:
        for shard in range(4):
            m=json.loads((high/f'results/{suite}/runtime_{shard}.json').read_text())
            for name,digest in m['code_hashes'].items():
                assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
            if suite=='main':uuid.append(re.search(r'uuid=([^,]+)',m['gpu']).group(1))
            assert m['peak_allocated_bytes']>0
    assert len(set(uuid))==4
    for shard in range(4):
        assert json.loads((high/f'data/table_{shard}.json').read_text())['dt']==1/3840
    for f in range(128):
        old=np.load(ROOT/f'data/family_{f:03d}.npz');new=np.load(high/f'data/family_{f:03d}.npz')
        for name in ['geometry','restitution']:assert np.array_equal(old[name],new[name])
        a=json.loads((ROOT/f'data/family_{f:03d}.json').read_text())
        b=json.loads((high/f'data/family_{f:03d}.json').read_text())
        assert a==b
    old=json.loads((ROOT/'results/analysis/summary.json').read_text())
    new=json.loads((high/'results/analysis/summary.json').read_text())
    index={(r['method'],r['budget'],r['region']):r for r in old}
    changes=[]
    for r in new:
        if r['region']=='all' and r['budget'] in [1,2]:
            previous=index[r['method'],r['budget'],r['region']]
            changes.append(dict(method=r['method'],budget=r['budget'],mse_960=previous['mse'],
                mse_3840=r['mse'],difference=r['mse']-previous['mse']))
    output=dict(source_hashes_match=True,four_distinct_gpus=uuid,matched_latent_worlds_and_queries=True,
                primary_dt=1/3840,physics_resolution_changes=changes)
    (high/'results/analysis/final_verification.json').write_text(json.dumps(output,indent=2))
    records={}
    paths=list(ROOT.glob('*.py'))+[ROOT/'requirements.lock.txt',ROOT.parent/'physics.py']
    for folder in ['precision3840/data','precision3840/results','results/timestep_audit']:
        paths.extend(p for p in (ROOT/folder).rglob('*') if p.is_file())
    for p in paths:
        records[str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else '../physics.py']=dict(
            sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
    (ROOT/'artifact_manifest.json').write_text(json.dumps(records,indent=2))
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
