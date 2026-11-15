"""Raw manifest, action replay and timing schedule audit."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
R=ROOT/'results'
def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    cfg=json.loads((ROOT/'design36.json').read_text()); meta=json.loads((R/'completion.json').read_text())
    hashes=meta['hashes']
    for name,sha in hashes.items(): assert digest(ROOT.parent/name)==sha
    seeds=set(); blocks=0; n=0; timing=[]
    method_names=cfg['methods']
    for c,cell in enumerate(cfg['cells']):
        for b in range(cfg['blocks_per_cell']):
            path=R/cell/f'block_{b:03d}'
            saved=json.loads((path/'meta.json').read_text())
            assert list(saved['sha256']) and len(saved['sha256'])==7
            assert saved['block']==b and saved['cell']==cell and saved['start']==(-1,1)[b%2]
            these=set(saved['pair']+saved['calibration_seeds']+saved['deployment_seeds'])
            assert len(these)==18 and not (seeds&these)
            seeds.update(these); blocks+=1
            expected=cfg[f'timed_cell_block{b}'][c] if b<2 else None
            for name,sha in saved['sha256'].items():
                f=path/name
                assert digest(f)==sha
                with np.load(f,allow_pickle=False) as d:
                    assert d['methods'].tolist()==method_names
                    assert d['risk'].shape==(6,8) and np.all(np.isfinite(d['risk']))
                    assert np.all((d['near_actions']==3).sum(1)==13)
                    assert np.all((d['far_actions']==3).sum(1)==13)
                    assert np.array_equal(d['far_y'][:,:3],d['near_y'][:,:3])
                    expected_dirs=saved['start']*np.array([1,-1,1,-1])
                    assert np.all(expected_dirs==d['dirs'])
                    ms=d['full_ms']
                    if f.stem==expected:
                        assert ms.shape==(3,6) and np.isfinite(ms).all() and np.all(ms>0)
                        for o in d['timing_order']:np.testing.assert_array_equal(np.sort(o),np.arange(6))
                        timing.append(ms)
                    else: assert ms.shape==(0,6)
                n+=1
    assert n==meta['archives'] and blocks==meta['clusters'] and len(timing)==8
    arr=np.stack(timing)
    out=dict(blocks=blocks,archives=n,source_hashes_verified=len(hashes),distinct_seeds=len(seeds),
             timed_clusters=len(timing),per_method_calls=arr.shape[0]*arr.shape[1],
             timed_methods=method_names,p90_ms=np.quantile(arr,.9,axis=(0,1)).tolist(),
             p90_by_repeat_ms=np.quantile(arr,.9,axis=0).tolist(),all_checks_passed=True)
    (ROOT/'archive_audit36.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
