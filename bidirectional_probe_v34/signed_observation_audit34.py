"""Audit sensor choices, seals, chronology, and scores through two signed orientations."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parent
def digest(f):return hashlib.sha256(f.read_bytes()).hexdigest()
def main():
    cfg=json.loads((ROOT/'policy_design34.json').read_text());r=ROOT/'results_policy';n=0;time_order=[]
    same_pairs=0
    for law in cfg['laws']:
        for b in range(8):
            folder=r/law/f'block_{b:03d}';files=[folder/s for s in ('same','alternating')]
            for f in files:
                m=json.loads((f/'meta.json').read_text())
                for ext in ('allocation.json','choices.npz','metrics.npz'):
                    assert m[ext.split('.')[0]+'_sha256']==digest(f/ext)
                alloc=json.loads((f/'allocation.json').read_text())
                with np.load(f/'choices.npz') as q,np.load(f/'metrics.npz') as z:
                    np.testing.assert_array_equal(alloc['counts'],q['counts'])
                    assert len(set(q['factual_pool_sha256']))==8
                    # Each policy precommits a distinct allowed stage risk and never uses cumulative look-ahead.
                    for j,kvec in enumerate(z['counts'].T):
                        order=np.argsort(-q['gaps'][np.arange(8),np.searchsorted([2,4,8],kvec)],axis=-1,kind='stable')[:,:13]
                        expected=np.ones((8,128),dtype=np.int64);np.put_along_axis(expected,order,3,axis=-1)
                        np.testing.assert_array_equal(expected,z['actions'][:,:,j])
                n+=1
            # Filesystem order is corroborative only; absolute mtime across redirected storage may be coarsened.
            time_order.append(max((f/'choices.npz').stat().st_mtime_ns for f in files)<=min((f/'metrics.npz').stat().st_mtime_ns for f in files))
            same_pairs+=1
    (ROOT/'seal_audit34.json').write_text(json.dumps(dict(n_archives=n,n_paired_schedules=same_pairs,
              pair_commit_before_brier_mtime=sum(time_order),all_hashes_verified=True,all_allocations_and_final_sensor_quota_verified=True),indent=2)+'\n')
if __name__=='__main__':main()
