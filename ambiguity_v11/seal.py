import os,json,hashlib
from pathlib import Path
import numpy as np

assert os.environ.get('SLURM_JOB_ID')
R=Path('results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
original=json.loads((R/'source_main.json').read_text())
for name,digest in original.items():assert sha(name)==digest,('source changed',name)
counts={};means=[]
for split,n in [('test',256),('misspec',128)]:
    frozen=json.loads((R/split/'freeze.json').read_text());extra=json.loads((R/split/'extra_freeze.json').read_text())
    assert len(frozen['policies'])==len(frozen['public'])==len(extra['files'])==n
    assert sha(R/'rules.npz')==frozen['rules_sha256']
    assert sha('extra.py')==extra['code_sha256']
    for name,digest in frozen['policies'].items():
        pp=R/split/'policies'/name;public=R/split/'public'/name;ep=R/split/'extra_policies'/name
        assert sha(pp)==digest and sha(public)==frozen['public'][name]
        assert sha(ep)==extra['files'][name]
        policy=json.loads(pp.read_text());ex=json.loads(ep.read_text())
        assert policy['public_sha256']==ex['public_sha256']==sha(public)
        data=np.load(R/split/'evaluation_61'/name.replace('.json','.npz'))
        truth=np.load(R/split/'private'/name.replace('.json','.npz'))
        for decoder in ['joint','plugin','mixture','factorized']:
            pred=data[decoder+'_prediction'];loss=data[decoder+'_loss']
            assert pred.shape==(4,32,2) and loss.shape==(4,32) and np.isfinite(pred).all() and np.isfinite(loss).all()
            expected=.5*((pred-truth['target'])**2).sum(-1)
            assert np.max(abs(expected-loss))<1e-14
        if split=='test':means.append(data['joint_loss'][policy['actions']['joint']].mean())
    counts[split]=n
summary=json.loads((R/'analysis/summary.json').read_text());recomputed=float(np.mean(means))
assert abs(recomputed-summary['test']['means']['joint'])<1e-14
for n in [81,101]:
    assert len(list((R/'numerical'/str(n)).glob('*.npz')))==32
    for p in (R/'numerical'/str(n)).glob('*.npz'):
        for x in np.load(p).values():assert np.isfinite(x).all()
assert len(json.loads((R/'ranking_audit.json').read_text()))==32
assert (R/'pipeline_done.json').exists()
validation=dict(frozen_worlds=counts,source_files_unchanged=len(original),native_losses_recomputed=True,joint_mean_recomputed=recomputed,numerical_worlds_per_order=32)
(R/'integrity.json').write_text(json.dumps(validation,indent=2))
files=[]
for p in sorted(Path('.').rglob('*')):
    if p.is_file() and '__pycache__' not in p.parts and p.name not in ['MANIFEST.json','MANIFEST.sha256']:
        files.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
manifest=Path('MANIFEST.json');manifest.write_text(json.dumps(dict(files=files,file_count=len(files),total_bytes=sum(f['bytes'] for f in files)),indent=2))
Path('MANIFEST.sha256').write_text(sha(manifest)+'  MANIFEST.json\n')
print(json.dumps(dict(**validation,file_count=len(files),manifest_sha256=sha(manifest))))
