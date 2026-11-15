import json,time,hashlib,os
from pathlib import Path
import numpy as np
R=Path('results')
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,ensure_ascii=False))
def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def sources():return {str(p):sha(p) for p in sorted(Path('.').glob('*.py'))}|{'PROTOCOL.md':sha('PROTOCOL.md')}
def require():
    assert os.environ.get('SLURM_JOB_ID'),'Compute node allocation required'
def save(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**arrays)
def freeze(path,files,**extra):write(path,dict(time=time.time(),files={str(p):sha(p) for p in files},**extra))
