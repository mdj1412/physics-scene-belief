"""Validate completed outputs after interrupt; quarantine only incomplete writes."""
import json,time
from pathlib import Path
from zipfile import BadZipFile
import numpy as np
out=Path('results/main');archive=Path('results/interrupted_partial');archive.mkdir(exist_ok=True)
kept=[];quarantined=[]
for p in sorted(out.glob('case_*.json')):
    case=int(p.stem.split('_')[1]);npz=out/f'factual_{case:03d}.npz'
    try:
        d=json.loads(p.read_text());assert d['measurement_model']=='finite_frame_50ms' and len(d['records'])==8
        with np.load(npz) as n:
            assert n['theta'].shape==(4096,4) and n['rounding'].shape==(4096,)
            assert np.isfinite(n['theta']).all()
        kept.append(case)
    except (ValueError,KeyError,AssertionError,FileNotFoundError,OSError,EOFError,BadZipFile):
        p.rename(archive/p.name)
        if npz.exists():npz.rename(archive/npz.name)
        quarantined.append(case)
record=dict(time=time.time(),kept=kept,quarantined=quarantined,pending=sorted(set(range(72))-set(kept)))
Path('results/validation/queue_resume.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
