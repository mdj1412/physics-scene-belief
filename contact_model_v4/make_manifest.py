import hashlib,json,platform,time,sys
from pathlib import Path
from importlib.metadata import version

root=Path(__file__).resolve().parent
files=list(root.glob('*.py'))+list(root.glob('*.md'))
for folder in ['results','reports']:
    files.extend(p for p in (root/folder).rglob('*') if p.is_file())
entries=[]
for p in sorted(files):
    entries.append(dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
manifest=dict(created_unix=time.time(),host=platform.node(),python=sys.version,
    packages={k:version(k) for k in ['torch','numpy','scipy','pybullet','matplotlib','Pillow']},
    counts={folder:len(list((root/'results'/folder).glob('case_*.json'))) for folder in ['main','refined','continuous_proxy']},
    calibration_count=len(list((root/'results/calibration').glob('*.json'))),files=entries)
(root/'run_manifest.json').write_text(json.dumps(manifest,indent=2));print('manifest',len(entries),'files',manifest['counts'])
