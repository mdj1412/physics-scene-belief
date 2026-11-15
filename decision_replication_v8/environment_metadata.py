import importlib.metadata as md,json,os,platform,subprocess
from pathlib import Path
packages={}
for name in ['torch','numpy','Pillow','matplotlib']:
 try:packages[name]=md.version(name)
 except md.PackageNotFoundError:pass
vendor=Path('../contact_model_v4/vendor').resolve()
for d in md.distributions(path=[str(vendor)]):
 if 'opencv' in d.metadata['Name'].lower():packages[d.metadata['Name']]=d.version
sm=subprocess.run(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv,noheader'],capture_output=True,text=True,check=True)
d=dict(host=platform.node(),platform=platform.platform(),python=platform.python_version(),job=os.environ.get('SLURM_JOB_ID'),packages=packages,opencv_vendor=str(vendor),gpu=sm.stdout.strip())
Path('results/environment.json').write_text(json.dumps(d,indent=2));print(json.dumps(d))
