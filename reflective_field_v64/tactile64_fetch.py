"""Fetch immutable first three indexed GelSight Mini markered sequences for field-data audit."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from urllib.request import urlopen
import shutil
REV='239c8ee8156bf11eb6d6c11c004f9a8721528d9d'
ROOT=Path('/scratch2/mdj1412/datasets/v64_taf_markered')/REV
def fetch(item):
    obj,name=item
    url=f'https://huggingface.co/datasets/jiamig/taf-dataset/resolve/{REV}/taf_dataset/gs_mini/gs_mini_obj{obj}/{name}'
    p=ROOT/str(obj)/name
    p.parent.mkdir(parents=True,exist_ok=True)
    with urlopen(url,timeout=90) as stream, p.with_suffix('.pending').open('wb') as f:
        shutil.copyfileobj(stream,f)
    p.with_suffix('.pending').replace(p)
    print(obj,name,p.stat().st_size,flush=True)
jobs=[(i,n) for i in (1,2,3) for n in ('meta/info.json','data/chunk-000/file-000.parquet','videos/observation.image/chunk-000/file-000.mp4')]
if __name__=='__main__':
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(fetch,jobs))
