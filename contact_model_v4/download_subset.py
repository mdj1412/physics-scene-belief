"""Download a bounded official Deform360 subset, preserving immutable revision."""
import json,urllib.request,hashlib,time,argparse
from pathlib import Path
ROOT=Path('/scratch2/mdj1412/datasets/deform360_contact_belief_audit')
BASE='https://huggingface.co'
def get(url):
    with urllib.request.urlopen(url,timeout=120) as r:return r.read()
def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    parser=argparse.ArgumentParser();parser.add_argument('--revision',default='5ea8c5d3fc7b4a7b4f9f921f2ceb1de24610f6a4');args=parser.parse_args()
    revision=args.revision
    entries=json.loads(get(BASE+f'/api/datasets/brownu/deform360/tree/{revision}/raw/008-pink-cloth?recursive=true&limit=1000'))
    (ROOT/'pink_inventory.json').write_text(json.dumps(entries,indent=2))
    # Recursive listing can be paginated. Query small required directories
    # directly instead of mistaking the first page for a full object manifest.
    for folder in ['raw/008-pink-cloth','raw/008-pink-cloth/calibration_refined','raw/008-pink-cloth/brics-odroid-001_cam0','raw/008-pink-cloth/brics-odroid-012_cam0']:
        entries.extend(json.loads(get(BASE+f'/api/datasets/brownu/deform360/tree/{revision}/{folder}?limit=1000')))
    files=list({e['path']:e for e in entries if e['type']=='file'}.values());chosen=[]
    for camera in ['brics-odroid-001_cam0','brics-odroid-012_cam0']:
        videos=sorted([e for e in files if f'/{camera}/' in e['path'] and e['path'].endswith('.mp4')],key=lambda e:e['path'])[:3]
        for v in videos:
            chosen.append(v);chosen.extend([e for e in files if e['path']==v['path'][:-4]+'.txt'])
    chosen.extend([e for e in files if 'calibration_refined/' in e['path'] or e['path'].endswith('metadata.json')])
    total=sum(e['size'] for e in chosen);assert total<2_000_000_000,total
    print(json.dumps(dict(revision=revision,files=len(chosen),bytes=total,paths=[e['path'] for e in chosen])),flush=True)
    manifest=[]
    for e in chosen:
        dest=ROOT/e['path'];dest.parent.mkdir(parents=True,exist_ok=True)
        url=BASE+f'/datasets/brownu/deform360/resolve/{revision}/'+e['path']
        if not dest.exists() or dest.stat().st_size!=e['size']:dest.write_bytes(get(url))
        assert dest.stat().st_size==e['size'],str(dest)
        sha=hashlib.sha256(dest.read_bytes()).hexdigest();manifest.append(dict(path=e['path'],bytes=e['size'],sha256=sha,url=url))
        print('downloaded',e['path'],e['size'],flush=True)
    (ROOT/'download_manifest.json').write_text(json.dumps(dict(dataset='brownu/deform360',revision=revision,files=manifest,time=time.time()),indent=2))
if __name__=='__main__':main()
