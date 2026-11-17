"""A forward/backward consistency check on real static DiLiGenT surfaces."""
import csv,io,json,os,sys,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'reflective_field_v64'
sys.path.insert(0,str(OLD))
from static_light64 import DATA,read_image,compare

def main():
    assert os.getenv('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    directions=json.loads((OLD/'static_tracks64.json').read_text())['directions']
    groups={}
    with (OLD/'static_tracks64.csv').open() as f:
        for row in csv.DictReader(f):
            key=(row['object'],int(row['reverse']),int(row['x']),int(row['y']))
            groups.setdefault(key,[]).append(row)
    rows=[]
    with zipfile.ZipFile(DATA/'DiLiGenT.zip') as z:
        for obj,spec in directions.items():
            folder=f'DiLiGenT/pmsData/{obj}/'
            ii=np.loadtxt(io.BytesIO(z.read(folder+'light_intensities.txt')))
            files=z.read(folder+'filenames.txt').decode().split()
            pics=[read_image(z,folder+n,ii[files.index(n)]) for n in spec['filenames']]
            for (nm,rev,x,y),track in groups.items():
                if nm!=obj:continue
                imgs=pics[::-1] if rev else pics
                ref=imgs[0];back=[];dx=[]
                ok=all(float(r['NCC'])>=.8 for r in track[1:])
                if not ok: continue
                for im,obs in zip(imgs[1:],track[1:]):
                    sx,sy=int(obs['dx']),int(obs['dy']);dx.append(sx)
                    xx,yy=x+sx,y+sy
                    assert 15<=xx<im.shape[1]-15 and 15<=yy<im.shape[0]-15
                    value=compare(ref,im,xx,yy)
                    back.append(value is not None and value[2]>=.8 and np.hypot(sx+value[0],sy+value[1])<=1)
                wrong=all(d>=2 for d in dx) or all(d<=-2 for d in dx)
                rows.append((nm,rev,x,y,int(wrong),int(all(back))))
    def summarize(data):
        return [len(data),sum(v[-2] for v in data),sum(v[-1] for v in data),sum(v[-1]*v[-2] for v in data)]
    counts=[summarize([r for r in rows if r[1]==i]) for i in (0,1)]
    assert [c[:2] for c in counts]==[[62,9],[64,14]]
    (ROOT/'static_closure65.json').write_text(json.dumps(dict(columns=['forward accepted','forward wrong','roundtrip accepted','roundtrip wrong'],
         forward_and_reverse=counts,grouped={nm:[summarize([r for r in rows if r[0]==nm and r[1]==i]) for i in (0,1)] for nm in directions},
         slurm=os.getenv('SLURM_JOB_ID')),indent=2)+'\n')
    with (ROOT/'static_closure65.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['object','reverse','x','y','consistent_forward_horizontal_error','roundtrip_pass']);w.writerows(rows)
if __name__=='__main__':main()
