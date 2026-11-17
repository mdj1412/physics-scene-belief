"""Audit the public SHPB sensor channels and digitize the authors' synchronized strain curves."""
import json
import os
from pathlib import Path
import hashlib
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent
DATA=Path('/scratch2/mdj1412/datasets/physics_scene_belief_v64/opara655/Longitudinal Concrete Analysis Data')

def audit(test, step):
    gauge=next(test.rglob('*80MHz.txt'))
    plot=test/'05_validation-results'/'strain validation.tif'
    arr=np.array(Image.open(plot).convert('RGB')).astype(int)
    h,w,_=arr.shape
    r,g,b=arr.transpose(2,0,1)
    blue=(b>160)&(r<50)&(g<50)
    orange=(r>210)&(g>100)&(g<200)&(b<45)
    gr=(g>75)&(r<25)&(b<25)
    # Use the early experiment path, which is to the left of the legend in each plot.
    for mask in (blue,orange,gr):
        mask[:,int(.44*w):]=False
        mask[:int(.08*h)]=False;mask[int(.83*h):]=False
    marker=np.where(gr.sum(0)>65)[0]
    x=int(round(float(np.median(marker))))
    grid=(np.max(arr,2)-np.min(arr,2)<3)&(r>145)&(r<219)
    gridrows=np.where(grid[:,int(.19*w):int(.44*w)].sum(1)>.18*w)[0]
    gridrows=gridrows[(gridrows>.09*h)&(gridrows<.831*h)]
    runs=np.split(gridrows,np.flatnonzero(np.diff(gridrows)>1)+1)
    rowcentres=np.array([np.median(grp) for grp in runs if len(grp)])
    space=np.median(np.diff(rowcentres))
    assert space>50 and len(rowcentres)>3 and np.ptp(marker)<14
    # The horizontal zero grid line passes through the first visible optical observations.
    first=np.flatnonzero(blue.any(0))[0]
    first_y=np.median(np.where(blue[:,first:first+17])[0])
    y0=rowcentres[np.argmin(abs(rowcentres-first_y))]
    print(test.name, rowcentres, first_y, y0, x, flush=True)
    assert abs(y0-first_y)<0.10*space
    def estimate(mask):
        yy=np.where(mask[:,x-8:x+9])[0]
        assert len(yy)>5
        return float((y0-np.median(yy))*step/space)
    with gauge.open() as f:
        head=[next(f).strip() for _ in range(3)]
        a=np.loadtxt(f)
    assert head[0].split()==['Time','Channel','F','Channel','G','Channel','H']
    assert a.shape[1]==4
    hz=1000 /(a[-1,0]-a[0,0])*(len(a)-1)
    assert 79.9e6 < hz < 80.1e6
    images=sorted((test/'01_raw-data').rglob('*.tif'))
    assert len(images)==4
    with Image.open(images[0]) as im:
        image_size=list(im.size)
    return dict(test=test.name,scope_samples=len(a),scope_hz=hz,scope_time_ms=[a[0,0],a[-1,0]],channel_ranges_V=np.stack([a[:,1:].min(0),a[:,1:].max(0)]).T.tolist(),image_size=image_size,
        image_files_checked=[im.name for im in images],x_green_px=x,grid_spacing_px=space,zero_row_px=y0,grid_step_percent=step,
        optical_strain_percent=estimate(blue),gauge_strain_percent=estimate(orange),
        gauge_file_sha256=hashlib.sha256(gauge.read_bytes()).hexdigest(),strain_plot_sha256=hashlib.sha256(plot.read_bytes()).hexdigest())

if __name__=='__main__':
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    rows=[audit(DATA/f'0{i}-Test',(.25,.25,.25,.5)[i-1]) for i in range(1,5)]
    archive=json.loads((ROOT/'opara64_inventory.json').read_text())
    counts={}
    for test in rows:
        nums=[item for item in archive if '/'+test['test']+'/' in item['name'] and '01_raw-data/Kirana' in item['name'] and item['name'].endswith('.tif')]
        counts[test['test']]=len(nums)
    (ROOT/'opara64_results.json').write_text(json.dumps(dict(source='10.25532/OPARA-655',job=os.environ['SLURM_JOB_ID'],rows=rows,
        gauge_and_optical_strain_positive_at_published_UCS=sum(row['optical_strain_percent']>0 and row['gauge_strain_percent']>0 for row in rows),
        archive_entries=len(archive),raw_tiff_per_test=counts),indent=2)+'\n')
