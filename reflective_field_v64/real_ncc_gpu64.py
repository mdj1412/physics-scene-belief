"""GPU timing of static patch localization on real images; no policy or contact prediction."""
import csv,io,json,os,socket,time,zipfile
from pathlib import Path
import numpy as np
import torch
from static_light64 import DATA, ROOT, read_image

@torch.inference_mode()
def localize(tile_cpu, ref_cpu):
    dev=torch.device("cuda:0")
    tile=torch.as_tensor(tile_cpu,device=dev)
    ref=torch.as_tensor(ref_cpu,device=dev)
    patches=tile.unfold(-2,15,1).unfold(3,15,1)
    centered=patches-patches.mean((-1,-2),keepdim=True)
    ref=ref-ref.mean((-1,-2),keepdim=True)
    ref=ref[:,None,None,None,:,:]
    numerator=(centered*ref).sum((-1,-2))
    den=((centered.square().sum((-1,-2)))*(ref.square().sum((-1,-2)))).sqrt()
    scores=torch.where(den>0,numerator/den,-2.).flatten(2)
    top=scores.max(-1).values
    v=torch.arange(-8,9,device=dev)
    distance=(v[:,None].square()+v[None,:].square()).flatten()
    winner=torch.where(scores>=top[...,None]-1e-12,distance,999).argmin(-1)
    dx=winner%17-8
    accepted=(top>=.8).all(1)
    shifted=(dx>=2).all(1)|(dx<=-2).all(1)
    return torch.stack((accepted.sum(),(accepted&shifted).sum())).tolist()

def main():
    assert os.getenv('SLURM_JOB_ID') and torch.cuda.device_count()==1
    baseline=json.loads((ROOT/'static_tracks64.json').read_text())
    points={}
    with (ROOT/'static_tracks64.csv').open() as f:
        for r in csv.DictReader(f):
            if r['light_index']=='0': points.setdefault((r['object'],int(r['reverse'])),[]).append((int(r['x']),int(r['y'])))
    cases=[]
    with zipfile.ZipFile(DATA/'DiLiGenT.zip') as z:
        for name,spec in baseline['directions'].items():
            prefix='DiLiGenT/pmsData/'+name+'/'
            intens=np.loadtxt(io.BytesIO(z.read(prefix+'light_intensities.txt')))
            fn=z.read(prefix+'filenames.txt').decode().split()
            pics=[read_image(z,prefix+f,intens[fn.index(f)]) for f in spec['filenames']]
            for reverse in [0,1]:
                images=pics[::-1] if reverse else pics
                xy=points[name,reverse]
                ref=np.stack([images[0][y-7:y+8,x-7:x+8] for x,y in xy])
                tiles=np.array([[im[y-15:y+16,x-15:x+16] for im in images[1:]] for x,y in xy])
                record=next(r for r in baseline['per_object'] if r['object']==name and r['reverse']==reverse and r['threshold']==.8)
                expect=[record['accepted'],record['consistent_shift']]
                cases.append((name,reverse,tiles,ref,expect))
    rng=np.random.default_rng(640117)
    times=[]
    for j in range(12):
        for i in rng.permutation(len(cases)):
            name,reverse,tile,ref,expect=cases[i]
            torch.cuda.synchronize()
            start=time.perf_counter_ns()
            result=localize(tile,ref)
            torch.cuda.synchronize()
            ms=(time.perf_counter_ns()-start)*1e-6
            assert result==expect,(name,reverse,result,expect)
            if j>=2:times.append(dict(object=name,reverse=reverse,patches=len(ref),ms=ms))
    vals=[r['ms'] for r in times]
    (ROOT/'real_ncc_gpu64.json').write_text(json.dumps(dict(algorithm='float64 NCC of 3 real light images vs reference, CPU-loaded ROI to host counts',gpu=torch.cuda.get_device_name(0),slurm=os.environ['SLURM_JOB_ID'],hostname=socket.gethostname(),measured_calls=len(vals),p90_ms=float(np.quantile(vals,.90)),p50_ms=float(np.median(vals)),observations=times),indent=2)+'\n')
if __name__=='__main__':main()
