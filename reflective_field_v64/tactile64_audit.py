"""Audit three prespecified real sequences. Channel ablation never defines illumination intervention."""
import hashlib
import json
import os
from pathlib import Path
import av
import cv2
import numpy as np
import pyarrow.parquet as pq
from tactile64_fetch import ROOT, REV

def audit(obj):
    directory=ROOT/str(obj)
    meta=json.loads((directory/'meta/info.json').read_text())
    data=pq.read_table(directory/'data/chunk-000/file-000.parquet')
    rows=data.slice(0,240)
    ts=rows['timestamp'].to_numpy()
    assert np.array_equal(rows['frame_index'].to_numpy(),np.arange(240))
    assert (rows['episode_index'].to_numpy()==0).all()
    observations=np.array(rows['observation.force_torque'].to_pylist())
    pressure=np.array(rows['observation.pressure_matrix'].to_pylist())
    assert pressure.shape==(240,12,12) and observations.shape==(240,6)
    assert np.isfinite(pressure).all() and np.isfinite(observations).all()
    with av.open(str(directory/'videos/observation.image/chunk-000/file-000.mp4')) as movie:
        stream=movie.streams.video[0]
        frames=[]; times=[]
        for f in movie.decode(stream):
            frames.append(f.to_ndarray(format='rgb24'));times.append(float(f.time))
            if len(frames)==len(ts):
                break
    assert len(frames)==240 and frames[0].shape==(240,320,3)
    assert np.abs(np.array(times)-ts).max()<.001
    counts={name:{'co_valid_features':0,'both_moves':0,'opposite':0} for name in ['R-G','R-B','G-B']}
    windows=[]
    # Each 6-frame block contributes one non-overlapping 5-frame displacement.
    for start in range(0,235,6):
        a=frames[start];b=frames[start+5]
        gray=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY)
        points=cv2.goodFeaturesToTrack(gray,100,.01,7)
        assert points is not None
        n=len(points);disp=[];valid=[]
        for ch in range(3):
            aa=np.ascontiguousarray(a[...,ch]);bb=np.ascontiguousarray(b[...,ch])
            p,forward,_=cv2.calcOpticalFlowPyrLK(aa,bb,points,None)
            q,backward,_=cv2.calcOpticalFlowPyrLK(bb,aa,p,None)
            ok=(forward.ravel()>0)&(backward.ravel()>0)&(np.linalg.norm(q[:,0]-points[:,0],axis=1)<=.5)
            disp.append(p[:,0,0]-points[:,0,0]);valid.append(ok)
        for i,j,name in ((0,1,'R-G'),(0,2,'R-B'),(1,2,'G-B')):
            ok=valid[i]&valid[j]
            large=ok&(np.abs(disp[i])>=.2)&(np.abs(disp[j])>=.2)
            opposite=large&(np.sign(disp[i])!=np.sign(disp[j]))
            vals=[int(x.sum()) for x in [ok,large,opposite]]
            for k,val in zip(counts[name],vals):
                counts[name][k]+=val
            windows.append([start,name,n,*vals])
    return dict(obj=obj,frames_in_metadata=meta['total_frames'],episodes_in_metadata=meta['total_episodes'],
        parquet_rows=data.num_rows,parquet_schema=data.column_names,episode0_analysed_frames=len(frames),
        time_max_error_seconds=np.abs(times-ts).max().item(),fz_range=[observations[:,2].min(),observations[:,2].max()],
        pressure_range=[pressure.min(),pressure.max()],windows=windows,counts=counts,
        sha256={str(p.relative_to(directory)):hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.rglob('*') if p.is_file()})
def main():
    assert os.getenv('SLURM_JOB_ID') and os.getenv('CUDA_VISIBLE_DEVICES')==''
    cv2.setNumThreads(1)
    result={'revision':REV,'slot':'first 240 indexed frames of episode zero, first three numerically indexed markered GelSight Mini sequences','rows':[audit(i) for i in (1,2,3)]}
    (Path(__file__).parent/'tactile64_results.json').write_text(json.dumps(result,indent=2)+'\n')
    for r in result['rows']:print(r['obj'],r['counts'],r['fz_range'],r['pressure_range'],flush=True)
if __name__=='__main__':main()
