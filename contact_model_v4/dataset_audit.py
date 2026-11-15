import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'vendor'))
import cv2
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path('/scratch2/mdj1412/datasets/deform360_contact_belief_audit/raw/008-pink-cloth')
OUT=Path('results/real_data_audit');OUT.mkdir(parents=True,exist_ok=True)
CAMERAS=['brics-odroid-001_cam0','brics-odroid-012_cam0']

def times(path):return np.array([int(line.split('_')[1]) for line in path.read_text().splitlines()],dtype=np.int64)

def main():
    metadata=json.loads((ROOT/'metadata.json').read_text());records=[];previews=[]
    # Official hashed calibration files use pickle-backed dictionaries.
    calibration={name:np.load(ROOT/'calibration_refined'/f'{name}.npy',allow_pickle=True).item() for name in ['intrinsics','extrinsics','dist']}
    for episode in range(3):
        timestamps=[]
        for camera in CAMERAS:
            video=sorted((ROOT/camera).glob('*.mp4'))[episode];tt=times(video.with_suffix('.txt'));timestamps.append(tt)
            cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS);declared=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));sample=[];n=0;shots=[]
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            while True:
                ok,frame=cap.read()
                if not ok:break
                if n%10==0:
                    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
                    mask=(((hsv[:,:,0]>145)|(hsv[:,:,0]<10))&(hsv[:,:,1]>65)&(hsv[:,:,2]>50)).astype(np.uint8)
                    num,labels,stats,centers=cv2.connectedComponentsWithStats(mask,8)
                    if num>1:
                        k=1+int(stats[1:,cv2.CC_STAT_AREA].argmax());area=int(stats[k,cv2.CC_STAT_AREA]);xy=centers[k].tolist()
                    else:area=0;xy=[None,None]
                    sample.append(dict(frame=n,area=area,centroid=xy))
                if n in [0,declared//2,declared-1]:
                    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB);im=Image.fromarray(rgb);im.thumbnail((384,216));shots.append(im)
                n+=1
            cap.release();assert n==declared==len(tt),(video,n,declared,len(tt))
            assert (np.diff(tt)>0).all()
            key=camera if camera in calibration['intrinsics'] else None
            record=dict(episode=episode,camera=camera,action=metadata['sequences'][str(episode)]['action'],frames=n,fps=fps,duration_s=float((tt[-1]-tt[0])/1e6),width=width,height=height,calibrated=key is not None,samples=sample)
            records.append(record)
            strip=Image.new('RGB',(1152,240),'white')
            for j,im in enumerate(shots):strip.paste(im,(384*j,24))
            ImageDraw.Draw(strip).text((5,5),f"episode {episode}: {record['action']} | {camera}",fill='black');previews.append(strip)
        a,b=timestamps;ids=np.searchsorted(b,a).clip(0,len(b)-1);before=(ids-1).clip(0);use=np.where(abs(b[ids]-a)<abs(b[before]-a),ids,before);delta=abs(b[use]-a)/1000
        records.append(dict(episode=episode,alignment_median_ms=float(np.median(delta)),alignment_p95_ms=float(np.quantile(delta,.95)),alignment_max_ms=float(delta.max()),same_index_median_ms=float(np.median(abs(a[:min(len(a),len(b))]-b[:min(len(a),len(b))])/1000))))
    canvas=Image.new('RGB',(1152,240*len(previews)),'white')
    for j,im in enumerate(previews):canvas.paste(im,(0,j*240))
    canvas.save(OUT/'recorded_interactions.jpg')
    report=dict(records=records,calibration_camera_counts={k:len(v) for k,v in calibration.items()},scope='Data integrity, synchronization and simple RGB feature audit only. No fitted deformable physics or counterfactual accuracy claim.',correspondence_warning='Silhouette centroids in two views need not refer to the same material point; they are not triangulated as GT.')
    (OUT/'audit.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2));print('decoded',sum(x.get('frames',0) for x in records),'frames',flush=True)
if __name__=='__main__':main()
