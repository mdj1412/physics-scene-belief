"""Real-video image-space boundary audit. No rigid physics fitting claim."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'contact_model_v4/vendor'))
import cv2
cv2.setNumThreads(1)
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import gaussian_filter1d

DATA=Path('/scratch2/mdj1412/datasets/deform360_contact_belief_audit/raw/008-pink-cloth')
OUT=Path('results/real_boundary');OUT.mkdir(parents=True,exist_ok=True)
CAMS=['brics-odroid-001_cam0','brics-odroid-012_cam0']

def segment(frame,hmin=145,smin=65):
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    m=((hsv[:,:,0]>=hmin)&(hsv[:,:,1]>smin)&(hsv[:,:,2]>50)).astype(np.uint8)
    n,l,s,c=cv2.connectedComponentsWithStats(m,8)
    if n<=1:return m*0
    k=1+np.argmax(s[1:,cv2.CC_STAT_AREA]);return (l==k).astype(np.uint8)
def iou(a,b):return float(((a>0)&(b>0)).sum()/max(1,((a>0)|(b>0)).sum()))
def boundary_distance(a,b):
    ea=cv2.Canny(a*255,100,200)>0;eb=cv2.Canny(b*255,100,200)>0
    da=cv2.distanceTransform((~ea).astype(np.uint8),cv2.DIST_L2,5)
    db=cv2.distanceTransform((~eb).astype(np.uint8),cv2.DIST_L2,5)
    ds=np.r_[da[eb],db[ea]]
    return float(np.quantile(ds,.95)) if len(ds) else None
def encode(mask):
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    contour=max(contours,key=cv2.contourArea).reshape(-1,2).astype(float)
    points=np.vstack([contour,contour[:1]]);dist=np.r_[0,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    u=np.linspace(0,dist[-1],128,endpoint=False)
    poly=np.stack([np.interp(u,dist,points[:,i]) for i in range(2)],1)
    reconstruction=np.zeros_like(mask);cv2.fillPoly(reconstruction,[np.rint(poly).astype(np.int32)],1)
    x,y,w,h=cv2.boundingRect(mask);box=np.zeros_like(mask);box[y:y+h,x:x+w]=1
    best=(-1,0,None)
    for radius in np.linspace(0,min(w,h)/2,31):
        im=Image.new('L',(mask.shape[1],mask.shape[0]),0)
        ImageDraw.Draw(im).rounded_rectangle([x,y,x+w-1,y+h-1],radius=float(radius),fill=1)
        rr=np.array(im);score=iou(mask,rr)
        if score>best[0]:best=(score,float(radius),rr)
    smooth=gaussian_filter1d(poly,1.5,axis=0,mode='wrap')
    a=np.roll(smooth,4,axis=0)-smooth;b=np.roll(smooth,-4,axis=0)-smooth
    cross=abs(a[:,0]*b[:,1]-a[:,1]*b[:,0]);den=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)*np.linalg.norm(a-b,axis=1)
    curvature=2*cross/np.maximum(den,1e-9)
    return dict(bbox_iou=iou(mask,box),rounded_iou=best[0],rounded_radius_px=best[1],contour128_iou=iou(mask,reconstruction),
        bbox_boundary_p95=boundary_distance(mask,box),rounded_boundary_p95=boundary_distance(mask,best[2]),contour128_boundary_p95=boundary_distance(mask,reconstruction),
        curvature_q50_q95=np.quantile(curvature,[.5,.95]).tolist(),area=int(mask.sum()),box=[x,y,w,h]),poly,best[2]

records=[];previews=[]
for camera in CAMS:
    for episode,video in enumerate(sorted((DATA/camera).glob('*.mp4'))[:3]):
        cap=cv2.VideoCapture(str(video));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));i=0
        while True:
            ok,frame=cap.read()
            if not ok:break
            if i%20==0:
                mask=segment(frame)
                if mask.sum()<1000:raise RuntimeError('cloth not found')
                stat,poly,rounded=encode(mask);variants=[]
                for hmin,smin in [(140,55),(150,85)]:
                    alt=segment(frame,hmin,smin)
                    variants.append(dict(hmin=hmin,smin=smin,iou=iou(mask,alt),boundary_p95_px=boundary_distance(mask,alt)))
                stat.update(camera=camera,episode=episode,frame=i,sensitivity=variants);records.append(stat)
                # Store the spatial boundary, rather than only area/centroid.
                np.savez_compressed(OUT/f'boundary_{camera}_{episode}_{i:04d}.npz',polyline_px=poly,
                    signed_distance=cv2.distanceTransform(mask,cv2.DIST_L2,5)-cv2.distanceTransform(1-mask,cv2.DIST_L2,5))
                if i in [0,(count//2//20)*20]:
                    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB);im=Image.fromarray(rgb);draw=ImageDraw.Draw(im)
                    x,y,w,h=stat['box'];draw.rectangle([x,y,x+w,y+h],outline=(0,160,255),width=3)
                    draw.line([tuple(p) for p in np.vstack([poly,poly[:1]])],fill=(255,220,0),width=3)
                    im.thumbnail((640,360));tile=Image.new('RGB',(640,390),'white');tile.paste(im,(0,30))
                    ImageDraw.Draw(tile).text((5,5),f"episode={episode}, frame={i}, cam={camera} | blue=bbox, yellow=contour",fill='black');previews.append(tile)
            i+=1
        cap.release();print('real',camera,episode,i,flush=True)
out=dict(records=records,scope='Image-space representation fidelity and segmentation sensitivity; no boundary GT, metric collision surface, or counterfactual physics validation.')
(OUT/'metrics.json').write_text(json.dumps(out,indent=2))
canvas=Image.new('RGB',(1280,390*((len(previews)+1)//2)),'white')
for j,im in enumerate(previews):canvas.paste(im,((j%2)*640,(j//2)*390))
canvas.save(OUT/'boundary_overlays.jpg')
summary={k:float(np.mean([d[k] for d in records])) for k in ['bbox_iou','rounded_iou','contour128_iou','bbox_boundary_p95','rounded_boundary_p95','contour128_boundary_p95']}
summary.update(n_frames=len(records),n_videos=6,segmentation_sensitivity_mean_iou=float(np.mean([s['iou'] for d in records for s in d['sensitivity']])),segmentation_sensitivity_p95_px=float(np.quantile([s['boundary_p95_px'] for d in records for s in d['sensitivity']],.95)))
(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
