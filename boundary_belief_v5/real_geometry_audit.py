"""Stronger real-image controls: orientation, quadrilateral, convex hull.

Added after visual inspection exposed pose/cropping confounds. No test physics
claims; only reconstruction of the canonical segmented visible region.
"""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'contact_model_v4/vendor'))
import cv2
cv2.setNumThreads(1)
import numpy as np
root=Path('results/real_boundary');data=json.loads((root/'metrics.json').read_text());records=[]
def fill(points,shape):
    a=np.zeros(shape,np.uint8);cv2.fillPoly(a,[np.rint(points).astype(np.int32)],1);return a
def iou(a,b):return float(((a>0)&(b>0)).sum()/max(1,((a>0)|(b>0)).sum()))
for d in data['records']:
    path=root/f"boundary_{d['camera']}_{d['episode']}_{d['frame']:04d}.npz"
    mask=(np.load(path)['signed_distance']>0).astype(np.uint8)
    cs,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE);contour=max(cs,key=cv2.contourArea)
    rect=cv2.minAreaRect(contour);(cx,cy),(w,h),angle=rect
    box=fill(cv2.boxPoints(rect),mask.shape);best=iou(mask,box);best_r=0.
    a=np.deg2rad(angle);rot=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
    for radius in np.linspace(0,min(w,h)/2,31):
        pts=[]
        for lo in [0,np.pi/2,np.pi,3*np.pi/2]:
            angles=np.linspace(lo,lo+np.pi/2,16);normal=np.stack([np.cos(angles),np.sin(angles)],1)
            signs=np.sign([np.cos(lo+np.pi/4),np.sin(lo+np.pi/4)])
            pts.extend(signs*(np.array([w,h])/2-radius)+radius*normal)
        candidate=fill(np.array(pts)@rot.T+np.array([cx,cy]),mask.shape);score=iou(mask,candidate)
        if score>best:best=score;best_r=float(radius)
    quad_best=iou(mask,box);perimeter=cv2.arcLength(contour,True)
    for epsilon in np.linspace(.002,.12,80):
        poly=cv2.approxPolyDP(contour,epsilon*perimeter,True)
        if len(poly)==4:quad_best=max(quad_best,iou(mask,fill(poly.reshape(-1,2),mask.shape)))
    hull=fill(cv2.convexHull(contour).reshape(-1,2),mask.shape)
    cropped=bool(mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any())
    records.append(dict(camera=d['camera'],episode=d['episode'],frame=d['frame'],cropped=cropped,
        axis_bbox_iou=d['bbox_iou'],axis_rounded_iou=d['rounded_iou'],oriented_box_iou=iou(mask,box),oriented_rounded_iou=best,
        oriented_rounding_px=best_r,quad_iou=quad_best,hull_iou=iou(mask,hull),contour128_iou=d['contour128_iou']))
keys=['axis_bbox_iou','axis_rounded_iou','oriented_box_iou','oriented_rounded_iou','quad_iou','hull_iou','contour128_iou']
groups={}
for group in ['all','not_cropped','cropped']:
    ds=[d for d in records if group=='all' or d['cropped']==(group=='cropped')]
    groups[group]=dict(n=len(ds),**{k:float(np.mean([d[k] for d in ds])) if ds else None for k in keys})
(root/'strong_geometry_controls.json').write_text(json.dumps(dict(groups=groups,records=records),indent=2));print(json.dumps(groups,indent=2))
