"""Raw RGB -> bounding geometry and corner-area deficit.

For a rounded rectangle A_bbox - A_mask = (4-pi) b^2. This descriptor
retains curvature discarded by bounding-box-only geometry. Pixel phase,
blur and segmentation produce bias/noise, calibrated independently.
Units of features are mm, mm, mm^2. No hidden scene state in extraction.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'contact_model_v4/vendor'))
import cv2
cv2.setNumThreads(1)
import numpy as np
from PIL import Image,ImageDraw,ImageFilter

SCALES={'high':4000.,'low':1000.,'pixel_only':4000.}
def layout(g,d,family):
    if family=='corner':return [( .12,.014+d+g,.015,d)]
    if family=='gate':return [(.12,g+d+.04,.015,.04),(.12,g-d-.04,.015,.04)]
    if family=='deflection':return [(.08,.014+d+g,.015,d),(.22,-.058+g,.015,.035)]
    raise ValueError(family)

def render(g,d,b,family,quality,rng):
    scale=SCALES[quality];size=int(.256*scale);ss=2
    # Camera/localization uncertainty shared by both feature baselines.
    gg,dd=np.array([g,d])+rng.normal(0,0. if quality=='pixel_only' else .0015,2)
    xphase=rng.uniform(-.5,.5)/scale
    im=Image.new('RGB',(size*ss,size*ss),(230,230,230));draw=ImageDraw.Draw(im)
    for x,y,hx,hy in layout(gg,dd,family):
        lo=(np.array([x-hx+xphase,y-hy])-np.array([.1,0]))*scale*ss+size*ss/2
        hi=(np.array([x+hx+xphase,y+hy])-np.array([.1,0]))*scale*ss+size*ss/2
        draw.rounded_rectangle([*lo,*hi],radius=b*scale*ss,fill=(30,60,210))
    im=im.resize((size,size),Image.Resampling.LANCZOS).filter(ImageFilter.GaussianBlur(rng.uniform(.35,.8)))
    # Image-space noise, not noise directly added to estimated curvature.
    a=np.array(im,dtype=np.float32)+rng.normal(0,3,(size,size,3))
    return Image.fromarray(np.clip(a,0,255).astype(np.uint8))

def extract(image,family,quality,return_detail=False):
    a=np.asarray(image).astype(float);h,w=a.shape[:2];scale=SCALES[quality]
    soft=np.clip((a[:,:,2]-a[:,:,0])/180.,0,1)
    mask=(soft>.5).astype(np.uint8)
    xcoord=(np.arange(w)-w/2)/scale+.1
    roi=np.abs(xcoord-(.08 if family=='deflection' else .12))<.019
    mask*=roi[None,:]
    num,labels,stats,centers=cv2.connectedComponentsWithStats(mask,8)
    ids=[k for k in range(1,num) if stats[k,cv2.CC_STAT_AREA]>50]
    ids=sorted(ids,key=lambda k:stats[k,cv2.CC_STAT_AREA],reverse=True)[:2 if family=='gate' else 1]
    if len(ids)!=(2 if family=='gate' else 1):raise RuntimeError('geometry components missing')
    comps=[];deficits=[]
    for k in ids:
        x,y,ww,hh,area=stats[k];comps.append((x,y,ww,hh))
        # Discrete pixel occupancy of the entire component boundary.
        deficit=float(ww*hh-area)/((4-np.pi)*scale**2)*1e6
        deficits.append(deficit)
    comps.sort(key=lambda z:z[1])
    if family=='gate':
        x,y,ww,hh=comps[0];upper=(y+hh-.5-h/2)/scale
        x,y,ww,hh=comps[1];lower=(y-.5-h/2)/scale
        g=(upper+lower)/2;d=(lower-upper)/2
    else:
        x,y,ww,hh=comps[0];top=(y-.5-h/2)/scale;bottom=(y+hh-.5-h/2)/scale
        d=(bottom-top)/2;g=top-.014
    feat=np.array([g*1e3,d*1e3,np.mean(deficits)])
    if return_detail:return feat,dict(mask=mask,components=comps,deficits=deficits)
    return feat

def calibration_np(cal,family,quality,b):
    c=cal[family][quality];grid=np.asarray(c['b_grid']);b=np.asarray(b)
    bias=np.stack([np.interp(b,grid,np.array(c['mean'])[:,j]) for j in range(3)],axis=-1)
    # A common calibrated covariance avoids unstable tiny-bin estimates and
    # keeps a well-defined continuous observation distribution near b=0.
    return bias,np.array(c['cov'])
