"""Synthetic RGB centroid/bounding-box sensor, not learned perception.

All raw generated measurements pass through rasterization and color detection.
Known calibration/object colors are privileged experimental assumptions.
"""
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFilter
from engine import tensor,boxes,simulate,NOISE

def centroid(image,color):
    a=np.asarray(image).astype(float)
    if color=='red':mask=(a[:,:,0]>130)&(a[:,:,0]-a[:,:,1]>60)&(a[:,:,0]-a[:,:,2]>60)
    else:mask=(a[:,:,2]>120)&(a[:,:,2]-a[:,:,0]>50)
    yy,xx=np.where(mask)
    if len(xx)==0:raise RuntimeError('RGB detection failed')
    return np.array([xx.mean(),yy.mean()])

def observe(theta,rounding,family,action,rng,save=None):
    theta=np.asarray(theta).copy()
    if action=='view':
        # Structured image-space localization error, independently calibrated g,d.
        theta[[0,3]]+=rng.normal(0,NOISE['view'])
        c,h=boxes(tensor(theta[None,:]),family);c=c.cpu().numpy()[0];h=h.cpu().numpy()[0]
        scale=4000.;size=1024;origin=np.array([.1,0.])
        image=Image.new('RGB',(size,size),(230,230,230));draw=ImageDraw.Draw(image)
        for ci,hi in zip(c,h):
            lo=(ci-hi-origin)*scale+size/2;up=(ci+hi-origin)*scale+size/2
            draw.rounded_rectangle([*lo,*up],radius=max(0,rounding*scale),fill=(30,60,210))
        image=image.filter(ImageFilter.GaussianBlur(.45));a=np.asarray(image).astype(float)
        mask=(a[:,:,2]-a[:,:,0]>50)&(a[:,:,2]>120)
        # Known scene family and first obstacle x window; no GT masks or coordinates.
        xpix=(np.arange(size)-size/2)/scale+origin[0]
        window=abs(xpix-(.08 if family=='deflection' else .12))<.018
        ys,xs=np.where(mask&window[None,:]);ymin=(ys.min()-size/2)/scale;ymax=(ys.max()-size/2)/scale
        if family=='gate':
            rowmask=(mask&window[None,:]).any(1);mid=int(round(size/2))
            # Two separated components, detected from contiguous foreground rows.
            rows=np.where(rowmask)[0];gaps=np.where(np.diff(rows)>1)[0]
            if not len(gaps):raise RuntimeError('gate masks merged')
            split=gaps[np.argmax(np.diff(rows)[gaps])]
            innerlo=(rows[split]-size/2)/scale;innerhi=(rows[split+1]-size/2)/scale
            result=np.array([(innerlo+innerhi)/2,(innerhi-innerlo)/2])
        else:
            d=(ymax-ymin)/2;g=(ymin+ymax)/2-d-.014;result=np.array([g,d])
        images=[image]
    else:
        shift={'factual':0.,'left':-.022,'right':.022,'center':.009}[action]
        out=simulate(tensor(theta[None,:]),tensor([rounding]),family,shift)
        if bool(out['overflow'].any()):raise RuntimeError('overflow in sensor')
        # Render two frames from the verified collision-free final 50 ms.
        if float(out['last_event'][0])>.75:raise RuntimeError('late collision invalidates free-flight camera')
        p=out['position'][0].cpu().numpy();v=out['velocity'][0].cpu().numpy()
        size=1536;scale=900.;dt=.05;origin=np.array([0.,0.]);coords=[];images=[]
        for point in [p-v*dt,p]:
            point=point+rng.normal(0,.025*dt/np.sqrt(2),2)
            pixel=(point-origin)*scale+size/2
            image=Image.new('RGB',(size,size),(230,230,230));draw=ImageDraw.Draw(image)
            rad=.02*scale;draw.ellipse([pixel[0]-rad,pixel[1]-rad,pixel[0]+rad,pixel[1]+rad],fill=(210,35,35))
            image=image.filter(ImageFilter.GaussianBlur(.6));coords.append(centroid(image,'red'));images.append(image)
        result=(coords[1]-coords[0])/scale/dt
        if action=='factual':result[1]=0. # front projection supplies x motion only
    if save:
        path=Path(save);path.mkdir(parents=True,exist_ok=True)
        for j,im in enumerate(images):im.save(path/f'{action}_{j}.png')
    return result
