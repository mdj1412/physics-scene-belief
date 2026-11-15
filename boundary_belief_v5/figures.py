import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from geometry import layout

out=Path('results/analysis');out.mkdir(parents=True,exist_ok=True)
def draw_envelope(ax,g,d,b,family,color,ls,label):
    first=True
    for x,y,hx,hy in layout(g,d,family):
        pts=[]
        for lo in [0,np.pi/2,np.pi,3*np.pi/2]:
            angles=np.linspace(lo,lo+np.pi/2,32)
            signs=np.sign([np.cos(lo+np.pi/4),np.sin(lo+np.pi/4)])
            center=np.array([x,y])+signs*(np.array([hx,hy])-b)
            pts.extend(center+(.02+b)*np.stack([np.cos(angles),np.sin(angles)],1))
        pts=np.array(pts);pts=np.vstack([pts,pts[:1]])
        ax.plot(pts[:,0]*1e3,pts[:,1]*1e3,color=color,linestyle=ls,lw=1.4,label=label if first else None);first=False
fig,axs=plt.subplots(2,3,figsize=(12,8))
for col,case in enumerate([1,25,49]):
    d=json.loads((Path('results/main')/f'case_{case:03d}.json').read_text());f=d['family']
    for ax,q in zip(axs[:,col],['high','low']):
        im=Image.open(Path('results/main/images')/f'{case:03d}'/f'{q}.png')
        ax.imshow(im,extent=(-28,228,128,-128));ax.set_xlim(35,170);ax.set_ylim(120,-70)
        for mode,color,ls in [('bbox','#b96a00','--'),('boundary','#006f95','-')]:
            r=next(r for r in d['records'] if r['name']=='view' and r['quality']==q and r['mode']==mode)
            p=r['parameters']['mean'];draw_envelope(ax,p[0],p[3],p[4],f,color,ls,mode+' posterior mean')
        draw_envelope(ax,d['theta'][0],d['theta'][3],d['rounding'],f,'#c62828',':','Truth (evaluation only)')
        ax.set_title(f'{f}, {q}, case {case}');ax.set_xlabel('x [mm]');ax.set_ylabel('y [mm]')
axs[0,0].legend(fontsize=7,loc='lower left');fig.suptitle('Raw RGB and sphere-center collision envelopes (radius = 20 mm)');fig.tight_layout();fig.savefig(out/'raw_image_geometry.png',dpi=180);plt.close(fig)
