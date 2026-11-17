from pathlib import Path
import csv,io,json,zipfile
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from static_light64 import ROOT,DATA,read_image

with (ROOT/'static_tracks64.csv').open() as f: rows=list(csv.DictReader(f))
groups={}
for r in rows:
    if r['reverse']=='0':groups.setdefault((r['object'],int(r['x']),int(r['y'])),[]).append(r)
for (name,x,y),track in groups.items():
    dx=np.array([int(r['dx']) for r in track[1:]])
    if (np.all(dx>=2) or np.all(dx<=-2)) and min(float(r['NCC']) for r in track[1:])>=.8:break
else:raise AssertionError('no illustrative track')
spec=json.loads((ROOT/'static_tracks64.json').read_text())['directions'][name]
with zipfile.ZipFile(DATA/'DiLiGenT.zip') as z:
    pre='DiLiGenT/pmsData/'+name+'/'
    intens=np.loadtxt(io.BytesIO(z.read(pre+'light_intensities.txt')))
    fn=z.read(pre+'filenames.txt').decode().split()
    patches=[read_image(z,pre+f,intens[fn.index(f)])[y-23:y+24,x-23:x+24] for f in spec['filenames']]
lo,hi=np.quantile(patches,[.01,.99])
fig,axes=plt.subplots(1,4,figsize=(9,2.8),layout='constrained')
for i,(ax,img,r) in enumerate(zip(axes,patches,track)):
    ax.imshow(img,cmap='gray',vmin=lo,vmax=hi,interpolation='nearest')
    ax.scatter(23,23,marker='+',s=100,color='#00d9ff',linewidths=1.5)
    ax.scatter(23+int(r['dx']),23+int(r['dy']),marker='o',s=102,facecolors='none',edgecolors='#ffb439',linewidths=1.7)
    ax.set_title(f"LED {int(spec['filenames'][i][:3])}   |   dx {int(r['dx']):+d} px",fontsize=10)
    ax.set_axis_off()
fig.suptitle(f'Same static patch under four lights ({name.removesuffix("PNG")})\n+  reference location      ○  NCC match',fontsize=12)
fig.savefig(ROOT/'static_patch64.png',dpi=160,bbox_inches='tight',facecolor='white')
(ROOT/'example64.json').write_text(json.dumps(dict(object=name,x=x,y=y,track=track),indent=2))
