"""Exploratory 0.025-mm scan on matched objects; verify every preregistered grid point."""
import json
from pathlib import Path
import time
import numpy as np
from stats42 import ROOT,FAMILY,HEIGHT,VARIANT,LAW,OFF,ci,raw
import runner42 as job
import physics_audit41 as legacy_audit
r=job.r
GRID=np.arange(-40,41)/40.
KEYS=[int(round(x*40))+40 for x in OFF]

def labels(theta,heights,f,var,law,sign,reg):
    # Replicate the static-friction inequality in local normal/tangent coordinates.
    h=heights[:,None]
    slope=theta[None,:,2]+theta[None,:,3]*h/.025+legacy_audit.geom(heights,f,var)[0][:,None]
    root=np.hypot(1.,slope); vt=(.425+slope)/root
    jn=(1+theta[None,:,5])*np.maximum(0.,(1-.425*slope)/root)
    abs_v=np.abs(vt)
    if law=='saturating':
        direction=.18*np.tanh(vt/.62)**3
        factor=1.015+.025*np.tanh(abs_v/.58)**2
    elif law=='crossover':
        direction=.20*np.tanh(vt/.32)*np.tanh((abs_v-.70)/.16)
        factor=1.035-.02*np.exp(-((abs_v-.68)/.24)**2)
    scale={'strong':1.,'weak':.3,'null':0.}[reg]
    envelope=1-(1-scale)*np.sin(np.minimum(abs(h)/.023,1)*np.pi/2)**2
    return abs_v<=jn*theta[None,:,4]*(1+sign*envelope*direction)*factor

def main():
    import os
    assert os.environ['BELIEF_DEVICE']=='cpu' and not r.torch.cuda.is_available()
    curves=np.zeros((12,2,2,len(GRID)))
    for n in range(12):
        all_paths=[raw((f,h,v,l),n) for f in FAMILY for h in HEIGHT for v in VARIANT for l in LAW]
        while not all(p.with_suffix('.json').exists() for p in all_paths):time.sleep(8)
        meta=json.loads(all_paths[0].with_suffix('.json').read_text())
        with np.load(all_paths[0]) as a: alpha=a['alpha']
        dummy=(FAMILY[0],VARIANT[0],.019,LAW[0],1,job.w.catalog_panel(n))
        theta=np.concatenate([r.world.draw_object(128,seed,20,aa,dummy).cpu().numpy()
                              for seed,aa in zip(meta['deployment_seeds'],alpha)])
        for i,f in enumerate(FAMILY):
            for j,hk in enumerate(HEIGHT):
                hs=r.hvalue(hk)+GRID*.001
                for v in VARIANT:
                    for law in LAW:
                        with np.load(raw((f,hk,v,law),n)) as a:
                            base=a['p_base']; stored=a['labels']
                        for si,sign in enumerate(r.SIGNS):
                            est,orig=base[2,si],base[0,si]
                            linear=(est-orig).mean(-1).reshape(-1)
                            square=(est**2-orig**2).mean()
                            for ri,reg in enumerate(r.REGIMES):
                                y=labels(theta,hs,f,v,law,sign,reg)
                                np.testing.assert_array_equal(y[KEYS],stored[:,si,ri].reshape(9,-1))
                                curves[n,i,j]+=(square-2*(y*linear).mean(1))/36
        print('dense panel',n,flush=True)
    out={'offset_mm':GRID,'curve':ci(curves,4*len(GRID)),'pooled':ci(curves.mean((1,2)),len(GRID)),
         'panel':curves}
    (ROOT/'dense42.json').write_text(json.dumps(out,indent=2,default=lambda u:u.tolist(),allow_nan=False)+'\n')
if __name__=='__main__':main()
