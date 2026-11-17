"""Source-only sensitivity to grid resolution and a possible left/right tilt cancellation."""
from replay import read, state, HERE, K, R
import json
import numpy as np

def candidates(x,y,mu,degs,align):
    # Align both incoming and outgoing states to incoming COM travel direction.
    sgn=np.where(x[:,0]<0,-1.,1.) if align else np.ones(len(x))
    xm=x*np.column_stack([sgn,np.ones(len(x)),sgn]); ym=y*np.column_stack([sgn,np.ones(len(x)),sgn])
    best=(np.inf,); counts=0
    for deg in degs:
        a=np.deg2rad(deg); co,si=np.cos(a),np.sin(a)
        t=co*xm[:,0]+si*xm[:,1]; n=-si*xm[:,0]+co*xm[:,1]
        if n.max()>=0:continue
        counts+=1
        no=-si*ym[:,0]+co*ym[:,1]
        e=np.clip(-(n@no)/(n@n),0,1)
        cap=mu[:,None]*(1+e)*(-n)
        jt=np.clip((xm[:,2]-t)/(1+1/K),-cap,cap)
        risk=np.mean((co*(t+jt)+si*e*n-ym[:,0])**2+(xm[:,2]-jt/K-ym[:,2])**2,1)/2
        i=np.argmin(risk)
        best=min(best,(risk[i],abs(deg),deg,mu[i],e))
    return best,counts

def main():
    from replay import rigid
    out={}; base=json.loads((HERE/'metrics.json').read_text())['folds']
    mu=np.r_[np.arange(41)*.025,1.1,1.25,1.5,2.]
    for name,p in base.items():
        src=sum([read('train',c) for c in p['source']],[]); tgt=sum([read('test',c) for c in p['held']],[])
        x,y=state(src,'in'),state(src,'out'); xx,yy=state(tgt,'in'),state(tgt,'out')
        methods={'signed_original_grid':(range(-35,36),False),'signed_fine':(np.arange(-20,21)/10,False),
                 'incident_aligned':(np.unique(np.r_[np.arange(-35,36),np.arange(-20,21)/10]),True)}
        out[name]={}
        for method,(degs,align) in methods.items():
            best,count=candidates(x,y,mu,degs,align)
            _,_,deg,fr,e=best
            sgn=np.where(xx[:,0]<0,-1.,1.) if align else np.ones(len(xx))
            transform=np.column_stack([sgn,np.ones(len(xx)),sgn])
            pred=rigid(xx*transform,dict(deg=deg,mu=fr,e=e))*transform
            delta=pred-yy
            out[name][method]=dict(deg=float(deg),mu=float(fr),e=float(e),n_eligible=count,n_all=len(degs),
                        velocity_mae=np.linalg.norm(delta[:,:2],axis=1).mean(),spin_mae=abs(delta[:,2]).mean()/R)
    (HERE/'orientation.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
