"""Exploratory constant effective slope; fit on the designated source only."""
from pathlib import Path
import json

import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def rotate(x, angle):
    c,s=np.cos(angle),np.sin(angle)
    return x[:,0]*c+x[:,1]*s, -x[:,0]*s+x[:,1]*c

def predict(x,e,mu,angle):
    t,n=rotate(x,angle)
    assert (n<0).all()
    cap=-n*mu*(1+e)
    jt=np.clip(-(t+x[:,2])/3.5,-cap,cap)
    vxf=(t+jt)*np.cos(angle)+e*n*np.sin(angle)
    return np.column_stack([vxf,x[:,2]+2.5*jt])

def run():
    studies=[]
    for source,target in [('A','B'),('B','A')]:
        def read(c):
            return np.loadtxt(ROOT/'data69'/'ESM_2'/f'data_campaign{c}.csv',delimiter=',')*.02135
        xs,xt=read(source),read(target)
        fit=(np.arange(len(xs))//24)%4!=1
        x,y=xs[fit,:3],xs[fit,3:]
        grid=[]
        for deg in np.arange(0,35.01,1):
            angle=np.deg2rad(deg)
            _,n0=rotate(x,angle)
            _,nf=rotate(y,angle)
            e=np.clip(-np.dot(n0,nf)/np.dot(n0,n0),0,1)
            for mu in np.r_[np.arange(0,1.001,.025),1.1,1.25,1.5,2.]:
                yp=predict(x,e,mu,angle)
                grid.append((np.mean((yp-y[:,[0,2]])**2),deg,mu,e))
        risk,deg,mu,e=min(grid)
        def pred(xx):return predict(xx[:,:3],e,mu,np.deg2rad(deg))
        err=(xs[~fit,3]+xs[~fit,5])-pred(xs[~fit]).sum(1)
        ts=pred(xt)
        u=xt[:,3]+xt[:,5]
        ui=xt[:,0]+xt[:,2]
        draws=ts.sum(1)[:,None]+err
        def brier(events, truth):
            return float(np.mean((((events.sum(1)+.5)/(len(err)+1))-truth)**2))
        studies.append(dict(source=source,target=target,slope_deg=deg,mu=mu,e=e,interior=bool(0<deg<35),
          vx_mae=float(np.mean(abs(ts[:,0]-xt[:,3]))),u_mae=float(np.mean(abs(ts.sum(1)-u))),
          brier=brier(abs(draws)<=.5,abs(u)<=.5),rev_brier=brier(draws*ui[:,None]<0,ui*u<0)))
    print(json.dumps(studies,indent=2))
    (ROOT/'analysis'/'inclined.json').write_text(json.dumps(studies,indent=2)+'\n')
if __name__=='__main__':run()
