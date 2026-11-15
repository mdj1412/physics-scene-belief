import numpy as np,json
from scipy.optimize import brentq
from direct_head34 import FAMILIES,SCHEDULES,cue_est
from decisions34 import ROOT,OUT,dump,sample,band
from target34 import calibration
read=lambda n: json.loads(n.read_text())
def apply(h,z,b):
    w=np.exp(np.clip(b*z,-30,30))
    return h*w/(1-h+h*w)
def main():
    lk=read(ROOT/'diag_models.json'); panels=[]
    for ext in (False,True):
        pp=[]
        for f,fam in enumerate(FAMILIES):
            H=[];Z=[];Y=[]
            for i in range(8):
                for d,ss in enumerate(('plus','minus')):
                    law=fam+'_'+ss
                    for a,sch in enumerate(SCHEDULES[:1 if ext else 2]):
                        p=(ROOT.parent/'directional_prefix_v33/results_eval'/law/f'block_{i:03d}') if ext else (OUT/law/f'block_{i:03d}'/sch)
                        with np.load(p/'metrics.npz') as m,np.load(p/'choices.npz') as z:
                            ac=(m['actions'][:,:,2 if ext else 1]==3).astype(int)
                            h=np.take_along_axis(m['probabilities'],ac[:,:,None,None],axis=2)[:,:,0]
                            H.append(h);Y.append(m['labels'][:,:,None])
                            if ext:
                                rs=[calibration(510000000+f*1000000+10000*i+100+e,al,law,'same',-1) for e,al in enumerate(m['alpha'])]
                                x=np.array([r[0][r[1]][:4] for r in rs]);y=np.array([r[2][:4] for r in rs]);di=np.array([r[3][:4] for r in rs])
                            else:x,y,di=[z[k][:,:4] for k in ('paid_x','paid_y','paid_direction')]
                            Z.append([cue_est(lk,ff,x,y,di,sch) for ff in (f,3-f)])
            pp.append((np.stack(H),np.stack(Y),np.stack(Z)))
        panels.append(pp)
    pars=[];final=[]
    for f in range(4):
        h,y,z=panels[0][3-f]
        co=[]
        for mode in (0,1):
            sign=np.broadcast_to(np.repeat(np.tile([1,-1],8),2)[:,None],(32,8)) if mode else z[:,1]
            t=sign[:,:,None,None]
            deriv=lambda b:float(((apply(h,t,b)-y)*t).mean()+.005*b)
            co.append(brentq(deriv,-20.,20.,xtol=1e-12))
        pars.append(co)
    for e,pp in enumerate(panels):
        A=1 if e else 2;v=np.empty((4,8,2,A,3))
        for f,(h,y,z) in enumerate(pp):
            sign=np.broadcast_to(np.repeat(np.tile([1,-1],8),A)[:,None],(16*A,8))
            for k,pr in enumerate((h,apply(h,z[:,0,:,None,None],pars[f][0]),apply(h,sign[:,:,None,None],pars[f][1]))):
                v[f,...,k]=((pr-y)**2).mean((1,2,3)).reshape(8,2,A)
        rng=np.random.default_rng(341115+e);idx=rng.integers(8,size=(12000,4,8)) if e else sample(rng,12000)
        point=v.mean((0,1,2));d=point[:,1:]-point[:,:1]; boot=v[np.arange(4)[None,:,None],idx].mean((1,2,3))
        final.append(dict(means=point,delta=d,ci=band(d.ravel(),(boot[:,:,1:]-boot[:,:,:1]).reshape(len(boot),-1)).reshape(A,2,2)))
    dump('anchored34.json',dict(beta=pars,methods=['original','estimated','true'],v34=final[0],v33_external=final[1],exploratory=True))
if __name__=='__main__':main()
