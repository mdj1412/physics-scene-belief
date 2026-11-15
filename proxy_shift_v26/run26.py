"""Prospective stress test of the frozen V25 raw-input gate."""
import json,os,time
from pathlib import Path
from itertools import product
import numpy as np
from physics26 import Engine, LEVELS, SOURCE, generate, variants, t, torch
from gate25 import Predictor
from common import R,save,write,read,require,freeze,sha
N=1024
BANKS=[(2634001+b*13003,2645101+b*17011) for b in range(5)]
CONDITIONS=list(product(LEVELS,repeat=2))
METHODS=['gate','k256','direct','k1024','query']
def frozen():
    cfg=read(SOURCE/'results/selected.json')
    return Predictor(cfg['model']),float(cfg['cut'])

def timed(engine,pred,cut,x_host,yQ_host,yP_host,ids,expected,reps,seed):
    rng=np.random.default_rng(seed);out=np.empty((len(ids),reps,2))
    for rep in range(-1,reps):
        order=np.arange(min(2,len(ids))) if rep<0 else rng.permutation(len(ids))
        for row in order:
            i=ids[row];x=x_host[i];q,p=yQ_host[i],yP_host[i]
            for method in rng.permutation(2):
                torch.cuda.synchronize()
                start=time.perf_counter()
                if method==0:
                    score=float(pred.score(x));conf=abs(score)>=cut
                else:conf=False
                tx=t(x)
                if conf:
                    a=1 if score>=0 else 3;w,z=engine.inner.fit(tx),engine.inner.label
                else:
                    st=engine.fit(tx);w,z=st[:2];a=engine.choose_pair(st,1024 if method==0 else 256)
                probability=engine.inner.update(w,a,t(q if a==1 else p))@z[:,0]/np.sqrt(2.)
                torch.cuda.synchronize()
                ms=(time.perf_counter()-start)*1000
                assert a==expected[i,method] and torch.isfinite(probability).item()
                if rep>=0:out[row,rep,method]=ms
    return out

def main():
    require();torch.set_num_threads(2);assert torch.cuda.is_available()
    sources=[Path(f) for f in ('physics26.py','run26.py','PROTOCOL.md')]
    base=Path('../within_calibration_v25')
    # Hash the actual v25 source and deployment parameters and enforce equality with original manifest.
    old=read(base/'results/sources.json')
    for file,h in old.items():
        p=base/file
        assert sha(p)==h,p
        sources.append(p)
    previous=read(base/'results/plan.json')['files']
    assert sha(base/'results/selected.json')==previous['results/selected.json']
    sources.extend([base/'results/selected.json',base/'results/plan.json'])
    write(R/'sources.json',{str(p):sha(p) for p in sources})
    write(R/'device.json',dict(host=os.environ['SLURMD_NODENAME'],gpu=torch.cuda.get_device_name(),job=os.environ['SLURM_JOB_ID']))
    write(R/'design.json',dict(conditions=CONDITIONS,levels=LEVELS,worlds=N,banks=BANKS,methods=METHODS,timed=METHODS[:2],seed=2613001,cut=frozen()[1]))
    freeze(R/'plan.json',[R/'sources.json',R/'design.json'])
    generate('test',N,2613001)
    xs=np.load(R/'test/public.npz')['x'];assign=np.load(R/'test/assignment.npz');cell=assign['cell']
    mapping=np.stack([variants(assign['u_flip'],c,f) for c,f in CONDITIONS])
    assert (mapping[0]==0).all() and (mapping[-1]==3).all()
    save(R/'scenarios.npz',variants=mapping)
    pred,cut=frozen()
    score=pred.score(xs.reshape(-1,11)).reshape(N,4)
    conf=np.abs(score)>=cut
    direct=np.where(score>=0,1,3)
    paths=[R/'plan.json',R/'scenarios.npz',R/'test/public.npz',R/'test/assignment.npz']
    for b,pair in enumerate(BANKS):
        eng=Engine(pair);dec=np.empty((N,4,len(METHODS)),np.int8)
        for i,v in enumerate(xs):
            for j,x in enumerate(v):
                st=eng.fit(t(x));mid=eng.choose_pair(st,256);high=eng.choose_pair(st,1024)
                dec[i,j]=[direct[i,j] if conf[i,j] else high,mid,direct[i,j],high,1]
        path=R/f'choices_{b}.npz';save(path,acts=dec,confident=conf,score=score);paths.append(path)
        print('Locked choices',b+1,flush=True)
    freeze(R/'choices_freeze.json',paths)
    private=np.load(R/'test/private.npz')
    target=t(private['label']);yy=[private[f'obs{a}'] for a in (1,3)];yd=[t(z) for z in yy]
    ids=np.sort(np.concatenate([np.flatnonzero(cell==i)[:12] for i in range(4)]));wi=np.arange(N)
    for b,pair in enumerate(BANKS):
        eng=Engine(pair);dec=np.load(R/f'choices_{b}.npz')['acts']
        loss=np.empty((N,4,2,32))
        for i,v in enumerate(xs):
            for j,x in enumerate(v):
                w=eng.inner.fit(t(x))
                for slot,a in enumerate((1,3)):
                    prob=eng.inner.update(w,a,yd[slot][i,j])@eng.inner.label[:,0]/np.sqrt(2.)
                    loss[i,j,slot]=(prob-target[i,j]).square().cpu().numpy()
        times=np.empty((len(CONDITIONS),len(ids),3,2));order=np.random.default_rng(261033+b).permutation(len(CONDITIONS))
        for c in order:
            v=mapping[c]
            times[c]=timed(eng,pred,cut,xs[wi,v],yy[0][wi,v,0],yy[1][wi,v,0],ids,dec[wi,v,:2],3,262207+113*b+13*c)
        save(R/f'eval_{b}.npz',loss=loss,times=times,ids=ids,order=order)
        print('Evaluated bank',b+1,flush=True)
if __name__=='__main__':main()
