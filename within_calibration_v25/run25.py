import os,time
from pathlib import Path
import numpy as np
from physics25 import Engine,generate,t,torch
from gate25 import Predictor,learn,FrozenGate
from common import R,save,write,read,freeze,sha,require
from policy22 import action
from run23 import loss

NAMES=['gate','k256','query','probe','direct','k1024','tail_gate','k128','global','k256_three','v24_frozen']
TIMED=['gate','k256','query','direct','k1024','tail_gate','k128','probe','v24_frozen']
BANKS=[(25341001+b*13003,25451003+b*17011) for b in range(5)]
def inputs(split):
    return np.load(R/split/'public.npz')['x'][:,0,:]
def outcomes(split):
    with np.load(R/split/'private.npz') as z:return {k:t(z[k]) for k in z.files}
def pair_action(r):
    return 1 if float(r[0]-r[2])<=1e-12 else 3

def development():
    start=time.perf_counter()
    sets=[('train',3072,2511001,[(2514001,2514401),(2517101,2517501)]),
          ('validation',1024,2512001,[(2524001,2524401)])]
    for split,n,seed,pairs in sets:
        generate(split,n,seed)
        xs=inputs(split);data=outcomes(split)
        all_cost=[];fall=np.zeros(n,dtype=int);static=np.zeros(n,dtype=int)
        for pair in pairs:
            engine=Engine(pair);co=np.empty((n,3))
            for i,x in enumerate(xs):
                st=engine.fit(t(x))
                co[i]=loss(engine,st,data,i,0).mean(axis=-1)
                if split=='validation':
                    fall[i]=engine.choose_pair(st,1024)
                    static[i]=engine.choose_pair(st,256)
            all_cost.append(co)
        save(R/split/'development.npz',cost=np.stack(all_cost),fallback=fall,static=static)
        print(split,'ready',flush=True)
    train=inputs('train');vt=inputs('validation')
    l=np.load(R/'train/development.npz')['cost'].mean(0);y=l[:,2]-l[:,0]
    vv=np.load(R/'validation/development.npz');cost=vv['cost'][0];fallback=vv['fallback']
    results=[]
    for family in ('quadratic','rbf'):
        for penalty in (.0001,.001,.01):
            model=learn(train,y,family,penalty);pred=Predictor(model)
            tr=pred.score(train);val=pred.score(vt)
            # Re-use one fitted model at all four prespecified cutoffs.
            for q in (.2,.4,.6,.8):
                cut=float(np.quantile(np.abs(tr),q));conf=np.abs(val)>=cut
                acts=np.where(conf,np.where(val>=0,1,3),fallback)
                mean=cost[np.arange(len(vt)),acts-1].mean()
                cfg=dict(family=family,penalty=penalty,quantile=q,cut=cut,confident=float(conf.mean()),validation_loss=float(mean))
                results.append((tuple([mean,-conf.mean(),family!='quadratic',penalty,q]),cfg,model))
    _,cfg,model=min(results,key=lambda a:a[0]);write(R/'selected.json',dict(**cfg,model=model))
    c=outcomes('validation')['cell'].cpu().numpy()
    margins=cost[:,2]-cost[:,0]
    write(R/'development.json',dict(candidates=[z[1] for z in results],cell_margin=[float(margins[c==i].mean()) for i in range(4)],
              static_brier=float(cost[np.arange(len(vt)),vv['static']-1].mean()),query_brier=float(cost[:,0].mean()),
              seconds=time.perf_counter()-start))
    print("LOCK",cfg,"dev physical margins",read(R/'development.json')['cell_margin'],flush=True)

def timing(engine,gate,frozen,cut,xs,data,ids,acts):
    rng=np.random.default_rng(255003);out=np.empty((len(ids),4,len(TIMED)))
    ys=[data[f'obs{a}'][:,0,0].cpu().numpy() for a in (1,2,3)]
    for rep in range(-1,4):
        for p,i in enumerate(ids[:3] if rep==-1 else ids):
            x=xs[i];signals=[y[i] for y in ys]
            for j in rng.permutation(len(TIMED)):
                name=TIMED[j];torch.cuda.synchronize();start=time.perf_counter()
                k=None
                if name in ('gate','direct','tail_gate'):
                    value=float(gate.score(x));conf=abs(value)>=cut
                    if name=='tail_gate':k=128 if conf else 1024
                    elif name=='gate' and not conf:k=1024
                    else:a=1 if value>=0 else 3
                elif name in ('query','probe'):a=1 if name=='query' else 3
                elif name=='v24_frozen':k=1024 if frozen.high(x) else 128
                else:k=int(name[1:])
                tx=t(x)
                if k is not None:
                    state=engine.fit(tx);w,z=state[:2];a=engine.choose_pair(state,k)
                else:w,z=engine.inner.fit(tx),engine.inner.label
                probability=engine.inner.update(w,a,t(signals[a-1]))@z[:,0]/np.sqrt(2.)
                torch.cuda.synchronize()
                duration=(time.perf_counter()-start)*1000
                assert acts[i,NAMES.index(name)]==a and torch.isfinite(probability).item()
                if rep>=0:out[p,rep,j]=duration
    return out

def run():
    require();torch.set_num_threads(2);assert torch.cuda.is_available()
    write(R/'device.json',dict(job=os.environ['SLURM_JOB_ID'],host=os.environ['SLURMD_NODENAME'],gpu=torch.cuda.get_device_name()))
    base=Path('../cheap_tail_v24')
    sources=[base/p for p in read(base/'results/sources.json')]+[Path(f) for f in ('physics25.py','gate25.py','run25.py','PROTOCOL.md','../calibrated_router_v24/results/selected.json')]
    write(R/'sources.json',{str(p):sha(p) for p in sources})
    development();cfg=read(R/'selected.json');model=Predictor(cfg['model']);frozen=FrozenGate()
    write(R/'methods.json',dict(names=NAMES,timed=TIMED));freeze(R/'plan.json',[R/f for f in ['sources.json','methods.json','selected.json','development.json']])
    generate('test',1280,2513001);xs=inputs('test');paths=[R/'plan.json',R/'test/public.npz']
    signal=np.array([model.score(x) for x in xs]);conf=np.abs(signal)>=cfg['cut'];direct=np.where(signal>=0,1,3)
    transport=np.array([frozen.high(x) for x in xs])
    for b,pair in enumerate(BANKS):
        engine=Engine(pair);dec=np.empty((len(xs),len(NAMES)),int)
        for i,x in enumerate(xs):
            st=engine.fit(t(x));risk=engine.risk(st,'keep256m2048')
            mid=pair_action(risk);low=engine.choose_pair(st,128);high=engine.choose_pair(st,1024)
            dec[i]=[direct[i] if conf[i] else high,mid,1,3,direct[i],high,low if conf[i] else high,
                    low,pair_action(engine.risk(st,'fixed')),action(risk),high if transport[i] else low]
        f=R/f'choices_{b}.npz';save(f,acts=dec,confident=conf,score=signal,transported_high=transport);paths.append(f)
        print('Choices',b+1,flush=True)
    freeze(R/'choices_freeze.json',paths)
    data=outcomes('test');cells=data['cell'].cpu().numpy()
    ids=np.sort(np.concatenate([np.flatnonzero(cells==c)[:20] for c in range(4)]))
    for b,pair in enumerate(BANKS):
        eng=Engine(pair);acts=np.load(R/f'choices_{b}.npz')['acts']
        costs=np.empty((len(xs),3,32))
        for i,x in enumerate(xs):costs[i]=loss(eng,eng.fit(t(x)),data,i,0)
        times=timing(eng,model,frozen,cfg['cut'],xs,data,ids,acts)
        save(R/f'eval_{b}.npz',loss=costs,times=times,ids=ids)
        print('Eval',b+1,flush=True)
if __name__=='__main__':run()
