import numpy as np
from policy22 import PAIRS
from analyze_v21 import bootstrap
from common import require,read,write,R,sha
from data_v21 import LAWS

def average(arr,mask):
    return float(arr[mask].mean()) if mask.any() else None

def counts_diag(d):
    before,after,atom=d[...,0],d[...,1],d[...,2]
    hit=(before<=.05)&(after>=.10)&(after>=5*before)
    q=d[...,6]; e1=np.abs(d[...,7]-q);e2=np.abs(d[...,8]-q)
    def summary(mask):
        return dict(n=int(mask.sum()),prior=average(before,mask),post=average(after,mask),
                    p_top_err=average(e1,mask),p_keep_err=average(e2,mask),
                    separation=average(np.abs(d[...,10]),mask),retained_fraction=average(d[...,11]/np.maximum(after,1e-20),mask),
                    abs_change_ge_05=average((e1>=.05).astype(float),mask))
    return dict(rate=float(hit.mean()),atom=float(atom.mean()),max_mass=float(after.max()),
                high=summary(hit),rest=summary(~hit))

def run():
    require()
    for path in ("sources","plan","choices_freeze"):
        obj=read(R/(path+".json"))
        for name,h in (obj if path=="sources" else obj["files"]).items():assert sha(name)==h,name
    names=read(R/"methods.json"); cfg=read(R/"design.json")
    choices=[np.load(R/f"choices_{i}.npz") for i in range(len(PAIRS))]
    evals=[np.load(R/f"evaluation_{i}.npz") for i in range(len(PAIRS))]
    acts=np.stack([p["acts"] for p in choices])-1
    risk=np.stack([p["risk"] for p in choices])
    full_loss=np.stack([p["loss"] for p in evals])
    details=np.stack([p["diagnostics"] for p in evals])
    times=np.stack([p["times"] for p in evals])
    assert np.isfinite(details).all() and np.isfinite(full_loss).all()
    loss=np.take_along_axis(full_loss.mean(-1),acts,axis=3)
    held=np.take_along_axis(full_loss[...,8:].mean(-1),acts,axis=3)
    base=np.load(R/"test/private.npz")["cls"]!=2
    for l in (1,2):
        assert np.array_equal(acts[:,base,0],acts[:,base,l])
        assert np.array_equal(full_loss[:,base,0],full_loss[:,base,l])
    pri=np.stack([loss[...,4]-loss[...,j] for j in (3,5)],-1)
    primary,world=bootstrap(np.concatenate([pri[:,:,0],pri[:,:,1:].mean(2)],axis=-1),.0125)
    mechan=np.stack([loss[...,0]-loss[...,2],loss[...,1]-loss[...,2],loss[...,1]-loss[...,0]],-1)
    secondary,_=bootstrap(mechan.reshape(5,512,-1),.05)
    when={};when_seed={};strata={};examples=[];measures={}
    for l,law in enumerate(LAWS):
        for g in (0,1):
            d=details[:,:,l,g]
            hit=(d[...,0]<=.05)&(d[...,1]>=.1)&(d[...,1]>=5*d[...,0])
            key=f"{law}/{g}"
            when[key]={}
            for a in range(3):
                da=d[:,:,a];flag=hit[:,:,a]
                entry=counts_diag(da)
                surprise=da[...,9]
                cuts=np.quantile(surprise,[.25,.75]);low=surprise<=cuts[0];high=surprise>=cuts[1]
                entry.update(low_surprise_rate=average(flag,low),high_surprise_rate=average(flag,high),
                             flex_rate=float(flag[:,~base].mean()),base_rate=float(flag[:,base].mean()))
                when[key][str(a+1)]=entry
            when_seed[key]=hit.mean((1,2,3)).tolist()
            # Split by any actual recovery on diagnostic repeats, then evaluate other noise repeats.
            worldhit=hit[...,:8].any(axis=(2,3))
            strata[key]={}
            for label,mask in (("recovery",worldhit),("other",~worldhit)):
                ti,ki=(0,1) if g==0 else (3,4)
                entry=dict(n=int(mask.sum()),top_keep_flip=average(acts[:,:,l,ti]!=acts[:,:,l,ki],mask),
                           keep_minus_top=average(held[:,:,l,ki]-held[:,:,l,ti],mask),
                           keep_minus_fixed=average(held[:,:,l,ki]-held[:,:,l,5],mask))
                if g==0:
                    entry.update(top_full_flip=average(acts[:,:,l,0]!=acts[:,:,l,2],mask),
                                 keep_full_flip=average(acts[:,:,l,1]!=acts[:,:,l,2],mask),
                                 top_minus_full=average(held[:,:,l,0]-held[:,:,l,2],mask))
                strata[key][label]=entry
            if g==0:
                chosen=d[np.arange(5)[:,None],np.arange(512)[None,:],acts[:,:,l,0]]
                chosen_keep=d[np.arange(5)[:,None],np.arange(512)[None,:],acts[:,:,l,1]]
                measures[law]={"top_chosen":counts_diag(chosen),"keep_chosen":counts_diag(chosen_keep)}
                # Action risk error compared with same finite-bank full external evaluator.
                for j,method in [(0,"top"),(1,"keep")]:
                    dr=risk[:,:,l,j]-risk[:,:,l,2]
                    dr-=dr.mean(-1,keepdims=True)
                    measures[law][method+"_risk_rmse"]=float(np.sqrt((dr**2).mean()))
                    measures[law][method+"_full_flip"]=float((acts[:,:,l,j]!=acts[:,:,l,2]).mean())
                if hit.any():
                    index=np.unravel_index(np.argmax(np.where(hit,d[...,1],-1)),hit.shape)
                    b,i,a,n=index
                    examples.append(dict(law=law,bank=int(b),world=int(i),action=int(a+1),repeat=int(n),
                                         metrics=d[index].tolist(),choices=(acts[b,i,l,:3]+1).tolist(),
                                         losses=loss[b,i,l,:3].tolist()))
    time_rows={}
    for key,ids in (("nominal",[0]),("unseen",[1,2]),("speed_threshold",[1]),("directional_damping",[2])):
        v=times.take(ids,axis=2).reshape(-1,4)
        time_rows[key]={n:[float(np.median(v[:,j])),float(np.quantile(v[:,j],.9))] for j,n in enumerate(names[3:])}
    write(R/"summary.json",dict(names=names,loss=loss.mean((0,1)).tolist(),primary=primary,world_ci=world,
        secondary=secondary,seed_deltas=pri.mean(1).tolist(),timing=time_rows,base_n=int(base.sum()),
        when=when,perseed_rates=when_seed,strata=strata,examples=examples,mechanism=measures,
        action_counts=[[np.bincount(acts[:,:,l,j].flatten()+1,minlength=4)[1:].tolist() for j in range(len(names))] for l in range(3)]))
    print("Paired bootstrap and recovery diagnostics validated",flush=True)
if __name__=="__main__":run()
