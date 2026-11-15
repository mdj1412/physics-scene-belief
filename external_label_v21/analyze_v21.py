from pathlib import Path
import numpy as np
from v21_policy import PAIRS,CELLS
from data_v21 import LAWS
from common import R,read,write,save,require,sha


def bootstrap(contrasts,alpha):
    """B x W x J. World and bank resampling shared over contrasts."""
    b,w,j=contrasts.shape
    rng=np.random.default_rng(917247)
    boot=[];conditional=[]
    for _ in range(200):
        worlds=rng.integers(w,size=(100,w))
        weights=rng.multinomial(b,np.ones(b)/b,size=100)/b
        avg=contrasts[:,worlds,:].mean(2)
        boot.extend(np.einsum("tb,btj->tj",weights,avg))
        conditional.extend(avg.mean(0))
    point=contrasts.mean((0,1))
    def make(values):
        limits=np.quantile(values,[alpha/2,1-alpha/2],axis=0)
        return [[float(point[i]),float(limits[0,i]),float(limits[1,i])] for i in range(j)]
    return make(boot),make(conditional)


def main():
    require()
    for path in ["sources","test_freeze","design_freeze"]:
        obj=read(R/(path+".json"))
        for file,digest in (obj if path=="sources" else obj["files"]).items():assert sha(file)==digest,file
    names=read(R/"methods.json");timed=read(R/"timed_names.json");selected=read(R/"design.json")["selected"]
    policies=[np.load(R/f"policy_{b}.npz") for b in range(len(PAIRS))]
    act=np.stack([p["acts"] for p in policies]);om=np.stack([p["omitted"] for p in policies])
    evals=[np.load(R/f"eval_{b}.npz") for b in range(len(PAIRS))]
    scores=np.stack([v["loss"] for v in evals])
    assert act.shape[:3]==(5,512,3) and np.isin(act,[1,2,3]).all()
    loss=np.take_along_axis(scores,act-1,axis=-1)
    data=np.load(R/"test/private.npz")
    base=data["cls"]!=2
    flex=~base
    assert all(np.array_equal(data["label"][base,0],data["label"][base,k]) and
               np.array_equal(act[:,base,0],act[:,base,k]) and
               np.array_equal(scores[:,base,0],scores[:,base,k]) for k in (1,2))
    times=np.stack([v["times"] for v in evals])
    source_e=loss[:,:,:, [i for i,n in enumerate(CELLS) if n.split("-")[1]=="E"]].mean(-1)
    source_c=loss[:,:,:, [i for i,n in enumerate(CELLS) if n.split("-")[1]=="C"]].mean(-1)
    idx=names.index
    delta=loss[:,:,:,idx(selected)]-loss[:,:,:,idx("fixed")]
    effect=source_e-source_c
    def pools(x):return [x[:,:,0],x[:,:,1:].mean(2)]
    primary,world_ci=bootstrap(np.stack(pools(effect)+pools(delta),axis=-1),.0125)
    control=selected.split("m")[0]+"m512"
    mechanism=loss[:,:,:,idx(control)]-loss[:,:,:,idx("E-E-mean")]
    vscompact=loss[:,:,:,idx(control)]-loss[:,:,:,idx("E-C-mean")]
    secondary,_=bootstrap(np.concatenate([effect,delta,mechanism,vscompact],axis=2),.05)
    perbank=np.stack([effect.mean(1),delta.mean(1)],axis=-1)
    leave_one=[np.delete(perbank,i,axis=0).mean(0).tolist() for i in range(len(PAIRS))]
    pooled={}
    for title,laws in [("nominal",[0]),("unseen",[1,2]),("speed_threshold",[1]),("directional_damping",[2])]:
        # Explicit take keeps all 5 banks and 16 worlds, even when pooling 2 conditions.
        tt=times.take(laws,axis=2).reshape(-1,len(timed))
        pooled[title]={n:dict(p50=float(np.median(tt[:,j])),p90=float(np.quantile(tt[:,j],.9))) for j,n in enumerate(timed)}
    seedtimes=np.quantile(times.transpose(0,2,1,3,4).reshape(5,3,-1,len(timed)),.9,axis=2)
    result=dict(methods=names,laws=LAWS,seedpairs=PAIRS,loss=loss.mean((0,1)).tolist(),source_e=source_e.mean((0,1)).tolist(),source_c=source_c.mean((0,1)).tolist(),
                primary=primary,world_only=world_ci,secondary=secondary,seed_effect=perbank.tolist(),leave_one=leave_one,
                timing=pooled,seed_timing=seedtimes.tolist(),timed=timed,seed_losses=loss.mean(1).tolist(),
                omitted_quantiles=np.quantile(om,[.5,.9,1],axis=(0,1)).tolist(),
                flex_n=int(flex.sum()),flex_rates=data["label"][flex].mean(0).tolist(),
                rates=data["label"].mean(0).tolist(),
                flex_effect=effect[:,flex].mean((0,1)).tolist(),flex_delta=delta[:,flex].mean((0,1)).tolist(),
                switch_counts=(act[:,:,:,idx(selected)]!=act[:,:,:,idx("fixed")]).sum((0,1)).tolist())
    write(R/"summary.json",result);save(R/"paired_losses.npz",loss=loss)
    print("Checked controls and sources; analyzed paired world x bank contrasts",flush=True)
if __name__=="__main__":main()
