"""Paired summaries and crossed intervals, calculated on the allocated compute node."""
import json
from pathlib import Path
import numpy as np
from run23 import PAIRS, R, read, write, require, sha
from analyze_v21 import bootstrap

def ci(x,alpha=.05):
    return bootstrap(x,alpha)[0]
def summary():
    require()
    for name in ("sources","plan","choices_freeze"):
        j=read(R/f"{name}.json")
        for path,h in j.get("files",j).items():assert sha(path)==h,path
    cfg=read(R/"design.json")
    acts=np.stack([np.load(R/f"choices_{b}.npz")["acts"] for b in range(4)])-1
    score=np.stack([np.load(R/f"choices_{b}.npz")["score"] for b in range(4)])
    loss=np.stack([np.load(R/f"evaluation_{b}.npz")["loss"] for b in range(4)])
    times=np.stack([np.load(R/f"evaluation_{b}.npz")["times"] for b in range(4)])
    private=np.load(R/"test/private.npz"); family=private["family"].astype(int)
    x=np.load(R/"test/public.npz")["x"]
    assert acts.shape==(4,512,2,8) and loss.shape==(4,512,2,3,32)
    assert np.isfinite(loss).all() and np.isfinite(times).all() and np.isfinite(score).all()
    risk=loss.mean(-1);selected=np.take_along_axis(risk,acts,axis=-1)
    gated=score>=cfg["cutoff"]
    assert np.array_equal(acts[:,:,:,0],np.where(gated[:,:,None],acts[:,:,:,4],acts[:,:,:,3]))
    delta=selected[...,0,None]-selected[...,1:3]
    primary,world=bootstrap(delta[:,:,0],.025)
    hi=loss[...,:16].mean(-1);lo=loss[...,16:].mean(-1)
    def ordered(source,target):
        return np.take_along_axis(target,source.argsort(axis=-1),axis=-1)
    ordered_risk=(ordered(hi,lo)+ordered(lo,hi))*.5
    gap=ordered_risk[...,1]-ordered_risk[...,0]
    worst=ordered_risk[...,2]-ordered_risk[...,0]
    split_consistency=(hi.argmin(-1)==lo.argmin(-1)).mean((0,1))
    def for_law(m):
        return np.broadcast_to(m[None,:,None,None],(4,512,2,1))
    family_act=np.where(family==0,0,2)
    family_chosen=np.take_along_axis(risk,for_law(family_act),axis=-1)[...,0]
    family_wrong=np.take_along_axis(risk,for_law(2-family_act),axis=-1)[...,0]
    fy=np.linspace(-.01,.01,9)
    survey=(x[:,:9]@fy/(fy@fy))
    simple=np.where(survey<0,0,2)
    cheap=np.take_along_axis(risk,for_law(simple),axis=-1)[...,0]
    rows={}
    for i,n in enumerate(("precise","degraded")):
        a=acts[:,:,i];sel=selected[:,:,i];tt=times[:,i].reshape(-1,3)
        subgroup={}
        for key,mask in (("geometry",np.broadcast_to(family==0,gated.shape)),("friction",np.broadcast_to(family==1,gated.shape)),("gate_on",gated),("gate_off",~gated)):
            subgroup[key]=dict(n=int(mask.sum()),gate=float(gated[mask].mean()),
                               mean_loss=sel[mask].mean(0).tolist(),gap=float(gap[:,:,i][mask].mean()),
                               flips=[float((a[:,:,0]!=a[:,:,j])[mask].mean()) for j in (1,2,3,4)])
        # A no-routing control with the same firing rate within each bank. Expected risk over random route assignments.
        rate=gated.mean(1)[:,None]
        random_route=rate*sel[:,:,4]+(1-rate)*sel[:,:,3]
        secondary=np.stack([sel[:,:,0]-sel[:,:,3],sel[:,:,0]-sel[:,:,4],sel[:,:,0]-random_route,
                            sel[:,:,6]-sel[:,:,5],sel[:,:,0]-cheap[:,:,i]],-1)
        # How much cost do actual disagreements carry? Descriptive conditional differences.
        flips=[]
        for j in (1,2):
            mask=a[:,:,0]!=a[:,:,j]
            flips.append(dict(n=int(mask.sum()),worlds=int(mask.any(0).sum()),rate=float(mask.mean()),
                              mean_delta=float((sel[:,:,0]-sel[:,:,j])[mask].mean()) if mask.any() else None))
        rows[n]=dict(mean_loss=sel.mean((0,1)).tolist(),cost_by_action=risk[:,:,i].mean((0,1)).tolist(),
                     actfreq=np.stack([(a==v).mean((0,1)) for v in range(3)],axis=-1).tolist(),timing=np.quantile(tt,[.5,.9,.95],axis=0).tolist(),
                     family_map=family_chosen[:,:,i].mean(),reverse_map=family_wrong[:,:,i].mean(),cheap=cheap[:,:,i].mean(),
                     map_ci=ci((family_wrong[:,:,i]-family_chosen[:,:,i])[:,:,None])[0],
                     gap=ci(np.stack([gap[:,:,i],worst[:,:,i]],-1)),split_agreement=split_consistency[i],
                     privileged_floor=ordered_risk[:,:,i,0].mean(),subgroup=subgroup,
                     secondary=ci(secondary),flips=flips,gate_seed=gated.mean(1).tolist(),
                     random_route=random_route.mean(),perbank=delta[:,:,i].mean(1).tolist())
    diff=ci(np.stack([gap[:,:,0]-gap[:,:,1],worst[:,:,0]-worst[:,:,1]],-1))
    out=dict(primary=primary,world_ci=world,degraded_ci=ci(delta[:,:,1]),rows=rows,interaction=diff,
             family_n=np.bincount(family,minlength=2).tolist(),label_rate=[float(private["label"][family==i].mean()) for i in (0,1)],
             seed_pairs=PAIRS,gate_rate=float(gated.mean()),cheap_family_acc=float(np.mean(simple==family_act)))
    write(R/"summary.json",out)
    print("Gap and increase",rows["precise"]["gap"][0],diff[0],flush=True)
    print("Paired contrasts",primary,rows["precise"]["timing"][1],flush=True)
if __name__=="__main__":summary()
