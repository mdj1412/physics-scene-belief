"""Score fixed policies on matched trials. Input state order: null, negative, positive."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parent
METHODS = ["original", "fixed", "preload", "none", "always_project", "preload_state_shuffled", "fixed_state_shuffled", "unsigned_preload", "preload_reversed", "fixed_reversed"]
WEIGHT = np.array([.5, .25, .25])


def interval(x, count=1):
    x=np.asarray(x,dtype=float)
    mean=x.mean(0)
    half=t.ppf(1-.025/count,len(x)-1)*x.std(0,ddof=1)/np.sqrt(len(x))
    return np.stack([mean,mean-half,mean+half],axis=-1)


def evaluate(panel):
    tag=f"panel_{panel:02d}.npz"
    with np.load(ROOT/"forecast"/tag) as raw,np.load(ROOT/"labels"/tag) as truth:
        o,p=raw["p_original"],raw["p_projected"]
        assert o.shape==p.shape==(4,2,2,2,4,128,4)
        assert all(np.isfinite(prob).all() and prob.min()>=-1e-10 and prob.max()<=1+1e-10 for prob in (o,p))
        y=truth["labels"]
        assert y.shape==(4,2,3,2,3,4,128) and np.isin(y,[0,1]).all()
        paid=np.isfinite(raw["query_paid"]).all(-1)
        assert np.all(paid==np.isfinite(raw["query_paid"]).any(-1))
        assert np.max(abs((p-o)[~paid]),initial=0)<1e-10
        # F,A,H,V,S,E,O,N. Average label over the matched R,X conditional on this object.
        # For binary Y, E[(q-Y)^2] = q^2 + (1-2q)E[Y].
        avg_y=y.mean((3,4))[:,None,:,None,:,:,:,None]
        po,pp=o[:,:,:,:,None],p[:,:,:,:,None]
        base=po**2+(1.-2.*po)*avg_y
        delta=(pp-po)*(pp+po-2*avg_y)
        fixed,pre,none=[np.broadcast_to(raw["gate_"+k],base.shape) for k in ("fixed","preload","none")]
        assert all(np.isin(x,[0,1]).all() for x in (fixed,pre,none))
        assert (none==none[:,:,:,:,:1]).all()
        def state_shuffle(g):
            return np.broadcast_to(np.average(g,axis=4,weights=WEIGHT)[:,:,:,:,None],base.shape)
        gates=[np.zeros_like(pre),fixed,pre,none,np.ones_like(pre),state_shuffle(pre),state_shuffle(fixed),np.broadcast_to(raw["gate_unsigned"],base.shape),
               np.take(pre,[0,2,1],axis=4),np.take(fixed,[0,2,1],axis=4)]
        risks, rates=[], []
        mask=np.broadcast_to(paid[:,:,:,:,None],pre.shape)
        for g in gates:
            risk=base+g*delta
            risks.append(risk.mean(axis=(3,5,6,7))) # F,A,H,S
            rw=np.average(g,axis=4,weights=WEIGHT)
            rates.append([rw.mean(),(rw*paid).sum()/paid.sum()])
        # Independent direct binary-label score over all radii/offsets for preload arm.
        picked=np.where(pre,pp,po)
        acc=np.zeros(3)
        for r,x in np.ndindex(2,3):
            yy=y[:,:,:,r,x,:,:][:,None,:,None,:,:,:,None]
            acc+=((picked-yy)**2).mean(axis=(0,1,2,3,5,6,7))/6
        np.testing.assert_allclose(acc,risks[2].mean(axis=(0,1,2)),rtol=0,atol=2e-12)
        # Cost of changed routes relative to the two fixed no-extra / fixed-probe policies.
        turnover={}
        for name,g in [("fixed",fixed),("none",none)]:
            inc=(pre.astype(bool)&~g.astype(bool))
            dec=(~pre.astype(bool)&g.astype(bool))
            def wmean(a):
                return np.average(a.mean(axis=(0,1,2,3,5,6,7)),weights=WEIGHT)
            turnover[name]=[wmean(inc),wmean(dec),wmean(inc*delta),wmean(-(dec*delta))]
        return np.asarray(risks),np.asarray(rates),turnover


def main():
    packets=[evaluate(i) for i in range(12)]
    fine=np.stack([x[0] for x in packets])  # catalog,method,F,A,H,S
    state=fine.mean((2,3,4))
    group=np.average(fine,axis=-1,weights=WEIGHT)
    avg=group.mean((2,3,4))
    contrasts=avg[:,2,None]-avg[:,[0,1]]
    h=group.mean((2,3));f=group.mean((3,4));a=group.mean((2,4))
    rates=np.stack([x[1] for x in packets])
    turnover={k:np.mean([x[2][k] for x in packets],axis=0) for k in ("fixed","none")}
    for j,k in [(1,"fixed"),(3,"none")]:
        np.testing.assert_allclose(sum(turnover[k][2:]),(avg[:,2]-avg[:,j]).mean(),atol=3e-14)
    out=dict(methods=METHODS, axes="cells per catalog,method,family,alignment,height,state",
             weights=WEIGHT,catalog_loss=avg,absolute=interval(avg),primary=interval(contrasts,2),
             primary_wins=(contrasts<0).sum(0),preload_minus_none=interval(avg[:,2]-avg[:,3]),
             signed_minus_unsigned=interval(avg[:,2]-avg[:,7]), fixed_minus_none=interval(avg[:,1]-avg[:,3]),new_minus_broken_pair=interval(avg[:,2]-avg[:,5]),
             fixed_minus_broken_pair=interval(avg[:,1]-avg[:,6]),by_family=f.mean(0),
             family_preload_original_fixed=interval(np.stack((f[:,2]-f[:,0],f[:,2]-f[:,1]),axis=2),8),
             by_alignment=a.mean(0),by_height=h.mean(0),height_preload_fixed=interval(h[:,2]-h[:,1],2),
             by_state=state.mean(0),state_preload_fixed=interval(state[:,2]-state[:,1],3),
             rates_all_and_paid=rates.mean(0),turnover=turnover)
    path=ROOT/"score48.json"
    path.write_text(json.dumps(out,indent=1,default=lambda x:np.asarray(x).tolist(),allow_nan=False)+"\n")
    print("Brier",avg.mean(0),"paired primary",out["primary"],flush=True)


if __name__=="__main__":
    main()
