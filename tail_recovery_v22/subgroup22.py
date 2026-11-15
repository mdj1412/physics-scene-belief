"""Exploratory uncertainty and threshold sensitivity, from frozen held-out measurements."""
from stats22 import R,read,write,require,np,PAIRS,LAWS

def conditional_ci(delta,mask):
    counts=mask.sum()
    if not counts:return None
    rng=np.random.default_rng(617991)
    b,w=mask.shape
    means=[]
    for _ in range(100):
        world=rng.integers(w,size=(100,w))
        weights=rng.multinomial(b,np.ones(b)/b,size=100)
        denominator=np.einsum("ab,baw->a",weights,mask[:,world],optimize=True)
        numerator=np.einsum("ab,baw->a",weights,(delta*mask)[:,world],optimize=True)
        means.extend(numerator[denominator>0]/denominator[denominator>0])
    return [float(delta[mask].mean()),*np.quantile(means,[.025,.975]).tolist(),len(means)]

def run():
    require()
    c=np.stack([np.load(R/f"choices_{b}.npz")["acts"] for b in range(5)])-1
    risks=np.stack([np.load(R/f"choices_{b}.npz")["risk"] for b in range(5)])
    out={}
    files=[np.load(R/f"evaluation_{b}.npz") for b in range(5)]
    det=np.stack([v["diagnostics"] for v in files])
    actual=np.stack([v["loss"] for v in files])
    gains=np.take_along_axis(actual[...,8:].mean(-1),c,axis=3)
    for l,law in enumerate(LAWS):
        rc=risks[:,:,l,2];ordered=np.sort(rc,axis=-1)
        gap=ordered[:,:,1]-ordered[:,:,0]
        out[law]={}
        for k in [0,1]:
            d=det[:,:,l,k]
            b,p=d[...,0],d[...,1]
            hit=(b<=.05)&(p>=.10)&(p>=5*b)
            row={}
            row["pooled_rate"]=float(hit.mean())
            row["rates_alt"]=[float(((b<=.05)&(p>=mass)&(p>=fold*b)).mean()) for mass,fold in [(0.05,2),(0.2,10)]]
            h=hit[...,:8].any((2,3))
            top,keep,ref=(0,1,2) if k==0 else (3,4,5)
            dh=gains[:,:,l,keep]-gains[:,:,l,top]
            row["recovery_ci"]=conditional_ci(dh,h)
            row["other_ci"]=conditional_ci(dh,~h)
            row["unique_worlds"]=int(h.any(0).sum())
            if k==0:
                for m in (0,1):
                    flip=c[:,:,l,m]!=c[:,:,l,2]
                    mdiff=c[:,:,l,0]!=c[:,:,l,1]
                    row["switched_actual_delta" + str(m)]=conditional_ci(gains[:,:,l,m]-gains[:,:,l,2],flip)
                    row["flip_counts"+str(m)]=int(flip.sum())
                    row["median_full_gap_flip"+str(m)]=float(np.median(gap[flip])) if flip.any() else None
                    row["median_full_gap_same"+str(m)]=float(np.median(gap[~flip])) if (~flip).any() else None
                row["disagreement_ci"]=conditional_ci(dh,mdiff)
            out[law][str(k)]=row
    write(R/"subgroups.json",out)
    print("Recovery stratification and crossed ratio CIs complete.")
if __name__=="__main__":run()
