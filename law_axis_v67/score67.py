"""Release new label laws only after the forecast seal, score crossed and nominal effects."""
import hashlib
import itertools
import json
from pathlib import Path
import sys
import numpy as np
from scipy.stats import t
H=Path(__file__).resolve().parent
P=H.parent
SRC=P/'invariance_audit_v66'/'locked'
sys.path[:0]=[str(P/'image_pressure_v45'),str(SRC)]
import pressure45 as phys
from locked_geometry import profile
FAM=('wendland','dispersive')
SURF=('inlay','woven')
OLD=('rational','gaussian_rate')
NEW=('strengthening','weakening')
VAR=('base','hollow','bulge')
HT=(19,31)
SEEDS=tuple(range(670101,670105))
LAWS=OLD+NEW
ORIG=phys.material
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def name(f,s,l,v,h):return f'{f}_{s}_{l}_{v}_h{h}.npz'
def panel(k):return f'panel_{k:02d}'
def law(mu,vt,h,sf,label,signs,reg):
    if label not in NEW:return ORIG(mu,vt,h,sf,label,signs,reg)
    base=ORIG(mu,vt,h,sf,'rational',signs,reg)
    v=np.abs(vt)
    old_factor=1.045-.046*v*v/(v*v+.58**2)
    z=np.tanh((v-.40)/.19)
    return base/old_factor*(1+(.22 if label=='strengthening' else -.22)*z)
def labels():
    phys.ROOT=SRC;phys.output_folder=lambda high:H/('fine' if high else 'labels')
    phys.slope=lambda x,f,v:profile(x,f,v)[1]
    phys.material=law;phys.LAWS=NEW
    for k,f,v,h in itertools.product(range(12),FAM,VAR,HT):
        phys.run_group(k,f,v,h,False)
    for f,v,h in itertools.product(FAM,VAR,HT):phys.run_group(0,f,v,h,True)
def ci(x,m=1):
    n=x.shape[0];a=x.mean(0);w=x.std(0,ddof=1)*t.isf(.025/m,n-1)/np.sqrt(n)
    return np.stack((a,a-w,a+w),axis=-1)
def main():
    meta=json.loads((H/'forecast_seal.json').read_text())
    for fn,hs in meta['shards'].items():assert sha(H/'forecasts'/fn)==hs
    assert meta['axes']==sha(H/'axes.json') and meta['fits']==sha(H/'fit.json') and meta['plan']==sha(H/'plan.sha256')
    labels()
    # risk shape catalog x family x terminal law x axis x model
    a=np.zeros((12,2,4,4,5));match=a.copy()
    flip=np.zeros((12,2,4));law_diff=np.zeros((12,2));cross=np.zeros((12,2,4))
    fine_correct=fine_n=0;fine_max=0.
    for k in range(12):
        pan=panel(k)
        with np.load(H/'forecasts'/(pan+'.npz')) as z:
            preds=z['p'];names=list(z['cells'])
        for fi,f in enumerate(FAM):
            for s,v,h in itertools.product(SURF,VAR,HT):
                with np.load(SRC/'routes'/pan/f'{f}_{v}_h{h}.npz') as rt:route=rt['route']
                ys=[]
                for li,l in enumerate(LAWS):
                    fol=SRC if l in OLD else H
                    fn=name(f,s,l,v,h)
                    with np.load(fol/'labels'/pan/fn) as z:y=z['labels'][1:].astype(np.float64)
                    ys.append(y)
                    if v=='hollow':
                        with np.load(fol/'labels'/pan/name(f,s,l,'bulge',h)) as z:
                            flip[k,fi,li]+=np.mean(z['labels'][1:]!=y)/4
                    if k==0 and l in NEW:
                        with np.load(H/'fine'/pan/fn) as z,np.load(H/'labels'/pan/fn) as zz:
                            fine_correct+=int((z['labels'][1:]==zz['labels'][1:]).sum())
                            fine_n+=y.size
                            fine_max=max(fine_max,float(abs(z['slip_speed'][1:]-zz['slip_speed'][1:]).max()))
                law_diff[k,fi]+=np.mean(ys[2]!=ys[3])/12
                for ai,acq in enumerate(OLD):
                    row=preds[names.index(name(f,s,acq,v,h))].astype(np.float64) # seed model e n repeat
                    # For first 3 methods, route available independently for each center offset.
                    # Common heads already supply same fallback as original on the probe branch.
                    mixed=np.broadcast_to(row[:,:,None],(4,5,3,*row.shape[2:])).copy()
                    mixed[:,1:3]=np.where(route[None,None,...,None],mixed[:,1:3],mixed[:,:1])
                    ymean=np.mean(ys,axis=1)[...,None] # law x offset x e x n x singleton
                    risk=(mixed[None]**2+(1-2*mixed[None])*ymean[:,None,None]).mean((3,4,5,6))
                    a[k,fi]+=risk/24
                    match[k,fi,ai]+=risk[ai]/12
                    effect=-2*(mixed[:,1]-mixed[:,0])*(ymean[2]-ymean[3])
                    cross[k,fi]+=effect.mean((1,2,3,4))/24
    soft=a[:,:,:,:,1]-a[:,:,:,:,0]
    restored=a[:,:,:,:,4]-a[:,:,:,:,3]
    diff=soft[:,:,2]-soft[:,:,3]
    np.testing.assert_allclose(diff,cross,atol=1e-14,rtol=0)
    primary={}
    for li,l in enumerate(LAWS):
        ls=dict(family_soft=ci(soft[:,:,li].mean(-1),6), pooled_rescue=ci(restored[:,:,li].mean((1,2)),6),
                family_rescue=ci(restored[:,:,li].mean(-1)), family_seed_soft=soft[:,:,li].mean(0),
                family_seed_rescue=restored[:,:,li].mean(0),pooled_seed_rescue=restored[:,:,li].mean((0,1)),
                pct_catalog_sign_W_D=[int((soft[:,0,li].mean(-1)<0).sum()),int((soft[:,1,li].mean(-1)>0).sum())])
        primary[l]=ls
    report=dict(primary=primary, methods=['full','soft075','hard','dropped','restored'],families=FAM,laws=LAWS,axis_seeds=SEEDS,
      absolute_family_law_methods=a.mean((0,3)),match_old_family_method_mean=match[:,:,:2].mean((0,2,3)),
      label_flip_variant=ci(flip),label_switch_new_laws=ci(law_diff),brier_law_identity_maxerr=float(abs(diff-cross).max()),
      family_soft_difference_strength_minus_weak=ci(diff.mean(-1),2),fine_label_agreement=fine_correct/fine_n,fine_n=fine_n,fine_max_speed_diff=fine_max,
      panels=a,archive_nominal_panels=match[:,:,:2],catalog_hollow_bulge_flip=flip,catalog_new_law_switch=law_diff,forecast_sha256=sha(H/'forecast_seal.json'))
    (H/'score.json').write_text(json.dumps(report,default=lambda x:np.asarray(x).tolist(),indent=2))
    print(json.dumps(primary,default=lambda x:np.asarray(x).tolist()),flush=True)
if __name__=='__main__':main()
