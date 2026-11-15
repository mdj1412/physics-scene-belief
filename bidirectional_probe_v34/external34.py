"""Frozen V34 heads replayed on disjoint V33 negative-only deployment worlds, not for model selection."""
import numpy as np,json
from scipy.special import expit
from target34 import calibration
from decisions34 import ROOT,dump,band,alloc
from direct_head34 import CUES,FAMILIES,load_artifact,design,cue_est
from terminal34 import features
read=lambda p:json.loads(p.read_text())
def main():
    ar=load_artifact();tm=read(ROOT/'terminal_models34.json');v=np.zeros((4,8,2,2,3))
    for f,fam in enumerate(FAMILIES):
        for b in range(8):
            seed=510000000+1000000*f+10000*b
            for d,di in enumerate(('plus','minus')):
                law=fam+'_'+di;p=ROOT.parent/'directional_prefix_v33/results_eval'/law/f'block_{b:03d}'
                with np.load(p/'metrics.npz') as z,np.load(p/'choices.npz') as c:
                    records=[calibration(seed+100+e,al,law,'same',-1) for e,al in enumerate(z['alpha'])]
                    x=np.array([r[0][r[1]] for r in records]);y=np.array([r[2] for r in records]);sign=np.array([r[3] for r in records])
                    prob=z['probabilities'];ac=(z['actions'][:,:,2]==3).astype(int); assert (z['counts'][:,2]==4).all()
                    pr=np.take_along_axis(prob,ac[:,:,None,None],axis=2)[:,:,0,:]
                    label=z['labels'][:,:,None];lf=np.log(np.clip(pr,.001,.999)/(1-np.clip(pr,.001,.999)))
                    branch=[]
                    for j,k in enumerate((2,4)):
                        est=cue_est(ar['direction'],f,x[:,:k],y[:,:k],sign[:,:k],'same')
                        # v33 all negative, so raw outgoing is already in the frozen feature convention.
                        np.testing.assert_allclose(c['features'][:,j,11],y[:,:k,1].std(1),atol=1e-12)
                        vals=[]
                        for mode,key in enumerate(CUES[:3]):
                            cue=(0,est,1-2*d)[mode]
                            xx=design(c['features'][:,j],-1.,0.,cue)
                            m=ar['head']['folds'][str(f)][key][str(k)]
                            vals.append((xx-m['mean'])/m['sd']@m['beta']+m['intercept'])
                            if k==4:
                                ms=tm[str(f)][mode]
                                inp=features(lf,np.broadcast_to(ac[:,:,None],lf.shape),est[:,None,None] if mode==1 else cue)
                                pred=expit((inp-ms['mean'])/ms['sd']@ms['beta'][1:]+ms['beta'][0])
                                v[f,b,d,1,mode]=((pred-label)**2).mean()
                        branch.append(vals)
                    ks=alloc(np.array(branch).transpose(1,0,2))
                    for mode in range(3):v[f,b,d,0,mode]=z['oracle_stage_risk'][np.arange(8),ks[mode]].mean()
    def dif(x):return (x[...,1:]-x[...,:1]).reshape(x.shape[:-2]+(4,))
    mu=v.mean((0,1,2));ids=np.random.default_rng(340444).integers(8,size=(12000,4,8))
    q=v[np.arange(4)[None,:,None],ids].mean((1,2,3))
    dd=dif(mu);dump('external34.json',dict(delta=dd,ci=band(dd,dif(q)),means=mu,schedule='negative-only v33',n_disjoint_clusters=32))
if __name__=='__main__':main()
