"""Evaluate speed response on the same independent source draws as the v36 probe experiment."""
import json
import numpy as np
from scipy.stats import rankdata
from gate37 import ROOT, SOURCE, SPEEDS, load_artifacts, predict
from gate36 import FAMILIES, SIGNS, MODES
from train_gate36 import STRENGTH, parameters, shape, factual, impact
from validate_gate36 import N_ENV, SEED

def auc(pos,neg):
    p,n=pos.ravel(),neg.ravel()
    ranks=rankdata(np.r_[p,n])
    return float((ranks[:len(p)].mean()-(len(p)+1)/2)/len(n))

def main():
    gates=load_artifacts()
    factors=np.asarray(json.loads((SOURCE.parent/'bidirectional_probe_v34/policy_design34.json').read_text())['alphas'])
    panel=np.empty((4,2,4,3,N_ENV,4))
    for f,family in enumerate(FAMILIES):
        rng=np.random.default_rng(SEED+f)
        alpha=factors[rng.integers(len(factors),size=N_ENV)]
        unit=rng.random((N_ENV*4,9))
        flips=(rng.random((N_ENV*4,2))<(1-np.repeat(alpha,4,axis=0))/2).astype(int)
        th=shape(parameters(unit,flips),family)
        x=factual(th,family).reshape(N_ENV,4,11)+rng.normal(size=(N_ENV,4,11))*np.array([.0016]*9+[.095]*2)
        noise=rng.normal(0,.027,(N_ENV,4,2))
        dirs=(rng.integers(2,size=(N_ENV,1))*2-1)*np.array([1,-1,1,-1])[None]
        for j,sign in enumerate(SIGNS):
            for k,amp in enumerate(STRENGTH):
                far=impact(th,0,dirs.ravel()*1.10,family,sign,float(k!=3))[0].reshape(N_ENV,4,2)+noise
                for v,speed in enumerate(SPEEDS):
                    art=gates[speed]
                    near,stick=impact(th.reshape(N_ENV,4,5)[:,3],.023,speed,family,sign,amp)
                    near+=noise[:,3]
                    for l in range(0,N_ENV,16):
                        sl=slice(l,l+16)
                        r=predict(art,x[sl],far[sl,:3],dirs[sl,:3],near[sl],'alternating')
                        gain=np.interp(r['z'],art['zgrid'],art['gain'][(f*2+j)*4+k])
                        panel[f,j,k,v,sl]=np.stack([~stick[sl],r['near_strength4'],r['accept4'],gain],axis=-1)
    results={}
    for i,speed in enumerate(SPEEDS):
        z=panel[:,:,:,i]
        results[str(speed)]=dict(slip=z[:,:,:3,:,0].mean(),strong_low_auc=auc(z[:,:,0,:,1],z[:,:,1:3,:,1]),
           weak_null_auc=auc(z[:,:,1,:,1],z[:,:,2,:,1]),apply=z[...,2].mean((0,1,3)).tolist(),
           archived_gain_gate_vs_original=float((z[:,:,:3,:,2]*z[:,:,:3,:,3]).mean()))
    reference=json.loads((SOURCE/'gate36_source_check.json').read_text())
    np.testing.assert_allclose(results['0.5']['strong_low_auc'],reference['near_strength_auc']['four'],atol=1e-12)
    out=dict(speeds=results,n_independent_draws=4*N_ENV,source_evaluation_seed=SEED,overlap_with_v36='same archived source validation conditions')
    (ROOT/'source37.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(results))
if __name__=='__main__':main()
