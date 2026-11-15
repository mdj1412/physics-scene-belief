"""Source-only held-out check for regime identification and expected-gain sign."""
import json
import numpy as np
from scipy.stats import rankdata
from gate36 import ROOT,load_artifact,predict,SIGNS,MODES,FAMILIES
from train_gate36 import parameters,shape,impact,factual,STRENGTH,SOURCE

N_ENV=128
SEED=64003617

def score(row,art,source_idx):
    z=row['z']
    gain=np.interp(z,art['zgrid'],art['gain'][source_idx])
    return np.column_stack((row['score3'],row['score4'],row['accept3'],row['accept4'],
              gain,row['near_strength3'],row['near_strength4']))

def main():
    a=load_artifact()
    alphas=np.array(json.loads((SOURCE/'policy_design34.json').read_text())['alphas'])
    panel=[]
    for family_idx,family in enumerate(FAMILIES):
        rng=np.random.default_rng(SEED+family_idx)
        alpha=alphas[rng.integers(len(alphas),size=N_ENV)]
        unit=rng.random((N_ENV*4,9))
        flip=(rng.random((N_ENV*4,2))<(1-np.repeat(alpha,4,axis=0))/2).astype(int)
        th=shape(parameters(unit,flip),family)
        x=factual(th,family).reshape(N_ENV,4,11)+rng.normal(size=(N_ENV,4,11))*np.array([.0016]*9+[.095]*2)
        noise=rng.normal(0,.027,(N_ENV,4,2))
        d=(rng.integers(2,size=(N_ENV,1))*2-1)*np.array([1,-1,1,-1])[None]
        for j,sign in enumerate(SIGNS):
            for k,mode in enumerate(MODES):
                conventional=impact(th,0,d.ravel()*1.10,family,sign,float(k!=3))[0].reshape(N_ENV,4,2)+noise
                matched=impact(th.reshape(N_ENV,4,5)[:,3],.023,.50,family,sign,STRENGTH[k])[0]+noise[:,3]
                # Bound memory even for the full panel, independent prefixes.
                pieces=[]
                for start in range(0,N_ENV,16):
                    sl=slice(start,start+16)
                    r=predict(a,x[sl],conventional[sl,:3],d[sl,:3],matched[sl],'alternating')
                    pieces.append(score(r,a,(family_idx*2+j)*4+k))
                panel.append(np.concatenate(pieces))
    arr=np.array(panel).reshape(4,2,4,N_ENV,7)
    def auc(x,y):
        pos=x.ravel();neg=y.ravel()
        rank=rankdata(np.r_[pos,neg])
        return (rank[:len(pos)].mean()-(len(pos)+1)/2)/len(neg)
    m={}
    for k,reg in enumerate(MODES):
        sub=arr[:,:,k]
        m[reg]=dict(accept3=sub[...,2].mean(),accept4=sub[...,3].mean(),
                    expected_brier_gain_gate3=(sub[...,2]*sub[...,4]).mean(),
                    expected_brier_gain_gate4=(sub[...,3]*sub[...,4]).mean(),
                    predicted_gain_gate4=sub[...,1].mean(),actual_archived_gain_if_always=sub[...,4].mean())
    diagnostic={s:auc(arr[:,:,0,:,i],arr[:,:,1:3,:,i]) for i,s in [(5,'three'),(6,'four')]}
    def digest(p):
        import hashlib
        return hashlib.sha256(p.read_bytes()).hexdigest()
    out=dict(artifact_sha256=digest(ROOT/'gate36_frozen.npz'),seed=SEED,independent_draws=4*N_ENV,
             orientation_and_local_modes='shared objects and noise within each family and draw',
             total_layout=int(np.prod(arr.shape[:4])),regimes=m,near_strength_auc=diagnostic)
    (ROOT/'gate36_source_check.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
