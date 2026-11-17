"""Post-hoc diagnostic of concurrent instrument disagreement versus two-object dependence."""
import os,json
from pathlib import Path
import numpy as np
from route59 import TABLE,log_odds,Q,MOMENTS,CAMERA_ERROR,binary_pair

ROOT=Path(__file__).resolve().parent
MODES=['full','within_object_marginal','camera_only_grouped','shuffled_second_full','shuffled_second_marginal']
def contrasts(table,codes):
    by=table[:,:,codes].sum(-1)
    z=np.logaddexp(by[0],by[1]); mechanism=np.logaddexp(by[:,0],by[:,1])
    return z[0]-z[1], mechanism[0]-mechanism[1]
def camera_table():
    emission=np.array([[1,1-CAMERA_ERROR,CAMERA_ERROR],[1,CAMERA_ERROR,1-CAMERA_ERROR]])
    result=np.empty((2,2,9))
    for k,z in np.ndindex(2,2):
        joint=emission.T@binary_pair(MOMENTS[z] if k==0 else Q**2)@emission
        result[k,z]=np.log(joint.ravel())
    return result

def main():
    assert os.getenv('SLURM_JOB_ID') and os.getenv('CUDA_VISIBLE_DEVICES')==''
    cams=camera_table()
    pieces=[]
    for file in sorted((ROOT/'pilots').glob('panel_*.npz')):
        with np.load(file) as f: codes=f['codes'][:,1]
        by=np.empty((2,2,2,len(MODES),2))
        for r,k,z in np.ndindex(2,2,2):
            x=codes[r,k,z].astype('int32')
            first,second=np.divmod(x,13)
            shuffled=13*first+np.roll(second,1,axis=-2)  # cyclic shift across independent pilots within material
            table=TABLE[r,1]
            symbols=np.arange(169)
            a,b=np.divmod(symbols,13)
            marg=table[...,13*a]+table[...,b]
            def cam(s):
                return np.where(s==0,0,(s-1)//4)
            cc=3*cam(first)+cam(second)
            for j,(laws,data) in enumerate([(table,x),(marg,x),(cams,cc),(table,shuffled),(marg,shuffled)]):
                z_odds,k_odds=contrasts(laws,data)
                by[r,k,z,j]=[(odds>1e-12).mean() for odds in (z_odds,k_odds)]
        pieces.append(by)
    x=np.stack(pieces)
    cause=(x[:,:,0,:,:,-1]+1-x[:,:,1,:,:,-1]).mean(axis=2)/2
    regime=(x[:,:,:,0,:,0]+1-x[:,:,:,1,:,0]).mean(axis=2)/2
    rng=np.random.default_rng(59032874)
    inds=rng.integers(24,size=(5000,24))
    result={'methods':MODES,'cause_acc':cause.mean(0).tolist(),'regime_acc':regime.mean(0).tolist()}
    for name,item in [('cause',cause),('regime',regime)]:
        dif=item[:,:,0,None]-item[:,:,1:]
        result[name+'_full_minus_controls_95']=np.quantile(dif[inds].mean(1),[.025,.975],axis=0).tolist()
    (ROOT/'cues59.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
