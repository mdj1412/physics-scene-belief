from pathlib import Path
import json,os
import numpy as np
from diag34_generate import FAMILIES,SCHED,STAGES,ALPHAS,OUT,stream,torch
from target34 import calibration
from target33 import OBJECTS,SURFACES
from diag34_model import raw_features,fit,predict
from diag34_stats import simultaneous,dump
ROOT=Path(__file__).resolve().parent

def generate(fam,base,n):
    zs=np.empty((n,8,2,2,3,44)); tags=np.broadcast_to(np.array([1.,0.]),(n,8,2))
    for b in range(n):
        alpha=np.array(ALPHAS)[stream(base+100*b,33).permutation(8)]
        for e,coupling in enumerate(alpha):
            start=(-1,1)[(b+e)%2]
            for d,tag in enumerate(('plus','minus')):
                for j,sch in enumerate(SCHED):
                    x,idx,y,dd=calibration(base+100*b+e,coupling,fam+'_'+tag,sch,start)
                    for s,k in enumerate(STAGES):
                        zs[b,e,d,j,s]=raw_features(x[idx[:k]][None],y[None,:k],dd[None,:k])[0]
    return zs,tags

def main():
    assert os.environ.get('SLURM_JOB_ID') and not torch.cuda.is_available()
    torch.set_num_threads(1)
    OBJECTS['reference']=(1.,0.)
    SURFACES['reference']=(1.,.20,.41)
    with torch.inference_mode():
        z,t=generate('reference_reference',1045000000,64)
        mdl={}
        targets=[generate(fam,1046000000+i*100000,16) for i,fam in enumerate(FAMILIES)]
    loss=np.empty((4,16,2,3));acc=np.empty_like(loss)
    for j,sch in enumerate(SCHED):
        mdl[sch]={}
        for s,k in enumerate(STAGES):
            m=fit(z[:,:,:,j,s].reshape(-1,44),t.ravel());mdl[sch][str(k)]=m
            for f,(zz,tt) in enumerate(targets):
                pred=predict(m,zz[:,:,:,j,s].reshape(-1,44)).reshape(tt.shape)
                loss[f,:,j,s]=((tt-pred)**2).mean((1,2));acc[f,:,j,s]=((pred>.5)==tt).mean((1,2))
    l=loss.mean((0,1));ids=np.random.default_rng(340517).integers(16,size=(16000,4,16))
    star=loss[np.arange(4)[None,:,None],ids].mean((1,2))
    diff=l[1]-l[0]
    dump(OUT/'universal_analysis.json',dict(mean=l,acc=acc.mean((0,1)),delta=diff,ci=simultaneous(diff,star[:,1]-star[:,0]),by_family=loss.mean(1),train_ref='new neutral reference',n_independent_test_blocks=64))
    dump(ROOT/'diag_universal_model.json',dict(models=mdl,description='single model per scheduler and paid stage, no family tag'))
if __name__=='__main__':main()
