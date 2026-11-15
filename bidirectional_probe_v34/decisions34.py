import hashlib,json,os,sys
from pathlib import Path
import numpy as np
from direct_head34 import FAMILIES,SCHEDULES,CUES,design,cue_est
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'results_policy'
read=lambda p: json.loads(p.read_text())
def dump(f,z):
    p=ROOT/(f+'.tmp'); p.write_text(json.dumps(z,indent=2,allow_nan=False,default=lambda a:a.tolist())+'\n'); p.replace(ROOT/f)
def band(p,draw):
    se=draw.std(0,ddof=1);c=np.quantile(abs((draw-p)/np.where(se>1e-14,se,1)).max(-1),.95)
    return np.stack([p-c*se,p+c*se],axis=-1)
def getdata():
    lock=read(ROOT/'diag_models.json')
    X=np.zeros((4,8,2,2,2,8,15));risk=np.empty((4,8,2,2,8,3));raw=np.empty((4,8,2,2,3))
    cue=np.zeros((4,4,8,2,2,2,8));sel=np.zeros((4,8,2,2,3,8),dtype='i')
    rng=np.random.default_rng(3409122);flip=np.zeros((4,8),dtype=int)
    for f in range(4):
        # orientation swap is balanced for each start sign in each material family.
        for start in (0,1):flip[f,2*rng.permutation(4)[:2]+start]=1
    for f,fam in enumerate(FAMILIES):
        for b in range(8):
            for d,orient in enumerate(('plus','minus')):
                for a,schedule in enumerate(SCHEDULES):
                    p=OUT/(fam+'_'+orient)/f'block_{b:03d}'/schedule
                    with np.load(p/'choices.npz') as z,np.load(p/'metrics.npz') as m:
                        x=z['paid_x'];y=z['paid_y'];dire=z['paid_direction'];u=z['features']
                        np.testing.assert_array_equal(dire[:,0],np.full(8,2*(b%2)-1))
                        assert np.array_equal(dire,np.broadcast_to(dire[:1],dire.shape))
                        assert all(len(set(v))==8 for v in z['paid_ids'])
                        X[f,b,d,a]=u.transpose(1,0,2);risk[f,b,d,a]=m['oracle_stage_risk']
                        raw[f,b,d,a]=m['brier'].mean(0);sel[f,b,d,a]=z['counts'].T
                        assert list(m['methods'])==['broad','fixed4','uncertainty']
                        for fold in {f,3-f}:
                            for j,k in enumerate((2,4)):
                                cue[fold,f,b,d,a,j]=cue_est(lock,fold,x[:,:k],y[:,:k],dire[:,:k],schedule)
                        if a==0:
                            yy=y.copy();xx=x.copy();ref=(m['labels'].copy(),m['probabilities'].copy())
                        else:
                            np.testing.assert_array_equal(xx,x);np.testing.assert_array_equal(yy[:,::2],y[:,::2])
                            np.testing.assert_array_equal(ref[0],m['labels']);np.testing.assert_array_equal(ref[1],m['probabilities'])
    return X,risk,raw,cue,flip,sel
def xmatrix(X,cue,flip,fold,f,key):
    s=(-1+2*(np.arange(8)%2))[:,None,None,None,None];a=np.arange(2)[None,None,:,None,None]
    true=np.array([1,-1])[None,:,None,None,None]
    cu=cue[fold,f]
    if key=='neutral':cu=cu*0
    if key=='oracle':cu=np.broadcast_to(true,cu.shape)
    if key=='placebo':cu=np.stack([v[::-1] if flip[f,i] else v for i,v in enumerate(cu)])
    return design(X[f],s,a,cu)
def fit(x,y):
    xx=x.reshape(-1,x.shape[-1]); yy=y.flatten();m=xx.mean(0);s=xx.std(0);s[s<1e-10]=1.
    zz=(xx-m)/s; ym=yy.mean();beta=np.linalg.solve(zz.T@zz/len(yy)+.3*np.eye(zz.shape[-1]),zz.T@(yy-ym)/len(yy))
    return dict(mean=m,sd=s,beta=beta,intercept=float(ym))
def alloc(sc):
    rank0=np.argsort(-sc[...,0,:],axis=-1,kind='stable');first=np.sort(rank0[...,:4],axis=-1)
    other=np.argsort(-np.take_along_axis(sc[...,1,:],first,axis=-1),axis=-1,kind='stable')[...,:2]
    second=np.take_along_axis(first,other,axis=-1)
    out=np.zeros(sc.shape[:-2]+(8,),int);np.put_along_axis(out,first,1,axis=-1);np.put_along_axis(out,second,2,axis=-1)
    return out
def pipeline(data,samples,storefit=False):
    X,risk,raw,cue,flip,sel=data
    vals=[];models={};groups=[];score_diag=[]
    for f in range(4):
        g=3-f;tr=samples[g];te=samples[f];fold=[];mout={};gg=[];sdia=[]
        for key in CUES:
            xd=xmatrix(X,cue,flip,f,g,key)[tr];xt=xmatrix(X,cue,flip,f,f,key)[te]
            gains=risk[g,tr,...,:-1]-risk[g,tr,...,1:]
            tgt=risk[f,te,...,:-1]-risk[f,te,...,1:]
            scored=[];mout[key]={};rhos=[]
            for j,k in enumerate((2,4)):
                mo=fit(xd[...,j,:,:],gains[...,j]);mout[key][str(k)]=mo
                sc=(xt[...,j,:,:]-mo['mean'])/mo['sd']@mo['beta']+mo['intercept'];scored.append(sc)
                if storefit:
                    cc=[np.corrcoef(a,b)[0,1] if np.std(a)*np.std(b)>1e-20 else 0. for a,b in zip(sc.reshape(-1,8),tgt[...,j].reshape(-1,8))]
                    rhos.append(np.mean(cc))
            ks=alloc(np.stack(scored,-2));rr=np.take_along_axis(risk[f,te],ks[...,None],axis=-1)[...,0]
            gg.append(ks);fold.append(rr.mean(-1));sdia.append(rhos)
        models[str(f)]=mout;groups.append(np.stack(gg,-2));score_diag.append(sdia)
        vals.append(np.concatenate((raw[f,te],np.stack(fold,-1)),axis=-1))
    return np.array(vals), models, np.array(groups), np.array(score_diag)
def dif(v):
    # leading dims before schedule x method; order broad,fixed,unc,neutral,estimated,oracle,placebo
    core=np.concatenate((v[...,1,:3]-v[...,0,:3],*[np.stack([v[...,a,0]-v[...,a,1],v[...,a,0]-v[...,a,2],v[...,a,2]-v[...,a,1]],-1) for a in (0,1)]),axis=-1)
    aug=np.stack([v[...,a,i]-v[...,a,j] for a in (0,1) for i,j in ((4,3),(5,3),(4,6),(4,1),(4,2))]+[v[...,1,4]-v[...,0,4]],axis=-1)
    return core,aug
def sample(rng,n):
    i=rng.integers(4,size=(n,4,4,2))*2+np.arange(2);return i.reshape(n,4,8)
def main():
    data=getdata(); ids=np.tile(np.arange(8),(4,1));v,mo,ks,sd=pipeline(data,ids,True)
    dump('direct_models34.json',dict(folds=mo,source_for_fold=[3,2,1,0],penalty=.3,diag_sha=hashlib.sha256((ROOT/'diag_models.json').read_bytes()).hexdigest()))
    mu=v.mean((0,1,2));p,q=dif(mu);rng=np.random.default_rng(340809);idx=sample(rng,12000)
    rr=v[np.arange(4)[None,:,None],idx].mean((1,2,3));pc,qc=dif(rr)
    repeat=[]
    for ii in sample(rng,1500):
        new,_,_,_=pipeline(data,ii);repeat.append(dif(new.mean((0,1,2)))[1])
    report=dict(methods=['broad','fixed4','uncertainty',*CUES],means=mu,core=p,core_band=band(p,pc),aug=q,aug_band=band(q,qc),refit_band=band(q,np.array(repeat)),
              by_family=v.mean((1,2)),by_sign=v.mean((0,1)),start_means=[v[:,i::2].mean((0,1,2)) for i in (0,1)],
              counts_changed_est_v_neutral=(ks[...,1,:]!=ks[...,0,:]).mean((0,1,2,4)),gain_score_corr=sd.mean(0),
              independent_blocks=32,realizations_per_schedule=65536,refit_replicates=1500)
    dump('decision_stats34.json',report)
if __name__=='__main__':main()
