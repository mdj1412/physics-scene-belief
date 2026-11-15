"""Read paired, fresh held-out contact laws without selecting features on test outcomes."""
from pathlib import Path
import sys,json,hashlib,time,os
import numpy as np
from diag34_model import raw_features,fit,predict,raw_probability
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'results_diag'
FAMS=('rounded_fine','rounded_coarse','ridged_fine','ridged_coarse')
SCHED=('same','alternating')
K=(2,4,8)
MODES=('factual','raw','features_linear','scalar_linear','features_rbf','scalar_rbf')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,obj):
    p.write_text(json.dumps(obj,indent=2,default=lambda a:a.tolist())+'\n')


def read_panel(split,small):
    n=8 if small else (64 if split=='train' else 48)
    fields={k:[] for k in ('x','y','d','alpha','features','scores','seeds')}
    for family in FAMS:
        for block in range(n):
            f=OUT/(('heads_' if small else '')+f'{split}_{family}_{block:03d}.npz')
            with np.load(f) as z:
                assert bool(z['has_heads'])==small
                if small:
                    with np.load(OUT/f'{split}_{family}_{block:03d}.npz') as raw:
                        for k in ('x','y','d','seeds','alpha'):np.testing.assert_array_equal(raw[k],z[k])
                for key in fields:fields[key].append(z[key])
    return {k:np.array(v).reshape((4,n)+v[0].shape) for k,v in fields.items()}


def transform(compressed,start,basis=None,nonlinear=False):
    # All readouts of existing inputs also receive known starting sign. Every environment contributes both orientations.
    if nonlinear:
        if basis is None:
            center=np.mean(compressed,axis=0);sd=np.std(compressed,axis=0)
            sd=np.where(sd>1e-8,sd,1.)
            basis=(center,sd)
        center,sd=basis;norm=(compressed-center)/sd
        columns=[norm]+[np.exp(-.5*((norm-q)/.6)**2) for q in (-2,-1,0,1,2)]
        z=np.concatenate(columns,axis=1)
    else:z=compressed
    return np.column_stack((start,z,start[:,None]*z)),basis


def panel_rows(panel,f,s,j,k,mode,basis=None):
    n=panel['x'].shape[1];y=panel['y'][f,:,:,:,j,:k]
    dirs=panel['d'][f,:,:,j,:k]
    x=panel['x'][f,:,:,:k]
    x=np.repeat(x[:,:,None,:,:],2,axis=2)
    d=np.repeat(dirs[:,:,None,:],2,axis=2)
    x=x.reshape(-1,k,11); y=y.reshape(-1,k,2);d=d.reshape(-1,k)
    ytrue=np.tile([1.,0.],n*8)
    start=d[:,0]
    if mode=='raw':
        z=raw_features(x,y,d)
    elif mode=='factual':
        obs=np.column_stack((x.mean(axis=1),x.std(axis=1)))
        z,_=transform(obs,start)
    elif mode.startswith('scalar') or mode.startswith('features'):
        if mode.startswith('scalar'):
            zz=panel['scores'][f,:,:,:,j,s].reshape(-1,1)
        else:
            zz=panel['features'][f,:,:,:,j,s].reshape(-1,15)
        z,basis=transform(zz,start,basis,nonlinear=mode.endswith('rbf'))
    else:raise ValueError(mode)
    return dict(z=z,labels=ytrue,x=x,y=y,d=d,start=start,basis=basis)


def simultaneous(v,draw):
    sd=draw.std(0,ddof=1)
    z=np.divide(draw-v,sd,out=np.zeros_like(draw),where=sd>1e-12)
    q=np.max(abs(z),axis=-1);c=np.quantile(q,.95)
    return np.stack((v-c*sd,v+c*sd),axis=-1)


def summarize_large(data,train,lock):
    shape=(4,48,8,2,2,3)
    pr=np.empty(shape);perm=np.empty(shape);fact=np.empty(shape);starts=np.empty(shape[:-1])
    coin_target=np.zeros((4,48,8,2));coin_train=np.zeros((4,64,8,2))
    for coin,se in ((coin_target,340302),(coin_train,340301)):
        rng=np.random.default_rng(se)
        coin[:,:,:,0]=rng.integers(2,size=coin.shape[:3]);coin[:,:,:,1]=1-coin[:,:,:,0]
    for f,family in enumerate(FAMS):
        for j,sched in enumerate(SCHED):
            for s,k in enumerate(K):
                row=panel_rows(data,f,s,j,k,'raw')
                pr[f,:,:,:,j,s]=raw_probability(lock,family,row['x'],row['y'],row['d'],sched).reshape(48,8,2)
                starts[f,:,:,:,j]=row['start'].reshape(48,8,2)
                tr=panel_rows(train,3-f,s,j,k,'raw')
                shuffled=fit(tr['z'],coin_train[3-f].flatten())
                perm[f,:,:,:,j,s]=predict(shuffled,row['z']).reshape(48,8,2)
                xx=panel_rows(data,f,s,j,k,'factual');fx=panel_rows(train,3-f,s,j,k,'factual')
                fact[f,:,:,:,j,s]=predict(fit(fx['z'],fx['labels']),xx['z']).reshape(48,8,2)
    y=np.broadcast_to(np.array([1.,0.])[None,None,None,:,None,None],pr.shape)
    sq=(pr-y)**2; cor=((pr>.5)==y).astype(float);cor[pr==.5]=.5
    loss=sq.mean((2,3));acc=cor.mean((2,3));mean=loss.mean((0,1))
    b=np.random.default_rng(340211).integers(48,size=(16000,4,48))
    boot=loss[np.arange(4)[None,:,None],b].mean((1,2))
    acc_draw=acc[np.arange(4)[None,:,None],b].mean((1,2))
    delta=mean[1]-mean[0]
    start_stats=np.stack([np.where(starts[...,None]==v,sq,np.nan) for v in (-1,1)])
    null=((perm-coin_target[:,:,:,:,None,None])**2)
    res=dict(schedules=SCHED,stages=K,mean=mean,ci=np.quantile(boot,[.025,.975],axis=0).transpose(1,2,0),accuracy=acc.mean((0,1)),
             accuracy_ci=np.quantile(acc_draw,[.025,.975],axis=0).transpose(1,2,0),
             delta=delta,delta_simultaneous95=simultaneous(delta,boot[:,1]-boot[:,0]),
             by_family=loss.mean(1),by_start=np.nanmean(start_stats,axis=(1,2,3,4)),
             factual_brier=((fact-y)**2).mean((0,1,2,3)),factual_max_deviation=float(np.max(abs(fact-.5))),
             permuted_target_brier=null.mean((0,1,2,3)),
             true_label='surface orientation, plus=1, minus=0',orientation_accuracy_tie_value=.5)
    # Fit isotropic-candidate directions in factual input is a negative control with exactly identical paired features.
    assert np.max(abs(fact-.5))<1e-9
    np.savez_compressed(OUT/'orientation_test.npz',probs=pr,labels=y[0,0,0,:,0,0],start=starts,shuffled=perm,coin=coin_target)
    return res


def summarize_small(data,train):
    shape=(4,8,8,2,2,2,len(MODES));pr=np.empty(shape)
    for f in range(4):
        for j,sched in enumerate(SCHED):
            for s,k in enumerate((2,4)):
                for m,mode in enumerate(MODES):
                    tr=panel_rows(train,3-f,s,j,k,mode)
                    te=panel_rows(data,f,s,j,k,mode,basis=tr['basis'])
                    fitted=fit(tr['z'],tr['labels'])
                    pr[f,:,:,:,j,s,m]=predict(fitted,te['z']).reshape(8,8,2)
    y=np.broadcast_to(np.array([1.,0.])[None,None,None,:,None,None,None],pr.shape)
    sq=(pr-y)**2;acc=((pr>.5)==y).astype(float);acc[abs(pr-.5)<1e-12]=.5
    risks=sq.mean((2,3));avg=risks.mean((0,1));ids=np.random.default_rng(340212).integers(8,size=(12000,4,8))
    boot=risks[np.arange(4)[None,:,None],ids].mean((1,2))
    # Positive means the compressed readout carries less usable orientation information.
    comp=avg[:,:,[2,3,4,5]]-avg[:,:,1,None]
    draw=boot[:,:,:,[2,3,4,5]]-boot[:,:,:,1,None]
    cis=simultaneous(comp.ravel(),draw.reshape(len(draw),-1)).reshape(comp.shape+(2,))
    sample_dir=np.mean((pr[:,:,:,0]-pr[:,:,:,1])>0,axis=(0,1,2))
    dump(OUT/'matched_metric_guide.json',dict(axes=['schedule','stage','mode'],effect='prediction Brier, lower is better; compressed minus raw positive denotes poorer decoding'))
    return dict(modes=MODES,means=avg,acc=acc.mean((0,1,2,3)),ci=np.quantile(boot,[.025,.975],axis=0).transpose(1,2,3,0),
                compressed_minus_raw=comp,simultaneous95_all16=cis,paired_true_orientation_order=sample_dir)


def main():
    assert os.environ.get('SLURM_JOB_ID')
    mode=sys.argv[1]
    if mode not in ('raw','small'):raise ValueError(mode)
    data=read_panel('test',mode=='small');tr=read_panel('train',mode=='small')
    assert not np.intersect1d(data['seeds'],tr['seeds']).size
    assert len(np.unique(data['seeds']))==data['seeds'].size
    lock=json.loads((ROOT/'diag_models.json').read_text())
    if mode=='raw':
        for file,digest in lock['training_hashes'].items():assert sha(OUT/file)==digest
        out=summarize_large(data,tr,lock)
    else:out=summarize_small(data,tr)
    dump(OUT/f'{mode}_analysis.json',out);print(mode,'DONE',flush=True)
if __name__=='__main__':main()
