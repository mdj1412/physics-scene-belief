"""Independent paired outcome reconstruction and statistical inference for v37."""
from pathlib import Path
import itertools, json,sys
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'query_matched_probe_v36'))
from stats36 import CELLS,REGIMES,choose_scope,draw,simultaneous,resampling,finite,sha,shift
from diagnose36 import response,normal_cdf

SPEEDS=(.50,.75,1.10)
METHODS=('original','always','shrink','gate')
SCOPE=('local_equal','weak_null_equal','strong','weak','null','global_null','four_equal')
BETA=.8289760825845313

def loss(p,labels):
    return np.square(p-labels[...,None]).mean(axis=(1,2))

def analyze_file(f,obj,sur,reg,sign,historical_scale):
    np.testing.assert_array_equal(f['speeds'],SPEEDS)
    assert float(f['beta'])==BETA
    labels=f['labels'].astype(bool)
    theta,theta4=f['theta'],f['theta4']
    factor={'strong':1.,'weak':.3,'null':0.,'global_null':0.}[reg]
    # At the matched and query heights, local-null has the same impulse as global-null.
    np.testing.assert_array_equal(labels,response(theta,obj,sur,sign,factor,.025,.425)[1])
    base=f['p_base'].astype(float); z=f['z']; actions=f['actions']
    assert base.shape==(3,8,128,32) and np.isfinite(base).all() and np.all((base>=-1e-12)&(base<=1+1e-12))
    assert (np.sum(actions==3,axis=-1)==13).all()
    gate=f['accept4'].astype(bool);scores=f['score4']
    np.testing.assert_array_equal(gate,scores>0)
    clean,stick,measure=f['matched_clean'],f['matched_stick'],f['fourth_y']
    noise=measure-clean
    np.testing.assert_allclose(noise,np.broadcast_to(noise[:1],noise.shape),atol=4e-14,rtol=0)
    assert len(np.unique(f['ids'][0]))==4 and np.all(np.diff(np.sort(f['ids'],axis=1),axis=1)>0)
    full=[];ex=[]
    p_ref=base[0]
    always_ref=shift(p_ref,z,BETA)
    for k,v in enumerate(SPEEDS):
        mu,st=response(theta4,obj,sur,sign,factor,.023,v)
        np.testing.assert_allclose(clean[k],mu,atol=1e-10,rtol=0.)
        np.testing.assert_array_equal(stick[k],st)
        p=base[k]; always=shift(p,z,BETA); fixed=shift(p,.325*z,BETA)
        gated=np.where(gate[k,:,None,None],always,p)
        values=np.stack([loss(q,labels) for q in (p,always,fixed,gated)])
        np.testing.assert_allclose(values,f['risk'][k],atol=1e-11,rtol=0.)
        full.append(values.mean(1))
        world_diff=(np.square(always-labels[...,None])-np.square(p-labels[...,None])).mean(-1)
        suba=world_diff[:,::2].mean(1); subb=world_diff[:,1::2].mean(1)
        oracle=values[0]+.5*((suba<0)*subb+(subb<0)*suba)
        common=loss(np.where(gate[k,:,None,None],always_ref,p_ref),labels)
        soft=loss(shift(p,f['near_strength4'][k]*z,BETA),labels)
        g3=f['score3']>0
        g3loss=np.where(g3,values[1],values[0])
        all_extra=np.stack([~st,gate[k],gate[k]!=gate[0],(actions[k]!=actions[0]).mean(1),
                      common,oracle,np.minimum(values[0],values[1]),soft,g3loss,
                      loss(shift(p,historical_scale*z,BETA),labels)],axis=1)
        ex.append(all_extra.mean(0))
    return np.array(full),np.array(ex),clean,stick.astype(bool)

def brier_contrasts(losses):
    diff=losses[...,3]-losses[...,2]
    changes=np.stack([losses[...,i,3]-losses[...,0,3] for i in (1,2)],axis=-1)
    did=diff[...,1:]-diff[...,:1]
    return np.concatenate((diff,changes,did),axis=-1)

def summarize(losses,extras,idx):
    groups={}
    for name in SCOPE:
        if name=='weak_null_equal':
            raw=losses[:,:,1:3].mean(2); extra=extras[:,:,1:3].mean(2)
        else:
            raw=choose_scope(losses,name,2); extra=choose_scope(extras,name,2)
        ctr=brier_contrasts(raw); point=ctr.mean((0,1)); bs=draw(ctr,idx)
        ed=extra[...,1:,:]-extra[...,:1,:]
        epoint=ed.mean((0,1)); ebs=draw(ed,idx)
        tuned=raw[...,3]-extra[...,9]; sampled_tuned=draw(tuned,idx)
        groups[name]=dict(tuned_fixed_delta=tuned.mean((0,1)), tuned_fixed_ci95=simultaneous(tuned.mean((0,1)),sampled_tuned),losses=raw.mean((0,1)),extras=extra.mean((0,1)),brier_delta=point,
            brier_simul95=simultaneous(point,bs),extra_speed_change=epoint,
            extra_speed_ci95=np.moveaxis(np.quantile(ebs,[.025,.975],axis=0),0,-1),
            slip_delta_simul95=simultaneous(epoint[:,0],ebs[:,:,0]),
            cell_gate_minus_shrink=(raw[...,3]-raw[...,2]).mean(1))
    return groups

def discrimination(scores,idx):
    # Score C×B×reg×sign×environment×speed. Keep the same block multiplicity in both classes.
    sample_ids=(idx+8*np.arange(4)[None,:,None]).reshape(len(idx),32)
    w=np.stack([(sample_ids==j).sum(1)/32 for j in range(32)],-1)
    all_boot=[];out={}; all_points=[]
    for tag,positive,negative in [('strong_vs_low',0,[1,2]),('weak_vs_null',1,[2]),('strong_vs_null',0,[2])]:
        p=[];b=[]
        for k in range(3):
            src=scores[...,k];left=src[:,:,positive].reshape(32,-1);right=np.take(src,negative,axis=2).reshape(32,-1)
            diff=left[:,None,:,None]-right[None,:,None,:]
            kernel=np.mean((diff>0)+.5*(diff==0),axis=(2,3))
            p.append(kernel.mean());b.append(((w@kernel)*w).sum(1))
        p=np.array(p);bs=np.stack(b,-1)
        ds=p[1:]-p[:1];bd=bs[:,1:]-bs[:,:1]
        out[tag]=dict(auc=p,ci95=np.quantile(bs,[.025,.975],axis=0).T,delta=ds)
        all_points.append(ds);all_boot.append(bd)
    # Primary family: two speed contrasts × two discrimination tasks.
    adj=simultaneous(np.stack(all_points[:2]),np.stack(all_boot[:2],axis=1))
    for j,tag in enumerate(['strong_vs_low','weak_vs_null']):
        out[tag]['delta_simul95']=adj[j]
    out['strong_vs_null']['delta_pointwise95']=np.quantile(all_boot[2],[.025,.975],axis=0).T
    return out

def oracle_resolution(clean,stick):
    # Each row pairs candidate surfaces with the same latent object, direction and Gaussian noise.
    out={}
    for name,i,j in [('strong_vs_null',0,2),('weak_vs_null',1,2)]:
        dist=np.linalg.norm((clean[:,:,i]-clean[:,:,j])/.027,axis=-1)
        # Dimensions C,B,sign,speed,E.
        acc=normal_cdf(dist/2)
        base_both_slip=(~stick[:,:,i]) & (~stick[:,:,j])
        rows=[]
        for k in (1,2):
            first_i,first_j=stick[:,:,i,:,0],stick[:,:,j,:,0]
            later_i,later_j=stick[:,:,i,:,k],stick[:,:,j,:,k]
            switched=(first_i & ~later_i)|(first_j & ~later_j)
            persistent=base_both_slip[:,:,:,0]&base_both_slip[:,:,:,k]
            masks=(persistent,switched,~(persistent|switched))
            assert not (persistent&switched).any()
            delta=acc[:,:,:,k]-acc[:,:,:,0]
            rows.append([dict(fraction=m.mean(),weighted_gain=np.mean(m*delta),conditional_gain=delta[m].mean() if m.any() else None) for m in masks])
        out[name]=dict(accuracy=acc.mean((0,1,2,4)),identical=(dist<1e-9).mean((0,1,2,4)),
                       strata=rows,stratum_names=['slip_in_both_at_both_speeds','any_stick_to_slip','other'])
    return out

def timing_summary(times):
    # 4 cells × 2 timing blocks × 3 repeats × 3 speeds × 2 policies
    raw=np.stack([[times[c,b] for b in range(2)] for c in range(4)])
    assert raw.shape==(4,2,3,3,2) and np.all(raw>0)
    p90=np.quantile(raw,.9,axis=(0,1,2)); logratio=np.log(p90[:,1]/p90[:,0])
    profiles={}
    for n in itertools.product(range(3),repeat=4):
        x=np.stack([raw[c,np.repeat([0,1],[k,2-k])] for c,k in enumerate(n)])
        q=np.quantile(x,.9,axis=(0,1,2))
        profiles[n]=np.log(q[:,1]/q[:,0])
    rng=np.random.default_rng(370519)
    sampled=rng.binomial(2,.5,size=(12000,4))
    reps=np.stack([profiles[tuple(p)] for p in sampled])
    return dict(p90_ms=p90,ratio=np.exp(logratio),ratio_simul95=np.exp(simultaneous(logratio,reps)),
                timed_blocks=8,calls_per_speed_method=24)

def process():
    raw=ROOT/'results'
    assert (raw/'completion.json').is_file()
    settings=json.loads((ROOT/'design37.json').read_text())
    source_hashes=json.loads((raw/'run.json').read_text())['hashes']
    for p,h in source_hashes.items():assert sha(ROOT.parent/p)==h,(p,h)
    historical_scale=json.loads((ROOT/'fixed_validation37.json').read_text())['scale']
    idx=resampling(8,seed=370517)
    risks=np.zeros((4,8,4,3,4));extra=np.zeros((4,8,4,3,10))
    scores=np.empty((4,8,3,2,8,3))
    clean=np.empty((4,8,3,2,3,8,2));stick=np.empty((4,8,3,2,3,8),dtype=bool)
    times={};seeds=set();n=0
    for c,cell in enumerate(CELLS):
        obj,sur=cell.split('_')
        for b in range(8):
            folder=raw/cell/f'block_{b:03d}'
            meta=json.loads((folder/'meta.json').read_text())
            expected=settings['base_seed']+c*settings['cell_stride']+b*settings['block_stride']
            assert meta['base']==expected
            ids=meta['calibration_seeds']+meta['deployment_seeds']+meta['pair']
            assert len(set(ids))==18 and seeds.isdisjoint(ids)
            seeds.update(ids)
            for si,sign in enumerate((1,-1)):
                cache={}
                for ri,reg in enumerate(REGIMES):
                    if ri==3 and sign==-1:continue
                    name=reg+('_plus' if sign==1 else '_minus')+'.npz'
                    path=folder/name
                    assert sha(path)==meta['sha256'][name]
                    with np.load(path,allow_pickle=False) as f:
                        np.testing.assert_array_equal(f['methods'],METHODS)
                        np.testing.assert_array_equal(f['dirs'],np.tile(np.array([1,-1,1])*((-1,1)[b%2]),(8,1)))
                        vals,xt,mu,st=analyze_file(f,obj,sur,reg,sign,historical_scale)
                        weight=.5 if ri<3 else 1.
                        risks[c,b,ri]+=weight*vals;extra[c,b,ri]+=weight*xt
                        if ri<3:
                            scores[c,b,ri,si]=f['near_strength4'].T
                            clean[c,b,ri,si]=mu;stick[c,b,ri,si]=st
                            for key in ('theta4','theta','first3_y','z','score3','far_x','ids','dirs','alpha'):
                                if ri==0:cache[key]=f[key]
                                else:np.testing.assert_array_equal(f[key],cache[key])
                        ms=f['full_ms']
                        if ms.size:
                            assert (c,b) not in times
                            np.testing.assert_array_equal(np.sort(f['timing_order'],1),np.tile(np.arange(6),(3,1)))
                            times[c,b]=ms
                    n+=1
    assert n==224 and len(times)==8
    out=dict(speeds=SPEEDS,methods=METHODS,extras=['slip','apply','switch_vs_reference','actions_changed_fraction','gate_on_shared_p',
             'twofold_oracle','optimistic_oracle','soft','three_reading_gate','v36_tuned_fixed'],
             brier_order=['gate-shrink .50','gate-shrink .75','gate-shrink 1.10','gate .75-.50','gate 1.10-.50',
             'change of gate-shrink .75-.50','change of gate-shrink 1.10-.50'],
             scopes=summarize(risks,extra,idx),discrimination=discrimination(scores,idx),sensor=oracle_resolution(clean,stick),
             timing=timing_summary(times),verified_archives=n,source_hashes=len(source_hashes),distinct_seeds=len(seeds))
    (ROOT/'stats37.json').write_text(json.dumps(out,indent=2,default=finite,allow_nan=False)+'\n')
if __name__=='__main__':process()
