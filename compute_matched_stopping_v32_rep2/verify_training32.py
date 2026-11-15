"""Independent NumPy audit of Coulomb probe responses, allocation quotas and realized Brier gains."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results_train'


def stream(seed, lane):
    return np.random.default_rng(np.random.SeedSequence([int(seed),270000,lane]))


def derive_velocity(seed, friction_coupling, incoming):
    draws = stream(seed,10).random((1024,9))
    wide = (draws[:,1] >= .5).astype(float)
    flips = np.random.default_rng(np.random.SeedSequence([int(seed),270520])).random((1024,2))
    reverse = flips[:,1] < (1-friction_coupling)/2
    slope = .07*(draws[:,7]-.5)
    friction = .251 + (2*(draws[:,3]>=.5)-1)*(.011+.127*wide) + .015*(draws[:,5]-.5)
    restitution = .70+.095*(2*wide-1)+.18*(draws[:,8]-.5)-.19*(2*wide-1)*reverse
    normal = np.sqrt(1+slope**2)
    tangential = (slope+incoming)/normal
    jn = (1+restitution)*np.maximum(0,(1-slope*incoming)/normal)
    jt = np.sign(tangential)*np.minimum(abs(tangential),friction*jn)
    return np.stack([1-(jn+jt*slope)/normal, incoming+(jn*slope-jt)/normal],axis=1)


def audit():
    cfg = json.loads((ROOT/'training_plan.json').read_text())
    lock = json.loads((OUT/'model_lock.json').read_text())
    guard = json.loads((ROOT/'training_worker/results/worker_exit.json').read_text())
    assert guard['foreign']==[] and guard['code']==0
    arms={};seed_catalog=set()
    for arm,c in cfg['arms'].items():
        features=[];ys=[];env_times=[];n=0;costs=0;max_sensor_err=0.;max_gain_err=0.;max_feature_err=0.;zeros=np.zeros(2,int)
        elapsed=0.;overlaps=np.zeros(2,int);cells=0
        for folder in sorted((OUT/arm).glob('block_*/meta.json')):
            md=json.loads(folder.read_text());block=md['block'];base=c['base_seed']+10000*block
            f=folder.parent/'data.npz'; sha=hashlib.sha256(f.read_bytes()).hexdigest()
            assert md['base_seed']==base and md['data_sha256']==sha
            assert lock['training'][arm]['blocks'][block]['data_sha256']==sha
            assert md['bank_pair']==[base+8901,base+8902]
            assert set(md['bank_pair']).isdisjoint(seed_catalog)
            seed_catalog.update(md['bank_pair']);elapsed+=md['elapsed_s']
            with np.load(f) as z:
                x=z['features'];gain=z['gain'];action=z['action'];cost=z['cost'];times=z['env_elapsed_s']
                ecount=len(action);F,B=c['futures'],c['batches']
                assert action.shape==(ecount,2,1+F,B,128) and cost.shape==(ecount,B,128,2)
                assert np.isin(action,[1,3]).all() and ((action==3).sum(-1)==13).all()
                assert np.isfinite(cost).all() and cost.min()>=0 and cost.max()<=1+1e-12
                assert np.all(times>0) and times.sum()<=md['elapsed_s']
                replay=np.where(action==3,cost[:,None,None,:,:,1],cost[:,None,None,:,:,0]).mean(-1)
                ref=(replay[:,:,0:1]-replay[:,:,1:]).swapaxes(0,1)
                max_gain_err=max(max_gain_err,float(np.max(abs(ref-gain))))
                zeros+=(gain==0).sum(axis=(1,2,3));cells+=ecount*F*B
                alphas=np.array(cfg['alpha_panel'])[np.random.default_rng(base+8903).permutation(8)[:ecount]]
                assert np.array_equal(alphas,z['alpha'])
                for e in range(ecount):
                    home=base+1000*e; seed=home+1
                    expected_ids=stream(seed,98).permutation(1024)[:4];sig=derive_velocity(seed,alphas[e,1],-1.1)
                    assert np.array_equal(expected_ids,z['prefix_ids'][e]) and z['prefix_seed'][e]==seed
                    yy=sig[expected_ids]+.027*stream(seed,99).normal(size=(8,2))[:4]
                    max_sensor_err=max(max_sensor_err,float(np.max(abs(yy-z['prefix_y'][e]))))
                    factual_v=derive_velocity(seed,alphas[e,1],0.)[:,0]+.095*stream(seed,11).normal(size=(1024,11))[:,-2]
                    local=[seed,*z['deployment_seeds'][e].tolist()]
                    assert np.array_equal(np.arange(B)+home+300,z['deployment_seeds'][e])
                    assert hashlib.sha256(action[e].tobytes()).hexdigest()==z['selection_sha256'][e]
                    for s,k in enumerate((2,4)):
                        v=factual_v[expected_ids[:k]];normal,tan=yy[:k].T
                        ac,af,sc,sf=x[s,e,:4]
                        ref_feat=[abs(ac),abs(af),ac*af,ac*abs(af),af*abs(ac),sf*ac,tan.mean(),tan.std(),normal.mean(),normal.std(),np.mean((v-v.mean())*(tan-tan.mean()))]
                        max_feature_err=max(max_feature_err,float(np.max(abs(x[s,e,4:]-ref_feat))))
                        avail=np.setdiff1d(np.arange(1024),expected_ids[:k])
                        ids=z[f'ids_{k}'][e];overlaps[s]+=ids.size-len(set(ids.ravel()))
                        for j in range(F):
                            future=home+100+20*s+j;local.append(future)
                            ref_ids=stream(future,81).choice(avail,size=k,replace=False)
                            assert np.array_equal(ref_ids,ids[j]) and len(set(ids[j]))==k
                            expected_y=sig[ref_ids]+.027*stream(future,82).normal(size=(k,2))
                            max_sensor_err=max(max_sensor_err,float(np.max(abs(expected_y-z[f'y_{k}'][e,j]))))
                    assert set(local).isdisjoint(seed_catalog) and len(set(local))==len(local)
                    seed_catalog.update(local)
                features.append(x);ys.append(gain.mean(axis=(2,3)));env_times.extend(times);n+=ecount;costs+=cost.size
        assert n==lock['training'][arm]['n_environments'] and max(max_sensor_err,max_feature_err,max_gain_err)<1e-12
        assert np.isclose(elapsed,lock['training'][arm]['labels_s'],rtol=0,atol=1e-10)
        y=np.concatenate(ys,axis=1);x=np.concatenate(features,axis=1)
        assert x.shape==(2,n,15) and y.shape==(2,n)
        arms[arm]=dict(n_prefixes=n,n_stages=n*2,n_cost_entries=costs,training_label_cells=cells*2,
                       exact_zero_gain_fraction_by_stage=(zeros/cells).tolist(),label_sd_by_stage=y.std(axis=1).tolist(),
                       elapsed_s=elapsed,env_time_median_s=float(np.median(env_times)),env_time_p90_s=float(np.quantile(env_times,.9)),
                       max_replay_sensor_error=max_sensor_err,max_brier_error=max_gain_err,max_feature_error=max_feature_err,
                       repeated_ids_across_futures=overlaps.tolist())
    wall={a:lock['training'][a]['labels_plus_fit_s'] for a in arms}
    result=dict(arms=arms,unique_training_seeds=len(seed_catalog),guard=guard,
                training_time_imbalance_fraction=(max(wall.values())/min(wall.values())-1),
                model_sha256=hashlib.sha256((OUT/'model_lock.json').read_bytes()).hexdigest())
    (OUT/'training_audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result))
if __name__=='__main__':
    audit()
