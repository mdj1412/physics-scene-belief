"""Blinded acquisition, paired risk and true empirical inverse CDF on fresh catalogs."""
from pathlib import Path
import hashlib
import json
import os
import sys
import numpy as np

from policy57 import choose, log_odds, MIXTURE
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'mismatch_switch_v56'))
from score56 import cdf_boot, EVENT
METHODS=['4x2_map','8x1_map','fixed_I','4x2_risk','v54_gate','cross_episode_map','cross_episode_risk','random_map_rate','random_risk_rate','known_regime']
C=24
M=len(METHODS)
def load():
    fine=[];ultra=[];real=[];all_routes=[];pilot=[];gate_rates=[];catalog_hashes=[]
    for i in range(C):
        path=ROOT/'scored57'/f'panel_{i:02d}.npz'
        with np.load(path) as f:
            material=((np.arange(4)+i)%4)//2
            for ls,field in ((fine,'exact_cells'),(ultra,'ultra_cells')):
                ls.append(np.stack([f[field][:,:,material==g,:2].mean((1,2)) for g in (0,1)]))
            gate_rates.append([f['gate'][:,material==g].mean() for g in (0,1)])
            actual=f['real'][...,:2]
            assert actual.shape==(2,3,2,5,4,256,2)
            real.append(np.stack([np.take(actual,np.flatnonzero(material==g),axis=4).reshape(2,3,2,-1,2) for g in (0,1)],axis=1))
        pilot_path=ROOT/'pilots'/f'panel_{i:02d}.npz'
        with np.load(pilot_path) as f:
            ref=f['pilot_ref'];imp=f['pilot_imp']
            assert ref.shape==imp.shape==(2,2,2,4096,8)
            assert np.isin(ref,[-1,0,1]).all() and np.isin(imp,[-1,0,1]).all()
            singles=[v[:,:,0] for v in (ref,imp)]
            groups=[v[:,:,1] for v in (ref,imp)]
            one=choose(*singles,'8x1','map')
            std=choose(*groups,'4x2','map')
            risk=choose(*groups,'4x2','risk')
            assert not one.any() and not choose(*singles,'8x1','risk').any()
            broken=[a.copy() for a in groups]
            for a in broken:a[...,1::2]=np.roll(a[...,1::2],1,axis=2)
            false_pair=choose(*broken,'4x2','map')
            false_pair_risk=choose(*broken,'4x2','risk')
            oracle=np.broadcast_to(np.array([1,0],bool)[None,:,None],std.shape)
            rows=np.stack((std,one,np.ones_like(one),risk,one,false_pair,false_pair_risk,one,one,oracle),axis=-1)
            all_routes.append(rows.mean(2))
            valid=(groups[0]!=0)&(groups[1]!=0)
            flips=valid&(groups[0]!=groups[1])
            complete=valid.reshape(*valid.shape[:-1],4,2).all(-1)
            pair_counts=flips.reshape(*valid.shape[:-1],4,2).sum(-1)
            pooled=np.stack([valid.sum((2,3)),flips.sum((2,3)),complete.sum((2,3)),(complete&(pair_counts==2)).sum((2,3)),
                             (complete&(pair_counts==1)).sum((2,3)),(complete&(pair_counts==0)).sum((2,3)),(~complete.any(-1)).sum(2),
                             (log_odds(*groups,'4x2')==0).sum(2)],axis=-1)
            pilot.append(pooled)
        catalog_hashes.append([hashlib.sha256(p.read_bytes()).hexdigest() for p in (path,pilot_path)])
    route=np.array(all_routes)
    for source,rnd in ((0,7),(3,8)):
        route[...,rnd]=route[...,source].mean()
    return tuple(map(np.asarray,(fine,ultra,real,route,pilot,gate_rates,catalog_hashes)))

def regime_risk(loss):
    return np.einsum('cfea,ze->cfza',loss,MIXTURE)
def evaluate(loss,route):
    rr=regime_risk(loss)
    return rr[...,0,None]+route*(rr[...,1]-rr[...,0])[...,None]
def quantile_weights(route,regime_dist):
    # Weights per catalog. Within a catalog average the two materials.
    return np.einsum('mcfza,z,ze,d->mcfeda',np.stack((1-route,route),-1).transpose(3,0,1,2,4),
                     regime_dist,MIXTURE,EVENT)/2
def interval(draws, comparisons=1):
    return np.quantile(draws,[.025/comparisons,1-.025/comparisons],axis=0).T
def pair(point,boots,row,columns):
    return np.column_stack([point[row]-point[columns],interval(boots[:,row,None]-boots[:,columns],len(columns))])
def losssummary(loss,route):
    return evaluate(loss,route).mean((1,2))
def cluster_summary(pm,route,loss,wt):
    means=wt@pm
    # Matched random's reference rate changes with the bootstrapped population.
    rr=regime_risk(loss).mean((1,2))
    base=wt@rr
    for source,rnd in ((0,7),(3,8)):
        propensity=wt@route[...,source].mean((1,2))
        means[:,rnd]=base[:,0]+propensity*(base[:,1]-base[:,0])
    return means
def primary(points,boots):
    return {METHODS[m]:pair(points,boots,m,[1,2]).tolist() for m in (0,3)}
def main():
    assert os.getenv('SLURM_JOB_ID')
    assert hashlib.sha256((ROOT/'policy57.py').read_bytes()).hexdigest()=='680784cdbf913ea44977f898d4dba36f9ab87d3377fa3f4bff9c905fb6e44c36'
    fine,ultra,real,route,stats,gate_rates,hashes=load()
    rg=np.random.default_rng(570804)
    choices=rg.integers(C,size=(5000,C))
    wt=np.stack([(choices==i).mean(1) for i in range(C)],axis=1)
    base=np.full((1,C),1/C);wts=np.vstack([base,wt])
    meanbase=losssummary(fine,route)
    assert np.array_equal(meanbase[:,1],meanbase[:,4])
    means=cluster_summary(meanbase,route,fine,wts)
    urs=cluster_summary(losssummary(ultra,route),route,ultra,wts)
    # Classification and deployment choice are distinct estimands; for A use I, for B gate.
    acc=(route[:,:,0]+1-route[:,:,1]).mean(1)/2
    accuracy=wts@acc
    correct_by_regime=np.stack((route[:,:,0],1-route[:,:,1]),axis=2)
    recf=correct_by_regime.mean((0,1))
    recfci=interval(np.einsum('bc,czm->bzm',wt,correct_by_regime.mean(1))[:,:,0])
    pooled=stats.sum((0,1))
    foot=(pooled[:,3:6]/pooled[:,2,None]).tolist()
    overall=means[0]
    practical={}
    for m,rm,broken in ((0,7,5),(3,8,6)):
        practical[METHODS[m]]=dict(vs_matched_random=pair(overall,means[1:],m,[rm])[0].tolist(),
                                   vs_broken=pair(overall,means[1:],m,[broken])[0].tolist())
    # seed x catalog x family x e x arm, after averaging 128-object realized sessions.
    snapshots=np.einsum('csfeda,d->scfea',real.mean(-2),EVENT)
    realized=[cluster_summary(losssummary(snapshot,route),route,snapshot,wts) for snapshot in snapshots]
    raw=np.sort(real.transpose(1,0,2,3,4,6,5),axis=-1)
    weights=quantile_weights(route,np.array([.5,.5]))
    tails=cdf_boot(raw[0],weights[:4],wts)
    q0=cdf_boot(raw[0],weights,base)[0];q1=cdf_boot(raw[1],weights,base)[0]
    assert np.array_equal(q0[:4],tails[0])
    qreg=[]
    for z in (0,1):
        rz=np.eye(2)[z]
        qreg.append(cdf_boot(raw[0],quantile_weights(route,rz)[:4],base)[0])
    # Independent weighted order statistic on full sessions; confirm the cached CDF search.
    flat=raw[0].ravel()
    order=np.argsort(flat,kind='stable')
    for m in (0,1,2,3):
        weight=np.broadcast_to(weights[m][...,None],raw[0].shape).ravel()[order]/(C*raw.shape[-1])
        i=np.searchsorted(np.cumsum(weight),.9)
        np.testing.assert_allclose(flat[order[i]],q0[m],rtol=0,atol=2e-8)
    per_region=regime_risk(fine)
    best=np.min(per_region,axis=-1).mean()
    full_fixed=np.minimum(overall[1],overall[2])
    result=dict(methods=METHODS,mean=overall,mean_contrast=primary(overall,means[1:]),
                session_p90=q0,repeat_p90=q1,p90_contrast=primary(tails[0],tails[1:]),p90_by_regime=np.array(qreg),
                route_by_regime=route.mean((0,1)),mean_by_regime=evaluate(fine,route).mean((0,1)),
                balanced_accuracy=accuracy[0],accuracy_contrast=pair(accuracy[0],accuracy[1:],0,[1])[0],map_regime_recall=recf[:,0],map_regime_ci=recfci,
                risk_accuracy_contrast=pair(accuracy[0],accuracy[1:],3,[1])[0],
                realized_means=np.stack([x[0] for x in realized]),realized_contrast=[primary(x[0],x[1:]) for x in realized],
                ultra_mean=urs[0],ultra_contrast=primary(urs[0],urs[1:]),cyclic_and_random=practical,
                impact_rate=(gate_rates[:,:,None,None]+route*(1-gate_rates[:,:,None,None])).mean((0,1,2)),
                catalog_wins={METHODS[m]:(meanbase[:,m,None]<meanbase[:,[1,2]]).sum(0) for m in (0,3)},
                family_means=evaluate(fine,route).mean((0,2)),complete_pair_double_one_zero=foot,
                pilot_mismatch=pooled[:,1]/pooled[:,0],mean_complete_pairs=pooled[:,2]/(24*2*4096),no_complete=pooled[:,6]/(24*2*4096),ties=pooled[:,7]/(24*2*4096),
                source_mapping_correct_cells=np.stack((per_region[:,:,0,0]>per_region[:,:,0,1],per_region[:,:,1,0]<per_region[:,:,1,1]),axis=-1).sum((0,1)),
                known_regime_mean=overall[9],material_hindsight=best,
                oracle_recovery={METHODS[m]:(full_fixed-overall[m])/(full_fixed-overall[9]) for m in (0,3)})
    np.savez_compressed(ROOT/'score57_cells.npz',by_e=fine,ultra_by_e=ultra,route=route,balanced_accuracy=acc,
                        methods=METHODS,pilot_stats=stats,realized_by_e=snapshots,base_gate_impact=gate_rates,mean_cells=evaluate(fine,route),
                        boot_mean=means[1:],boot_p90=tails[1:],sha256=hashes)
    (ROOT/'score57.json').write_text(json.dumps(result,indent=2,default=lambda x:np.asarray(x).tolist())+'\n')
    print('mean',overall[:4],'p90',q0[:4],'accuracy',accuracy[0,:4],flush=True)
if __name__=='__main__':main()
