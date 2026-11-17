"""Retrospective attribution of sign errors, using only archived measurements and predictions."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

KAPPA = 2.0 / 3.0
R = .020
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
V73 = ROOT / "racket_video_v73/gpu/control"
SOURCE = ROOT / "racket_transfer_v70/data/upstream"

def avg(a):
    return float(np.mean(a))

def count(a):
    return int(np.count_nonzero(a))

def sign(x):
    return x > 0

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read_target(part):
    with (SOURCE / part / 'racket_10.csv').open() as f:
        rows=list(csv.DictReader(f))
    def states(prefix, spin):
        return np.array([[float(r[prefix+'x']),float(r[prefix+'z']),R*float(r[spin+'y'])] for r in rows])
    ids=[f"{int(r['racket_angle'])}/{int(r['traj_dir']):03d}" for r in rows]
    return states('v_in_','spin_before_'),states('v_out_','spin_after_'),ids

def release(states):
    return states[...,0]-states[...,2]

def summarise(x,y,p):
    real=release(y)
    sin=release(x)
    pred=release(p)
    pred_u=p[...,0]
    pred_q=p[...,2]
    eu=pred_u-y[:,0]
    eq=pred_q-y[:,2]
    ez=p[...,1]-y[:,1]
    wrong=sign(pred)!=sign(real)
    # The two repairs give the measured value to one component, leaving the other frozen.
    fix_u=sign(y[:,0]-pred_q)==sign(real)
    fix_q=sign(pred_u-y[:,2])==sign(real)
    true_u=(x[:,0]-y[:,0])/sin
    true_q=KAPPA*(y[:,2]-x[:,2])/sin
    factor=1+1/KAPPA
    # An effective alpha fits measured release exactly, even if measured momentum is inconsistent.
    true_a=(1-real/sin)/factor
    est_a=(x[:,0]-pred_u)/sin
    np.testing.assert_allclose((1-factor*est_a)*sin,pred,atol=2e-14)
    false_cross = (est_a>1/factor)!=(true_a>1/factor)
    ties = (abs(sin)<1e-10) | (abs(real)<1e-10)
    assert not np.any(ties)
    assert np.array_equal(false_cross,wrong)
    split_repair=dict(u_only=count(wrong & fix_u & ~fix_q),q_only=count(wrong & ~fix_u & fix_q),
                      either=count(wrong & fix_u & fix_q),neither=count(wrong & ~fix_u & ~fix_q))
    assert sum(split_repair.values()) == count(wrong)
    product=eu*eq
    m1=avg(eu**2); m2=avg(eq**2); cross=-2*avg(product)
    assert abs(m1+m2+cross-avg((pred-real)**2))<1e-10
    inv=(y[:,0]-x[:,0])+KAPPA*(y[:,2]-x[:,2])
    return dict(impacts=len(real),draws=len(p),pairs=wrong.size,mistakes=count(wrong),accuracy=avg(~wrong),
                correct_after_u_repair=avg(fix_u),correct_after_rim_repair=avg(fix_q),
                wrong_repaired_by=split_repair,
                correct_made_wrong_u_repair=count(~wrong & ~fix_u),correct_made_wrong_rim_repair=count(~wrong & ~fix_q),
                velocity_error=avg(np.hypot(eu,ez)),spin_error_rads=avg(abs(eq))/R,
                abs_err_u=avg(abs(eu)),abs_err_normal=avg(abs(ez)),abs_err_rim=avg(abs(eq)),release_mae=avg(abs(pred-real)),
                signed_err_u=avg(eu),signed_err_rim=avg(eq),mean_product_errors=avg(product),
                release_mse_attribution=dict(tangent=m1,rim=m2,cross=cross,total=avg((pred-real)**2)),
                measured_release_abs=avg(abs(real)),predicted_release_abs=avg(abs(pred)),pred_within_margin=avg(abs(pred)<=.2),
                median_abs_incident_release=float(np.median(abs(sin))),
                mean_abs_momentum_residual=avg(abs(inv)),mean_abs_alpha_u_minus_alpha_rim=avg(abs(true_u-true_q)),
                outside_feasible_alpha=count((true_a<0)|(true_a>.8)),observed_alpha_above=count(true_a>.4),
                pred_alpha_above=avg(est_a>.4),observed_close_to_boundary_002=count(abs(true_a-.4)<=.02),
                observed_alpha_q25_50_75=np.percentile(true_a,[25,50,75]).tolist(),
                mean_abs_alpha_distance_to_boundary=avg(abs(true_a-.4)),
                predicted_abs_alpha_distance_to_boundary=avg(abs(est_a-.4)),
                mean_abs_alpha_error=avg(abs(est_a-true_a)),
                conditional_true_alpha_above=avg((est_a>.4)[:,true_a>.4]) if (true_a>.4).any() else None,
                conditional_true_alpha_below=avg((est_a<=.4)[:,true_a<=.4]) if (true_a<=.4).any() else None)

def thresholds(x,y,mask):
    # Every binary reversal rule allowed by affine alpha(|s_in|) has <=1 transition on ordered |s_in|.
    idx=np.flatnonzero(mask); state=abs(release(x[idx])); tgt=sign(release(y[idx])) != sign(release(x[idx]))
    s=np.unique(state); cuts=np.r_[-np.inf,(s[1:]+s[:-1])/2,np.inf]
    scores=[]
    for ge in (True,False):
        candidate=(state[:,None]>cuts) if ge else (state[:,None]<=cuts)
        for j,n in enumerate(np.sum(candidate==tgt[:,None],axis=0)):
            scores.append((int(n),ge,j))
    best=max(a[0] for a in scores)
    opts=[(ge,cuts[j]) for n,ge,j in scores if n==best]
    return opts,best,np.sum(tgt)

def apply_threshold(x,y,mask,ge,cut):
    s=release(x[mask])
    reverse=abs(s)>cut if ge else abs(s)<=cut
    return count((sign(s) ^ reverse)==sign(release(y[mask])))

def run():
    metadata=json.loads((V73/'run.json').read_text())
    cutoff=metadata['fast_cutoff_mps']; n=metadata['target_held']
    tape=np.load(V73/'splits.npz'); bank=np.load(V73/'predictions.npz')
    x,y,ids=read_target('test'); assert ids==tape['target_ids'][:n].tolist()
    xx,yy,trainids=read_target('train'); assert trainids==tape['calibration_ids'].tolist()
    np.testing.assert_allclose(bank['target'][:n],y);np.testing.assert_allclose(bank['incident'][:n],x)
    sig=release(y); fast=abs(x[:,1])>=cutoff; elig=abs(sig)>.2
    groups={'fast':elig & fast,'other':elig & ~fast,
            'fast_margin_020_to_050':fast & elig & (abs(sig)<=.5),
            'fast_margin_above_050':fast & (abs(sig)>.5),
            'fast_positive':elig & fast & sign(sig),'fast_negative':elig & fast & ~sign(sig)}
    fits={k:bank[f'elastic_impulse_k{k}'][:,:n,:] for k in (4,8)}
    bi={}
    for g,keep in groups.items():
        pairs={}
        for k,p in fits.items():
            pairs[str(k)]=summarise(x[keep],y[keep],p[:,keep])
        z4=sign(release(fits[4][:,keep]))==sign(sig[keep])
        z8=sign(release(fits[8][:,keep]))==sign(sig[keep])
        pairs['k4_to_k8']=dict(correct_to_wrong=count(z4 & ~z8),wrong_to_correct=count(~z4 & z8),
                                unchanged=count(z4==z8),at_least_one_flip_impacts=count(np.any(z4!=z8,axis=0)))
        bi[g]=pairs
    thresholds_info={}
    for name,a,b,mask in [('best_held_fast',x,y,groups['fast']),
                         ('all_training',xx,yy,abs(release(yy))>.2),
                         ('slower_training',xx,yy,(abs(release(yy))>.2)&(abs(xx[:,1])<cutoff)),
                         ('fast_training',xx,yy,(abs(release(yy))>.2)&(abs(xx[:,1])>=cutoff))]:
        opts,train_hits,reverse=thresholds(a,b,mask)
        held=[apply_threshold(x,y,groups['fast'],ge,t) for ge,t in opts]
        clean=[{'reverse_if_abs_slip_higher':ge,'threshold':float(cut) if np.isfinite(cut) else str(cut)} for ge,cut in opts]
        thresholds_info[name]={'training_eligible':count(mask),'best_hits':train_hits,'observed_reverse':int(reverse),
                               'tied_rules':clean,'held_hits_across_tied_rules':[min(held),max(held)],'held_n':count(groups['fast'])}
    # Direct estimate of K4->K8 vector gain due to each component, allowing norm interactions.
    VE={}
    p4,p8=fits[4],fits[8]
    for name,mask in [('fast_all',fast),('fast_resolved',groups['fast']),('other_all',~fast)]:
        err4=(p4-y)[:,mask,:];err8=(p8-y)[:,mask,:]
        R44=avg(np.hypot(err4[...,0],err4[...,1]));R88=avg(np.hypot(err8[...,0],err8[...,1]))
        R48=avg(np.hypot(err4[...,0],err8[...,1]));R84=avg(np.hypot(err8[...,0],err4[...,1]))
        VE[name]={'k4_error':R44,'k8_error':R88,'reduction':R44-R88,
                  'tangential_share_mps':((R44-R84)+(R48-R88))/2,
                  'normal_share_mps':((R44-R48)+(R84-R88))/2}
    learned={}
    slip=release(x)
    levels=np.column_stack([np.ones(len(slip)),np.abs(slip)])
    for k,p in fits.items():
        coefs=[]; cuts=[]; max_res=0.
        for row in p:
            alpha=(1-release(row)/slip)/2.5
            free=(alpha>1e-8)&(alpha<.8-1e-8)
            assert sum(free)>=2
            coeff=np.linalg.lstsq(levels[free],alpha[free],rcond=None)[0]
            max_res=max(max_res,float(abs(np.clip(levels@coeff,0,.8)-alpha).max()))
            coefs.append(coeff)
            cuts.append((.4-coeff[0])/coeff[1])
        assert max_res<1e-8
        coefs=np.array(coefs);cuts=np.array(cuts)
        learned[k]={'decreasing_slope_draws':count(coefs[:,1]<0),'draws':len(coefs),
                    'decreasing_cutoff_quantiles_10_50_90':np.percentile(cuts[coefs[:,1]<0],[10,50,90]).tolist() if (coefs[:,1]<0).any() else None,
                    'opposite_slope_draws':count(coefs[:,1]>0),'max_reconstruction_error':max_res,
                    'within_resolved_fast_abs_slip_range':count((cuts>=abs(slip[groups['fast']]).min())&(cuts<=abs(slip[groups['fast']]).max()))}
    paid={}
    for k in (4,8):
        c=tape['balanced'][:,:k]; v=abs(release(yy)[c])>.2; f=abs(xx[c,1])>=cutoff; both=v&f
        rev=sign(release(yy)[c])!=sign(release(xx)[c])
        counts=np.sum(both,axis=1)
        paid[k]={'fast_paid':count(f)//len(c),'fast_resolved_hist':np.bincount(counts,minlength=k//2+1).tolist(),
                 'fast_with_both_reversal_classes':count((both & rev).any(axis=1)&(both & ~rev).any(axis=1))}
    ret={'specimen':'A-B-S II soft, existing Gossard control', 'frame':'racket local, [vx,vz,R*omega_y] m/s',
         'changes':'Paired within observed impacts and frozen paid subsets. No new acquisition or intervention.',
         'input_sha256':{p.name:sha(p) for p in (V73/'run.json',V73/'predictions.npz',V73/'splits.npz')},
         'sweep':bi,'single_threshold_diagnostic':thresholds_info,'velocity_shapley':VE,'paid':paid,'fitted_rule':learned}
    (OUT/'results.json').write_text(json.dumps(ret,indent=2,allow_nan=False)+'\n')
    rows=[]
    for j in np.flatnonzero(groups['fast']):
        s=release(x[j]); z=release(y[j]); alp=(1-z/s)/2.5
        vals=dict(id=ids[j],s_in=s,s_observed=z,abs_s_in=abs(s),alpha_eff=alp,measured_reverse=bool(s*z<0))
        for k,p in fits.items():
            vals[f'accuracy_K{k}']=avg(sign(release(p[:,j]))==sign(z));vals[f'avg_alpha_K{k}']=avg((1-release(p[:,j])/s)/2.5)
        rows.append(vals)
    with (OUT/'fast_impacts.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(sorted(rows,key=lambda r:r['abs_s_in']))
if __name__=='__main__':run()
