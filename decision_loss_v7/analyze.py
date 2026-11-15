"""Post-policy-freeze evaluation. NumPy only: safe to run on the local mirror."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
from config import ACTIONS,COSTS,N_WORLDS,AUDIT,RASTER,CONDITIONS
from decision import load,stability
R=Path('results');OUT=R/'analysis';OUT.mkdir(exist_ok=True)
BOOT=np.random.default_rng(880001).integers(N_WORLDS,size=(50000,N_WORLDS))

def serial(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 raise TypeError(type(x))
def dump(name,x): (OUT/name).write_text(json.dumps(x,default=serial,indent=2,allow_nan=False))
def interval(x,seed=880002):
 x=np.asarray(x,float);x=x[np.isfinite(x)]
 if not len(x):return dict(n_worlds=0,mean=None,ci95=None,ci_bonferroni12=None)
 ix=BOOT if len(x)==N_WORLDS else np.random.default_rng(seed).integers(len(x),size=(50000,len(x)))
 b=x[ix].mean(1)
 return dict(n_worlds=len(x),mean=float(x.mean()),ci95=np.quantile(b,[.025,.975]).tolist(),ci_bonferroni12=np.quantile(b,[.05/24,1-.05/24]).tolist())
def take(loss,choice):return np.take_along_axis(loss,np.asarray(choice)[None,:,None],axis=2)[:,:,0]
def condition(precision,action):return f'{precision if action=="view" else "native"}_{action}'
def outcomes(case,precision,phase='eval',decoder='full_0'):
 world=R/'worlds'/f'case_{case:03d}'
 truth=np.array(json.loads((world/'private/truth.json').read_text())['velocity'])
 folder=R/phase/world.name
 means=np.stack([np.load(folder/f'{condition(precision,a)}_{decoder}.npz')['mean'] for a in ACTIONS],-1)
 return ((means-truth[None,:,:,None])**2).mean(2),means

def verify():
 assert json.loads((R/'validation/smoke.json').read_text())['passed']
 assert json.loads((R/'validation/gaussian.json').read_text())['passed']
 plan=R/'plans/final_policy.json';frozen=plan.stat().st_mtime
 phases={'worlds':list(range(48)),'acq256':list(range(48)),'acq512':json.loads((R/'plans/refine512.json').read_text())['cases'],'full_audit':AUDIT,'eval':list(range(48)),'decoder_audit':AUDIT,'raster_eval':RASTER}
 checks=[];nt=0;miness=1.;maxbeta=0.;rollouts=0.;seconds=0.;populations=0;seed_batches={}
 budgets={'acq256':(256,1024,12,2,'bridge'),'acq512':(512,1024,12,2,'bridge'),'full_audit':(128,2048,18,2,'full'),'eval':(32,2048,18,1,'full'),'decoder_audit':(32,4096,18,1,'full'),'raster_eval':(8,2048,18,1,'full')}
 for phase,cases in phases.items():
  for c in cases:
   d=R/phase/f'case_{c:03d}';assert (d/'complete.json').exists(),str(d)
   if phase in ['eval','decoder_audit','raster_eval']:assert (d/'complete.json').stat().st_mtime>frozen,'policy freeze chronology'
   if phase in budgets:
    hh,nn,mm,rr,method=budgets[phase];done=json.loads((d/'complete.json').read_text())
    assert (done['h'],done['n'],done['moves'],done['reps'],done['method'])==(hh,nn,mm,rr,method)
    conditions=[x for x in CONDITIONS if phase!='raster_eval' or x[0]=='native']
    for precision,action in conditions:
     for rep in range(rr):
      for ext in ['npz','json']:assert (d/f'{precision}_{action}_{method}_{rep}.{ext}').exists()

   for p in d.glob('*_full_*.json') if phase in ['full_audit','eval','decoder_audit','raster_eval'] else d.glob('*_bridge_*.json'):
    st=json.loads(p.read_text());assert np.allclose(st['final_beta'],1,atol=1e-10)
    assert (st['h'],st['n'],st['moves'])==budgets[phase][:3] and len(st['final_beta'])==st['h']
    batchkey=(phase,tuple(st['batch_cases']),st['precision'],st['action'],st['rep'],st['h'],st['n'],st['moves'])
    assert st['seed'] not in seed_batches or seed_batches[st['seed']]==batchkey,'accidental seed collision'
    seed_batches[st['seed']]=batchkey

    ratio=min(st['min_stage_ess'])/st['n'];assert ratio>=.7499
    miness=min(miness,ratio);maxbeta=max(maxbeta,max(abs(np.array(st['final_beta'])-1)))
    rollouts+=st['trajectory_rollouts'];seconds+=st['seconds'];populations+=st['h'];nt+=1
   for p in d.glob('*.npz'):
    with np.load(p) as data:
     for k in data.files:assert np.isfinite(data[k]).all(),(p,k)
  checks.append(dict(phase=phase,completed=len(cases)))
 truth_hashes=[]
 for c in range(48):
  d=R/'worlds'/f'case_{c:03d}';p=json.loads((d/'public.json').read_text());assert p['world_seed']==202611110700+c
  truth_hashes.append(hashlib.sha256((d/'private/truth.json').read_bytes()).hexdigest())
  with np.load(d/'bank.npz') as b,np.load(d/'generator.npz') as g:
   assert b['x'].shape==(16384,5) and g['x'].shape==(8192,5) and not np.array_equal(b['x'][:8192],g['x'])
  with np.load(d/'outer.npz') as a,np.load(d/'private/gaussian.npz') as b:
   assert all(len(a[k])==512 and len(b[k])==32 for k in a.files)
 assert len(set(truth_hashes))==48
 return dict(passed=True,phases=checks,conditional_populations=populations,conditional_result_files=nt,unique_conditional_batches=len(seed_batches),distinct_batch_seeds=True,min_stage_ess_ratio=miness,max_final_beta_error=maxbeta,
  conditional_trajectory_rollouts=rollouts,conditional_gpu_seconds=seconds,policy_frozen_before_actual_evaluation=True,unique_new_worlds=48)

def main():
 integrity=verify();dump('integrity.json',integrity)
 plan=json.loads((R/'plans/final_policy.json').read_text())['rows'];assert [p['case'] for p in plan]==list(range(48))
 summary=dict(integrity=integrity,comparisons=[],selection_counts=[],ranking=[],policies=[],noise_prefix=[],ambiguity=[],reference=[],decoder_audit=[],raster=[],decoder_ablation=[],cost=[],ess=[],collapse_diagnostic=[],families=[],ess_groups=[],influence=[],counterexamples=[])
 raw=[];classraw=[];npz={}
 for precision in ['native','precise']:
  loss=np.stack([outcomes(c,precision)[0] for c in range(48)]) # W,R,Q,A
  acqs={m:[] for m in ['bridge','snis4096','snis16384']};ess=[];banktime=[];bridgecost=[]
  for c in range(48):
   folder=R/plan[c]['phase']/f'case_{c:03d}'
   for m in acqs:acqs[m].append(load(folder,precision,m))
   ess.append({m:acqs[m][-1]['ess'] for m in ['snis4096','snis16384']})
   t={}
   for m in ['snis4096','snis16384']:
    t[m]=sum(float(np.load(folder/f'{condition(precision,a)}_{m}.npz')['seconds']) for a in ACTIONS)
   t['bridge']=sum(json.loads((folder/f'{condition(precision,a)}_bridge_{rep}.json').read_text())['seconds'] for a in ACTIONS for rep in [0,1]);banktime.append(t)
  for m in ['snis4096','snis16384']:
   e=np.concatenate([x[m][:,0] for x in ess]);summary['ess'].append(dict(precision=precision,method=m,median=float(np.median(e)),p05=float(np.quantile(e,.05)),min=float(e.min()),fraction_lt20=float((e<20).mean()),n_hypotheses=len(e)))
  for m in ['snis4096','snis16384']:
   es=[];vv=[]
   for c in AUDIT:
    folder=R/plan[c]['phase']/f'case_{c:03d}';ref=R/'full_audit'/folder.name
    es.extend(np.load(folder/f'{precision}_view_{m}.npz')['ess'][:128])
    vv.extend(np.mean([np.load(ref/f'{precision}_view_full_{k}.npz')['variance'] for k in [0,1]],0).mean((1,2)))
   es=np.array(es);vv=np.array(vv);collapsed=es<=1.01
   summary['collapse_diagnostic'].append(dict(precision=precision,method=m,n=len(es),n_collapsed=int(collapsed.sum()),reference_variance_median_collapsed=float(np.median(vv[collapsed])) if collapsed.any() else None,scope='post-hoc diagnostic of existing hypothetical observations; not a new primary subgroup test'))
  summary['cost'].append(dict(precision=precision,scope='final selected H only; excludes factual banks, discarded H256, summaries, planning, evaluation, serialization overhead beyond individual timers; not end-to-end fair compute matching',seconds={m:sum(t[m] for t in banktime) for m in ['bridge','snis4096','snis16384']}))
  noobs=[]
  for c in range(48):
   d=R/'worlds'/f'case_{c:03d}';truth=np.array(json.loads((d/'private/truth.json').read_text())['velocity']);current=np.array(json.loads((d/'public.json').read_text())['current']);noobs.append(((truth-current)**2).mean(1))
  for label,cost in COSTS.items():
   choices={m:np.array([p['conditions'][precision][label]['bridge']['choice'] if m=='bridge' else p['conditions'][precision][label]['snis_choices'][m] for p in plan]) for m in acqs}
   for m in acqs:
    recalculated=np.array([(d['samples'].mean(0)-cost).argmax(1) for d in acqs[m]]);assert np.array_equal(recalculated,choices[m]),'frozen choice changed'
   bsrows=[p['conditions'][precision][label]['bridge'] for p in plan]
   active=np.array([b['active'] for b in bsrows]);practical=np.array([b['practical'] for b in bsrows]);decisive=np.array([b['decisive'] for b in bsrows]);half=np.array([b['half_agreement'] for b in bsrows])
   summary['ranking'].append(dict(precision=precision,cost=label,n_active=int(active.sum()),n_practical_active=int((practical&active).sum()),n_decisive_active=int((decisive&active).sum()),half_agreement_active=float(half[active].mean()) if active.any() else None))
   for m in acqs:summary['selection_counts'].append(dict(precision=precision,cost=label,method=m,all_counts=np.bincount(choices[m].ravel(),minlength=4).tolist(),active_counts=np.bincount(choices[m][active],minlength=4).tolist(),n_active=int(active.sum())))
   choices['always_view']=np.zeros((48,6),int)
   choices['probe_only_bridge']=np.array([(d['samples'].mean(0)[:,1:]-np.array(cost)[1:]).argmax(1)+1 for d in acqs['bridge']])
   total=loss+np.array(cost);values={m:np.stack([take(total[c],choice[c]) for c in range(48)]) for m,choice in choices.items()}
   values['random']=total.mean(-1)
   values['no_observation']=np.repeat(np.array(noobs)[:,None,:],32,axis=1)
   empirical=total.mean(1).argmin(2);values['empirical_oracle']=np.stack([take(total[c],empirical[c]) for c in range(48)])
   cv=[]
   for c in range(48):
    a=total[c,:16].mean(0).argmin(1);b=total[c,16:].mean(0).argmin(1)
    cv.append(np.concatenate([take(total[c,:16],b),take(total[c,16:],a)]))
   values['crossfit_oracle']=np.stack(cv)
   for m,v in values.items():summary['policies'].append(dict(precision=precision,cost=label,policy=m,**interval(v.mean((1,2)))))
   for m in ['snis4096','snis16384']:
    delta=values['bridge']-values[m];wm=delta.mean((1,2));summary['comparisons'].append(dict(precision=precision,cost=label,baseline=m,**interval(wm),world_sem=float(wm.std(ddof=1)/np.sqrt(48)),repeat_sem=float(np.sqrt(delta.mean(2).var(1,ddof=1).sum()/32/48**2)),model_predicted_gain=float(np.mean([(acqs['bridge'][c]['samples'].mean(0)-cost).max(1).mean()-(acqs['bridge'][c]['samples'].mean(0)-cost)[np.arange(6),choices[m][c]].mean() for c in range(48)])),relative_reduction=float(-wm.mean()/values[m].mean()),n_changed_queries=int((choices['bridge']!=choices[m]).sum()),n_active_queries=int(active.sum()),n_changed_active_queries=int(((choices['bridge']!=choices[m])&active).sum()),n_changed_worlds=int((choices['bridge']!=choices[m]).any(1).sum())))
    for count in [1,4,8,16,32]:
     d=delta[:,:count].mean((1,2));sem=None if count==1 else float(np.sqrt(delta[:,:count].mean(2).var(1,ddof=1).sum()/count/48**2))
     summary['noise_prefix'].append(dict(precision=precision,cost=label,baseline=m,repeats=count,**interval(d),conditional_noise_sem=sem))
    for family in range(3):summary['families'].append(dict(precision=precision,cost=label,baseline=m,family=['corner','gate','deflection'][family],bridge_loss=float(values['bridge'][16*family:16*(family+1)].mean()),baseline_loss=float(values[m][16*family:16*(family+1)].mean()),**interval(wm[16*family:16*(family+1)])))
    for tag,mask in [('view_ESS_lt20',np.array([np.median(e[m][:,0])<20 for e in ess])),('view_ESS_ge20',np.array([np.median(e[m][:,0])>=20 for e in ess]))]:
     summary['ess_groups'].append(dict(precision=precision,cost=label,baseline=m,group=tag,**interval(wm[mask])))
    loo=(wm.sum()-wm)/47
    summary['influence'].append(dict(precision=precision,cost=label,baseline=m,loo_min=float(loo.min()),loo_max=float(loo.max()),world_min=int(wm.argmin()),world_max=int(wm.argmax()),min_contribution=float(wm.min()/48),max_contribution=float(wm.max()/48)))
    npz[f'{precision}_{label}_bridge_minus_{m}']=delta
    for c in range(48):
     for q in range(6):raw.append(dict(case=c,family=['corner','gate','deflection'][c//16],query=q,precision=precision,cost=label,baseline=m,bridge_action=ACTIONS[choices['bridge'][c,q]],baseline_action=ACTIONS[choices[m][c,q]],mean_difference=float(delta[c,:,q].mean()),noise_sd=float(delta[c,:,q].std(ddof=1))))
   # Stability classes depend only on acquisition, never actual outcomes.
   masks={k:np.zeros((48,6),bool) for k in ['inactive','bridge_unresolved','snis_unresolved','stable_agreement','stable_near_tie_disagreement','stable_material_disagreement']}
   for c in range(48):
    bs=plan[c]['conditions'][precision][label]['bridge'];ss=stability(acqs['snis4096'][c]['samples'],cost,770000+c)
    net=acqs['bridge'][c]['samples'].mean(0)-cost;bg=net.max(1)-net[np.arange(6),choices['snis4096'][c]]
    for q in range(6):
     if not bs['active'][q]:cl='inactive'
     elif not bs['practical'][q]:cl='bridge_unresolved'
     elif not ss['practical'][q]:cl='snis_unresolved'
     elif choices['bridge'][c,q]==choices['snis4096'][c,q]:cl='stable_agreement'
     elif bg[q]<=bs['tolerance'][q]:cl='stable_near_tie_disagreement'
     else:cl='stable_material_disagreement'
     masks[cl][c,q]=True
     classraw.append(dict(case=c,query=q,precision=precision,cost=label,category=cl,bridge_practical=bs['practical'][q],bridge_decisive=bs['decisive'][q],bridge_winner_probability=bs['winner_probability'][q],bridge_upper_regret=bs['upper_regret'][q],tolerance=bs['tolerance'][q],snis_practical=ss['practical'][q],bridge_net_gap_to_snis=float(bg[q])))
   delta=(values['bridge']-values['snis4096']).mean(1)
   for cl,mask in masks.items():
    eligible=mask.sum(1)>0;v=(delta*mask).sum(1)[eligible]/mask.sum(1)[eligible]
    summary['ambiguity'].append(dict(precision=precision,cost=label,category=cl,n_queries=int(mask.sum()),overall_contribution=float((delta*mask).mean()),contribution_ci95=interval((delta*mask).mean(1))['ci95'],**interval(v)))
   # Shared-decoder audit, with the selection frozen.
   audit_losses=np.stack([outcomes(c,precision,'decoder_audit')[0] for c in AUDIT])+cost
   old=total[AUDIT]
   for m in ['snis4096','snis16384']:
    newdelta=np.array([(take(audit_losses[i],choices['bridge'][c])-take(audit_losses[i],choices[m][c])).mean() for i,c in enumerate(AUDIT)])
    olddelta=np.array([(take(old[i],choices['bridge'][c])-take(old[i],choices[m][c])).mean() for i,c in enumerate(AUDIT)])
    pred_rmse=float(np.sqrt(np.mean([np.mean((outcomes(c,precision,'decoder_audit')[1]-outcomes(c,precision)[1])**2) for c in AUDIT])))
    summary['decoder_audit'].append(dict(precision=precision,cost=label,baseline=m,main_same_subset=interval(olddelta),audit4096=interval(newdelta),prediction_rmse=pred_rmse))
   for decoder in ['snis4096','snis16384']:
    other=np.stack([outcomes(c,precision,decoder=decoder)[0] for c in range(48)])+cost
    for selector in ['bridge','snis4096','snis16384']:
     alt=np.array([take(other[c],choices[selector][c]).mean() for c in range(48)])
     summary['decoder_ablation'].append(dict(precision=precision,cost=label,selector=selector,decoder=decoder,loss=interval(alt),minus_common_full=interval(alt-values[selector].mean((1,2)))))
   if precision=='native':
    ras=np.stack([outcomes(c,precision,'raster_eval')[0] for c in RASTER])+cost
    for m in ['snis4096','snis16384']:
     dd=np.array([(take(ras[i],choices['bridge'][c])-take(ras[i],choices[m][c])).mean() for i,c in enumerate(RASTER)])
     gauss=np.array([(values['bridge'][c]-values[m][c]).mean() for c in RASTER])
     summary['raster'].append(dict(cost=label,baseline=m,raster=interval(dd),gaussian_same_subset=interval(gauss)))
   # Reference audit uses the SAME first 128 hypothetical observations.
   for m in ['bridge','snis4096','snis16384']:
    agreement=[];regret=[];rmse=[];utilityerr=[];ref_active=[]
    for c in AUDIT:
     d=load(R/plan[c]['phase']/f'case_{c:03d}',precision,m,prefix=128);ref=load(R/'full_audit'/f'case_{c:03d}',precision,'full')
     gu=ref['samples'].mean(0);nu=gu-cost;pick=(d['samples'].mean(0)-cost).argmax(1);aa=gu.max(1)>5e-5
     agreement.extend((pick==nu.argmax(1))[aa]);regret.extend((nu.max(1)-nu[np.arange(6),pick])[aa]);ref_active.extend(aa)
     rmse.append(float(np.sqrt(((d['means']-ref['means'])**2).mean())));utilityerr.append(float(abs(d['samples'].mean(0)-gu).mean()))
    summary['reference'].append(dict(precision=precision,cost=label,method=m,n_active=int(sum(ref_active)),agreement=float(np.mean(agreement)) if len(agreement) else None,reference_regret=float(np.mean(regret)) if len(regret) else None,mean_prediction_rmse=float(np.mean(rmse)),mean_utility_abs_error=float(np.mean(utilityerr))))
  npz[precision+'_actual_action_mse']=loss
 # Descriptive extremes are explicitly post-outcome examples, not confirmatory subgroups.
 subset=[r for r in raw if r['cost']=='equal' and r['baseline']=='snis4096' and r['bridge_action']!=r['baseline_action']]
 summary['counterexamples']=sorted(subset,key=lambda r:r['mean_difference'])[:5]+sorted(subset,key=lambda r:r['mean_difference'],reverse=True)[:5]
 dump('summary.json',summary)
 np.savez_compressed(OUT/'paired_losses.npz',**npz)
 for name,rows in [('query_outcomes.csv',raw),('acquisition_classes.csv',classraw)]:
  with (OUT/name).open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(json.dumps({'passed':True,'comparisons':summary['comparisons']},indent=2))
if __name__=='__main__':main()
