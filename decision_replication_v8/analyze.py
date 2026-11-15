"""Post-freeze analysis only; independent world is the inferential unit."""
import hashlib,json,csv
from pathlib import Path
import numpy as np
ROOT=Path('results');OUT=ROOT/'analysis';OUT.mkdir(exist_ok=True)
N=64;R=32;Q=6
METHODS=['bridge','standard4096','standard_fixed8192','crossfit_fixed8192','standard_fixed16384','crossfit_fixed16384','standard_budget','crossfit_budget']
PRIMARY=['standard4096','crossfit_fixed16384','standard_budget','crossfit_budget']
rng=np.random.default_rng(882711);boots=rng.integers(N,size=(50000,N))
def interval(x,alpha=.05):
 x=np.asarray(x);assert x.shape==(N,)
 return np.quantile(x[boots].mean(1),[alpha/2,1-alpha/2]).tolist()
def load_samples(c):
 f=ROOT/'acquisition'/f'case_{c:03d}';d={'bridge':np.load(f/'bridge.npz')['samples'],'standard4096':np.load(f/'standard4096.npz')['standard']}
 for label in ['fixed8192','fixed16384','budget']:
  p=np.load(f/f'{label}.npz')
  for est in ['standard','crossfit']:d[f'{est}_{label}']=p[est]
 return d

def stability(x,s):
 h,q,a=x.shape;g=x.mean(0);choice=g.argmax(1);rr=np.random.default_rng(s)
 boot=x[rr.integers(h,size=(2000,h))].mean(1);alt=boot.copy();alt[:,np.arange(q),choice]=-np.inf
 gap=boot[:,np.arange(q),choice]-alt.max(2);regret=boot.max(2)-boot[:,np.arange(q),choice]
 tol=np.maximum(5e-5,.05*g.max(1));upper=np.quantile(regret,.95,axis=0);active=g.max(1)>5e-5
 return dict(choice=choice.tolist(),active=active.tolist(),decisive=(np.quantile(gap,.025,axis=0)>0).tolist(),practical=(upper<=tol).tolist(),upper=upper.tolist(),tolerance=tol.tolist(),half=(x[:h//2].mean(0).argmax(1)==x[h//2:].mean(0).argmax(1)).tolist())

losses={m:[] for m in METHODS};noobs=[];policies=[];rank=[];raw=[];means=[];actual=[];integrity=[];hashes=[];choices={m:[] for m in METHODS};ess={m:[] for m in ['standard4096','fixed8192','fixed16384','budget']};worlds=[]
for c in range(N):
 name=f'case_{c:03d}';w=ROOT/'worlds'/name;a=ROOT/'acquisition'/name;e=ROOT/'evaluation'/name
 for f in [w,a,e]:assert (f/'complete.json').exists(),f
 p=json.loads((a/'policy.json').read_text());mark=json.loads((a/'complete.json').read_text());ev=json.loads((e/'complete.json').read_text());sha=hashlib.sha256((a/'policy.json').read_bytes()).hexdigest();assert sha==mark['policy_sha256']==ev['policy_sha256'];assert mark['time']<=ev['time']
 pub=json.loads((w/'public.json').read_text());true=json.loads((w/'private/truth.json').read_text());hashes.append(hashlib.sha256((w/'private/truth.json').read_bytes()).hexdigest())
 pred=np.load(e/'predictions.npz')['mean'];truth=np.array(true['velocity']);assert pred.shape==(R,Q,2,4)
 L=((pred-truth[None,:,:,None])**2).sum(2)/2;actual.append(L);samples=load_samples(c)
 row=dict(case=c,methods={})
 for m in METHODS:
  x=samples[m];assert x.shape==(512,Q,4) and np.isfinite(x).all();choice=x.mean(0).argmax(1);assert choice.tolist()==p['choices'][m];choices[m].append(choice)
  losses[m].append(L[:,np.arange(Q),choice]);row['methods'][m]=stability(x,899000+c)
 rank.append(row);policies.append(p);worlds.append(dict(case=c,x=true['x'],factual=pub['factual']))
 noobs.append(((np.array(pub['current'])-truth)**2).sum(1)/2)
 for key in ess:
  file=a/f'{key}.npz';ess[key].append(np.load(file)['ess'])
 assert p['snis_budget_seconds']<=p['bridge_budget']
assert len(set(hashes))==N
assert min(json.loads((ROOT/'evaluation'/f'case_{c:03d}'/'complete.json').read_text())['time'] for c in range(N))>=max(json.loads((ROOT/'acquisition'/f'case_{c:03d}'/'complete.json').read_text())['time'] for c in range(N))
losses={k:np.stack(v) for k,v in losses.items()};choices={k:np.stack(v) for k,v in choices.items()};actual=np.stack(actual);world_loss={m:v.mean((1,2)) for m,v in losses.items()};diffs={m:world_loss['bridge']-world_loss[m] for m in METHODS if m!='bridge'}
summary=dict(n_worlds=N,repeats=R,queries=Q,means={m:float(v.mean()) for m,v in world_loss.items()},no_observation_mse=float(np.mean(noobs)),contrasts={},estimator_effects={},ranking={},budget={},ess={},reference={},decoder_audit={},noise={},scope='64 independent corner worlds, precise proxy only, equal observation cost')
for m,x in diffs.items():
 summary['contrasts'][m]=dict(mean=float(x.mean()),ci95=interval(x),ci_bonferroni=interval(x,.05/4) if m in PRIMARY else None,world_wins=int((x<0).sum()),world_losses=int((x>0).sum()),choice_disagreement=int((choices['bridge']!=choices[m]).sum()),per_world=x.tolist(),leave_one_out_range=[float((x.sum()-x).min()/63),float((x.sum()-x).max()/63)])
for label in ['fixed8192','fixed16384','budget']:
 d=world_loss['crossfit_'+label]-world_loss['standard_'+label];summary['estimator_effects'][label]=dict(mean=float(d.mean()),ci95=interval(d),disagreement=int((choices['crossfit_'+label]!=choices['standard_'+label]).sum()))
for m in METHODS:
 records=[r['methods'][m] for r in rank];active=np.array([r['active'] for r in records]);summary['ranking'][m]=dict(active=int(active.sum()),decisive=int((np.array([r['decisive'] for r in records])&active).sum()),practical=int((np.array([r['practical'] for r in records])&active).sum()),half_agreement=float(np.array([r['half'] for r in records])[active].mean()),view_choices=int((choices[m]==0).sum()),left_choices=int((choices[m]==1).sum()),right_choices=int((choices[m]==2).sum()),center_choices=int((choices[m]==3).sum()))
for key in ['bridge_budget','snis_budget_seconds','snis_budget_n','budget_utilization','all_snis_compute_seconds']:
 v=np.array([p[key] for p in policies]);summary['budget'][key]=dict(min=float(v.min()),median=float(np.median(v)),max=float(v.max()),mean=float(v.mean()),sum=float(v.sum()))
summary['budget']['below95_worlds']=[p['case'] for p in policies if p['budget_utilization']<.95]
for m,rows in ess.items():
 v=np.stack(rows);summary['ess'][m]={a:dict(median=float(np.median(v[:,:,i])),collapse_fraction=float((v[:,:,i]<=1.01).mean()),p05=float(np.quantile(v[:,:,i],.05))) for i,a in enumerate(['view','left','right','center'])}
# Repeated-noise uncertainty vs independent-world uncertainty.
for r in [1,4,8,16,32]:
 summary['noise'][str(r)]={}
 for m in PRIMARY:
  d=(losses['bridge'][:,:r]-losses[m][:,:r]).mean(2);w=d.mean(1)
  summary['noise'][str(r)][m]=dict(mean=float(w.mean()),ci95=interval(w),world_sem=float(w.std(ddof=1)/8),conditional_repeat_sem=float(np.sqrt(d.var(1,ddof=1).sum()/r)/N) if r>1 else None)
# Matched-H finite full-SMC acquisition reference, independent of evaluation.
ref_cases=[0,1,2,3,8,9,10,11];refrows=[]
for c in ref_cases:
 f=ROOT/'reference'/f'case_{c:03d}';assert (f/'complete.json').exists();ref=np.load(f/'samples.npz');g=ref['samples'].mean(0);choice=g.argmax(1);active=g.max(1)>5e-5;d=load_samples(c);row=dict(case=c,active=active.tolist(),reference_choice=choice.tolist(),methods={})
 for m in METHODS:
  cc=d[m][:128].mean(0).argmax(1);reg=g.max(1)-g[np.arange(Q),cc]
  file='bridge' if m=='bridge' else 'standard4096' if m=='standard4096' else m.split('_',1)[1]
  candidate=np.load(ROOT/'acquisition'/f'case_{c:03d}'/(file+'.npz'))['mean'][:128]
  row['methods'][m]=dict(agreement=(cc==choice).tolist(),reference_regret=reg.tolist(),conditional_mean_rmse=float(np.sqrt(((candidate-ref['mean'])**2).mean())))
 refrows.append(row)
for m in METHODS:
 mask=np.array([r['active'] for r in refrows]);agree=np.array([r['methods'][m]['agreement'] for r in refrows]);reg=np.array([r['methods'][m]['reference_regret'] for r in refrows]);summary['reference'][m]=dict(active=int(mask.sum()),agreement=float(agree[mask].mean()),regret=float(reg[mask].mean()),max_regret=float(reg[mask].max()),conditional_mean_rmse=float(np.mean([r['methods'][m]['conditional_mean_rmse'] for r in refrows])))
# Decoder particle sensitivity, fixed same actual observation arrays.
audit=[]
for c in range(4):
 folder=ROOT/'decoder_audit'/f'case_{c:03d}';assert (folder/'complete.json').exists();p=np.load(folder/'predictions.npz')['mean'];truth=np.array(json.loads((ROOT/'worlds'/f'case_{c:03d}'/'private/truth.json').read_text())['velocity']);ll=((p-truth[None,:,:,None])**2).sum(2)/2;audit.append(ll)
audit=np.stack(audit)
for m in PRIMARY:
 b=audit[np.arange(4)[:,None,None],np.arange(R)[None,:,None],np.arange(Q)[None,None,:],choices['bridge'][:4,None,:]]
 s=audit[np.arange(4)[:,None,None],np.arange(R)[None,:,None],np.arange(Q)[None,None,:],choices[m][:4,None,:]]
 old=float((losses['bridge'][:4]-losses[m][:4]).mean());new=float((b-s).mean());summary['decoder_audit'][m]=dict(n2048_difference=old,n4096_difference=new,change=new-old)
# SMC completion / ESS / seeds across all phases, including bank statistics.
seeds=[];populations=0;minratio=1.;seconds=0.;traj=0
for phase in ['bridge','evaluation','reference','decoder_audit']:
 for path in (ROOT/phase).glob('case_*/*.json'):
  if path.name=='complete.json':continue
  st=json.loads(path.read_text());assert min(st['final_beta'])>=1-1e-9;ratio=min(st['min_stage_ess'])/st['n'];assert ratio>=.74999;minratio=min(minratio,ratio);seeds.append(st['seed']);populations+=st['h'];seconds+=st['compute_seconds'];traj+=st['trajectory_rollouts']
assert len(seeds)==len(set(seeds))
summary['integrity']=dict(passed=True,unique_worlds=len(set(hashes)),unique_conditional_seeds=len(seeds),conditional_populations=populations,min_stage_ess_ratio=minratio,conditional_compute_seconds=seconds,conditional_trajectory_rollouts=traj,all_policies_frozen_before_any_evaluation=True)
# Global mean contributions, masking other queries to zero; partition must sum to aggregate.
active=np.array([r['methods']['bridge']['active'] for r in rank]);practical=np.array([r['methods']['bridge']['practical'] for r in rank]);decisive=np.array([r['methods']['bridge']['decisive'] for r in rank]);groups={'decisive':active&decisive,'practical_ambiguous':active&~decisive&practical,'unresolved':active&~practical,'inactive':~active}
# Decisive and unresolved can overlap numerically under different criteria. Use exclusive partition.
groups['unresolved']=active&~decisive&~practical
assert np.all(sum(g.astype(int) for g in groups.values())==1)
summary['ambiguity_contributions']={}
for name,mask in groups.items():
 row=dict(queries=int(mask.sum()),contrasts={})
 for m in PRIMARY:
  diff=(losses['bridge']-losses[m]).mean(1);x=(diff*mask).mean(1);row['contrasts'][m]=dict(mean=float(x.mean()),ci95=interval(x))
 summary['ambiguity_contributions'][name]=row
for m in PRIMARY:assert abs(sum(r['contrasts'][m]['mean'] for r in summary['ambiguity_contributions'].values())-summary['contrasts'][m]['mean'])<1e-12
# Supplementary model-aware nuisance control, declared before any actual evaluation.
summary['nuisance']={};rb_losses={};rb_choices={}
for estimator in ['standard','crossfit']:
 choices_rb=[];mse=[];timings=[];ess_rb=[];rank_rb=[];rmse=[];agreements=[];regrets=[]
 for c in range(N):
  f=ROOT/'nuisance'/f'case_{c:03d}';marker=json.loads((f/'complete.json').read_text());p=json.loads((f/'policy.json').read_text());assert hashlib.sha256((f/'policy.json').read_bytes()).hexdigest()==marker['policy_sha256']
  v=np.load(f/'samples.npz');choice=v[estimator].mean(0).argmax(1);assert choice.tolist()==p['choices'][estimator];choices_rb.append(choice);mse.append(actual[c][:,np.arange(Q),choice]);timings.append(p['seconds']);ess_rb.append(v['ess'][:,0]);rank_rb.append(stability(v[estimator],910000+c))
  if c in ref_cases:
   ref=np.load(ROOT/'reference'/f'case_{c:03d}'/'samples.npz');g=ref['samples'].mean(0);active_ref=g.max(1)>5e-5;cc=v[estimator][:128].mean(0).argmax(1);agreements.extend((cc==g.argmax(1))[active_ref].tolist());regrets.extend((g.max(1)-g[np.arange(Q),cc])[active_ref].tolist());rmse.append(float(np.sqrt(((v['mean'][:128]-ref['mean'])**2).mean())))
 mse=np.stack(mse);rb_losses[estimator]=mse;rb_choices[estimator]=np.stack(choices_rb);diff=world_loss['bridge']-mse.mean((1,2));active_rb=np.array([v['active'] for v in rank_rb]);summary['nuisance'][estimator]=dict(mse=float(mse.mean()),bridge_minus=float(diff.mean()),ci95=interval(diff),mean_seconds=float(np.mean(timings)),median_seconds=float(np.median(timings)),mean_budget_fraction=float(np.mean(np.array(timings)/np.array([p['bridge_budget'] for p in policies]))),view_median_ess=float(np.median(ess_rb)),view_collapse_fraction=float((np.array(ess_rb)<=1.01).mean()),choice_disagreement=int((choices['bridge']!=rb_choices[estimator]).sum()),active=int(active_rb.sum()),decisive=int((active_rb&np.array([v['decisive'] for v in rank_rb])).sum()),practical=int((active_rb&np.array([v['practical'] for v in rank_rb])).sum()),reference_agreement=float(np.mean(agreements)),reference_regret=float(np.mean(regrets)),conditional_mean_rmse=float(np.mean(rmse)),per_world=diff.tolist(),minus_original16k=float((mse.mean((1,2))-world_loss[estimator+'_fixed16384']).mean()),minus_original16k_ci95=interval(mse.mean((1,2))-world_loss[estimator+'_fixed16384']))
summary['nuisance_validation']=json.loads((ROOT/'validation/nuisance.json').read_text())
np.savez_compressed(OUT/'nuisance_losses.npz',**rb_losses,**{'choices_'+k:v for k,v in rb_choices.items()})
np.savez_compressed(OUT/'paired_losses.npz',**losses,action_losses=actual,**{'choices_'+k:v for k,v in choices.items()})
summary['fixed_compute']={}
for label in ['standard4096','fixed8192','fixed16384','budget']:
 t=np.array([float(np.load(ROOT/'acquisition'/f'case_{c:03d}'/(label+'.npz'))['seconds']) for c in range(N)])
 summary['fixed_compute'][label]=dict(mean_seconds=float(t.mean()),median_seconds=float(np.median(t)))
for m in summary['contrasts']:
 diff=choices['bridge']!=choices[m];summary['contrasts'][m]['active_choice_disagreement']=int((diff&active).sum());summary['contrasts'][m]['inactive_choice_disagreement']=int((diff&~active).sum())
summary['nuisance_bounds_validation']=json.loads((ROOT/'validation/nuisance_bounds.json').read_text());assert summary['nuisance_bounds_validation']['passed']
# Supplementary Monte Carlo precision check for a borderline adjusted endpoint.
x=np.array(summary['contrasts']['standard_budget']['per_world']);rr=np.random.default_rng(987611);bs=np.concatenate([x[rr.integers(N,size=(10000,N))].mean(1) for _ in range(50)])
summary['bootstrap_sensitivity']=dict(draws=500000,seed=987611,contrast='bridge - standard_budget',ci95=np.quantile(bs,[.025,.975]).tolist(),ci9875=np.quantile(bs,[.00625,.99375]).tolist(),purpose='posthoc bootstrap Monte Carlo sensitivity; predefined primary uses original 50000 draws')
(OUT/'bootstrap_sensitivity.json').write_text(json.dumps(summary['bootstrap_sensitivity'],indent=2))
for name,obj in [('summary',summary),('ranking',rank),('reference',refrows),('worlds',worlds)]: (OUT/f'{name}.json').write_text(json.dumps(obj,indent=2))
with (OUT/'world_losses.csv').open('w') as f:
 writer=csv.writer(f);writer.writerow(['case',*METHODS]);writer.writerows([[c,*[world_loss[m][c] for m in METHODS]] for c in range(N)])
print(json.dumps(summary,indent=2))
