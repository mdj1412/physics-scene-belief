import os,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import R,private,check,write
from physics import CONDITIONS

OUT=R/'analysis';OUT.mkdir(exist_ok=True)
def bootstrap(a,b=None,level=.95):
    r=np.random.default_rng(20261112);vals=[]
    for _ in range(30):
        ids=r.integers(len(a),size=(1000,len(a)));aa=a[ids].mean(1)
        vals.append(aa if b is None else aa/b[ids].mean(1))
    v=np.concatenate(vals);q=np.quantile(v,[(1-level)/2,(1+level)/2])
    return dict(mean=float(a.mean()) if b is None else float(a.mean()/b.mean()),low=float(q[0]),high=float(q[1]),level=level)
def main():
    assert os.environ.get('SLURM_JOB_ID')
    selected=json.loads((R/'development.json').read_text())['selected']['name'];truth=private('test');target=truth['target']
    data={p.stem:np.load(p) for p in (R/'evaluations').glob('*.npz')};policies={p.stem:np.load(p)['actions'] for p in (R/'policies').glob('*.npz')}
    methods={};actions={};all_losses={}
    for nm,d in data.items():
        check(nm);loss=.5*((d['predictive']-target[:,None,None,:])**2).sum(-1);all_losses[nm]=loss
        a=policies[nm];methods[nm]=loss[np.arange(len(a)),a].mean(1).reshape(256,3);actions[nm]=a.reshape(256,3)
    for nm in ['qmc16384','reference61']:
        alias=nm+'_tree';check(alias);a=policies[alias]
        methods[alias]=all_losses[nm][np.arange(len(a)),a].mean(1).reshape(256,3);actions[alias]=a.reshape(256,3)
        ml=.5*((data[nm]['mixture']-target[:,None,None,:])**2).sum(-1)
        methods[nm+'_mixture']=ml[np.arange(len(a)),policies[nm]].mean(1).reshape(256,3)
    for decoder in ['mixture','hard']:
        loss=.5*((data[selected][decoder]-target[:,None,None,:])**2).sum(-1)
        methods[selected+'_'+decoder]=loss[np.arange(len(target)),policies[selected]].mean(1).reshape(256,3)
    for nm in [selected,'qmc16384','reference61']:
        for action,label in [(0,'none'),(3,'probe')]:methods[nm+'_'+label]=all_losses[nm][:,action].mean(1).reshape(256,3)
    methods['compact_choices_reference_decoder']=all_losses['reference61'][np.arange(len(target)),policies[selected]].mean(1).reshape(256,3)
    methods['reference_choices_compact_decoder']=all_losses[selected][np.arange(len(target)),policies['reference61']].mean(1).reshape(256,3)
    theta=truth['theta'].reshape(256,3,5);changed=theta[:,0,1]==1
    bench=json.loads((R/'benchmark.json').read_text());summary=dict(selected=selected,worlds=256,conditions=CONDITIONS,changed_contact_worlds=int(changed.sum()),conditions_results={},paired_shift={},timing=bench,primary_comparisons={})
    for j,condition in enumerate(CONDITIONS):
        summary['conditions_results'][condition]=dict(means={k:float(v[:,j].mean()) for k,v in methods.items()},ci95={k:bootstrap(v[:,j]) for k,v in methods.items()},
            actions={k:np.bincount(v[:,j],minlength=4).tolist() for k,v in actions.items()},agreement_with_reference={k:float((v[:,j]==actions['reference61'][:,j]).mean()) for k,v in actions.items()})
        for baseline in ['reference61','qmc16384']:
            delta=bootstrap(methods[selected][:,j]-methods[baseline][:,j],level=1-.05/6)
            ratio=bootstrap(methods[selected][:,j],methods[baseline][:,j],level=1-.05/6)
            speed=bench[baseline]['median_total_s']/bench[selected]['median_total_s']
            summary['primary_comparisons'][condition+'/'+baseline]=dict(delta=delta,ratio=ratio,serial_speedup=speed,passes_screen=bool(ratio['high']<=1.10 and speed>=2))
    for k,v in methods.items():
        summary['paired_shift'][k]=dict(boundary=bootstrap(v[:,1]-v[:,0]),contact_all=bootstrap(v[:,2]-v[:,0]),contact_changed=bootstrap((v[:,2]-v[:,0])[changed]),contact_negative_control_max=float(abs((v[:,2]-v[:,0])[~changed]).max()))
    summary['contact_changed_means']={k:float(v[changed,2].mean()) for k,v in methods.items()}
    summary['numerical']=json.loads((R/'reference_audit.json').read_text());summary['validation']=json.loads((R/'validation.json').read_text())
    np.savez_compressed(OUT/'world_losses.npz',**methods)
    write(OUT/'summary.json',summary)
    # Native loss vs measured warm latency; compact configs were fixed in dev.
    figure,axes=plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
    keys=[k for k in bench if k in methods]
    for j,ax in enumerate(axes):
        for k in keys:
            x=bench[k]['median_total_s']*1000;y=methods[k][:,j].mean()
            color='#c43d48' if k==selected else '#ce8b89' if k.startswith('compact') else '#196681'
            marker='*' if k==selected else 'o' if k.startswith('compact') else 's'
            ax.scatter(x,y,c=color,marker=marker,s=100 if k==selected else 35)
            label={'reference61':'Full reference','qmc4096':'QMC4k','qmc16384':'QMC16k','qmc65536':'QMC64k',selected:'Selected compact'}.get(k)
            if label:ax.annotate(label,(x,y),xytext=(4,5),textcoords='offset points',fontsize=8)
        ax.set_xscale('log');ax.set_xlabel('Warm serial latency (ms, batch 1)');ax.set_ylabel('Native decision MSE');ax.set_title(CONDITIONS[j]);ax.grid(alpha=.15)
    figure.savefig(OUT/'loss_latency.png',dpi=180);plt.close(figure)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    show=[selected,'qmc16384','reference61'];labels=['Compact','QMC16k','Full reference']
    for ax,field,title in zip(axes,['boundary','contact_changed'],['Boundary-only shift','Contact-law shift (changed subset)']):
        vals=[summary['paired_shift'][k][field] for k in show];means=[v['mean'] for v in vals]
        ax.bar(labels,means,color=['#c43d48','#5293ab','#196681']);ax.errorbar(range(3),means,yerr=np.array([[v['mean']-v['low'],v['high']-v['mean']] for v in vals]).T,fmt='none',color='black',capsize=4)
        ax.axhline(0,color='gray',lw=.8);ax.set_title(title);ax.set_ylabel('Paired MSE change from matched (95% CI)')
    fig.savefig(OUT/'separated_shifts.png',dpi=180);plt.close(fig)
    print(json.dumps({'selected':selected,'means':{c:summary['conditions_results'][c]['means'] for c in CONDITIONS},'primary':summary['primary_comparisons'],'changed_contact_worlds':int(changed.sum())},indent=2))
if __name__=='__main__':main()
