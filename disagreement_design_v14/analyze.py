import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *
CONDITIONS=['matched','viscous','saturating_half','saturating_full','weakening_half','weakening_full']
def ci(a,b=None,level=.95,n=30000):
    rng=np.random.default_rng(20261115);values=[]
    for start in range(0,n,1000):
        ix=rng.integers(len(a),size=(min(1000,n-start),len(a)));v=a[ix].mean(1)
        if b is not None:v=v/b[ix].mean(1)
        values.append(v)
    v=np.concatenate(values);q=np.quantile(v,[(1-level)/2,(1+level)/2])
    return dict(mean=float(a.mean()) if b is None else float(a.mean()/b.mean()),low=float(q[0]),high=float(q[1]),level=level)
def main():
    assert os.environ.get('SLURM_JOB_ID');f=read(R/'decision_freeze.json');assert sha(R/'decision_predictions.npz')==f['predictions']
    for p in (R/'freezes').glob('*.json'):check(p.stem)
    pred=np.load(R/'decision_predictions.npz');truth=np.load(R/'data/test/private.npz');target=truth['target'].reshape(-1,2);mask=truth['theta'][:,1]==1
    losses={k:(.5*((v-target[:,None,:])**2).sum(-1)).mean(1).reshape(256,6) for k,v in pred.items()};np.savez_compressed(R/'world_losses.npz',**losses)
    policies=np.load(R/'decision_actions.npz');out=dict(worlds=256,changed_worlds=int(mask.sum()),means={},changed_means={},primary={},secondary={},actions={},negative_controls={},audit={},pair_diagnostics={})
    for j,c in enumerate(CONDITIONS):
        out['means'][c]={k:float(v[:,j].mean()) for k,v in losses.items()};out['changed_means'][c]={k:float(v[mask,j].mean()) for k,v in losses.items()}
        out['actions'][c]={k:np.bincount(v.reshape(256,6)[:,j],minlength=4).tolist() for k,v in policies.items()}
        if j in [3,5]:
            for baseline in ['qmc16_native','qmc16_probe','qmc16_nested128']:
                out['primary'][c+'/'+baseline]=dict(delta=ci(losses['qmc16_pair'][:,j]-losses[baseline][:,j],level=1-.05/6),ratio=ci(losses['qmc16_pair'][:,j],losses[baseline][:,j],level=1-.05/6))
        out['secondary'][c]={}
        comparisons=[('qmc16_pair',k) for k in ['qmc16_unfiltered','qmc16_agnostic','qmc16_query','qmc64_native','qmc64_pair','qmc16_nested128']]+[('compact_pair',k) for k in ['compact_native','compact_probe','qmc16_pair','compact_pair_qmc16_decoder']]
        for method,baseline in comparisons:out['secondary'][c][method+'/'+baseline]=ci(losses[method][:,j]-losses[baseline][:,j],n=10000)
    out['matched_ratio']=ci(losses['qmc16_pair'][:,0],losses['qmc16_native'][:,0])
    out['success_screen']=bool(out['matched_ratio']['high']<=1.10 and all(v['delta']['high']<0 for v in out['primary'].values()))
    for k,v in losses.items():out['negative_controls'][k]=float(abs(v[~mask,1:]-v[~mask,0,None]).max())
    for method in ['compact_pair','qmc16_pair','qmc64_pair']:
        d=np.load(R/'policies'/f'{method}.npz');out['pair_diagnostics'][method]={c:{k:float(d[k].reshape(256,6)[:,j].mean()) for k in ['eligible_fraction','query_disagreement','fallback']} for j,c in enumerate(CONDITIONS)}
    audit=np.load(R/'ranking_audit.npz');eval16=np.load(R/'evaluations/qmc16.npz')['predictive'][:96];lossall=.5*((eval16-target[:96,None,None,:])**2).sum(-1)
    for label,g in audit.items():
        a=g.argmax(1);baseline='qmc16_pair' if label.startswith('pair') else 'qmc16_nested128';old=policies[baseline][:96]
        dif=(lossall[np.arange(96),a]-lossall[np.arange(96),old]).mean(1).reshape(16,6)
        out['audit'][label]=dict(agreement=float((a==old).mean()),by_condition_agreement={c:float((a==old).reshape(16,6)[:,j].mean()) for j,c in enumerate(CONDITIONS)},loss_difference={c:ci(dif[:,j],n=10000) for j,c in enumerate(CONDITIONS)})
    out['audit']['nested512_between_seeds_agreement']=float((audit['nested512'].argmax(1)==audit['nested512_seed2'].argmax(1)).mean())
    out['timing_gpu']={k:{a:b for a,b in v.items() if a!='rows'} for k,v in read(R/'timing_gpu.json').items()};out['timing_cpu']={k:{a:b for a,b in v.items() if a!='rows'} for k,v in read(R/'timing_cpu.json').items()}
    out['speedup_vs_nested']=out['timing_gpu']['qmc16_nested128']['median_s']/out['timing_gpu']['qmc16_pair']['median_s']
    write(R/'summary.json',out)
    keys=['compact_pair','qmc16_pair','qmc16_native','qmc16_probe','qmc16_nested128','qmc64_native'];labels=['Compact pair','QMC16 pair','QMC16 LMMSE','QMC16 probe','Cross-fitted nested128','QMC64 LMMSE']
    fig,axes=plt.subplots(1,3,figsize=(13,4.5))
    for ax,j in zip(axes,[1,3,5]):
        for k,label in zip(keys,labels):ax.scatter(out['timing_gpu'][k]['median_s']*1000,losses[k][:,j].mean(),label=label,marker='*' if k=='qmc16_pair' else 'o',s=80 if k=='qmc16_pair' else 45)
        ax.set_title(CONDITIONS[j]);ax.set_xlabel('Warm GPU latency (ms)');ax.set_ylabel('Mean decision MSE');ax.set_xscale('log');ax.grid(alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False);fig.subplots_adjust(bottom=.26,wspace=.3);fig.savefig(R/'loss_latency.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,condition in zip(axes,['saturating_full','weakening_full']):
        show=['qmc16_native','qmc16_pair','qmc16_nested128','compact_pair'];counts=np.array([out['actions'][condition][k] for k in show])/256
        bottom=np.zeros(len(show))
        for j,label in enumerate(['None','Query view','Factual view','Probe']):ax.bar(range(len(show)),counts[:,j],bottom=bottom,label=label);bottom+=counts[:,j]
        ax.set_xticks(range(len(show)),['LMMSE','Pair','Nested','Compact pair'],rotation=20);ax.set_ylim(0,1);ax.set_title(condition);ax.set_ylabel('Action frequency')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=4,frameon=False);fig.subplots_adjust(bottom=.26,wspace=.25);fig.savefig(R/'action_choices.png',dpi=180);plt.close(fig)
    print({k:out[k] for k in ['means','primary','success_screen','matched_ratio','audit','speedup_vs_nested']})
if __name__=='__main__':main()
