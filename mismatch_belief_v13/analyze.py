import os,json
import numpy as np
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *
CONDITIONS=['matched','viscous','saturating_half','saturating_full','weakening_half','weakening_full']

def ci(a,b=None,level=.95,n=30000):
    r=np.random.default_rng(20261113);res=[]
    for i in range(0,n,1000):
        ids=r.integers(len(a),size=(min(1000,n-i),len(a)));v=a[ids].mean(1)
        if b is not None:v=v/b[ids].mean(1)
        res.append(v)
    z=np.concatenate(res);q=np.quantile(z,[(1-level)/2,(1+level)/2])
    return dict(mean=float(a.mean()) if b is None else float(a.mean()/b.mean()),low=float(q[0]),high=float(q[1]),level=level)
def auc(positive,negative):
    n=len(positive);m=len(negative);r=rankdata(np.r_[positive,negative]);return float((r[:n].sum()-n*(n+1)/2)/(n*m))
def main():
    assert os.environ.get('SLURM_JOB_ID')
    freeze=read(R/'decision_freeze.json');assert sha(R/'decision_predictions.npz')==freeze['predictions']
    for p in (R/'freezes').glob('*.json'):check(p.stem)
    pred=np.load(R/'decision_predictions.npz');truth=np.load(R/'data/test/private.npz');target=truth['target'].reshape(-1,2);mask=truth['theta'][:,1]==1
    losses={k:(.5*((pred[k]-target[:,None,:])**2).sum(-1)).mean(1).reshape(256,6) for k in pred.files}
    # Common action diagnostic, derived from frozen all-action evaluations only.
    aa=np.load(R/'decision_actions.npz')['primary_gate'];exp=np.load(R/'evaluations/expanded.npz')['predictive'][np.arange(len(aa)),aa]
    losses['expanded_decoder_primary_actions']=(.5*((exp-target[:,None,:])**2).sum(-1)).mean(1).reshape(256,6)
    np.savez_compressed(R/'world_losses.npz',**losses)
    summary=dict(worlds=256,changed_worlds=int(mask.sum()),negative_controls=int((~mask).sum()),conditions=CONDITIONS,means={},changed_means={},primary={},secondary={},detection={},negative_control={},qmc_audit={})
    for j,cond in enumerate(CONDITIONS):
        summary['means'][cond]={k:float(v[:,j].mean()) for k,v in losses.items()};summary['changed_means'][cond]={k:float(v[mask,j].mean()) for k,v in losses.items()}
        if j in [1,3,5]:
            for baseline in ['qmc16_probe','expanded_qmc16_probe']:
                summary['primary'][cond+'/'+baseline]=dict(delta=ci(losses['primary_gate'][:,j]-losses[baseline][:,j],level=1-.05/6),ratio=ci(losses['primary_gate'][:,j],losses[baseline][:,j],level=1-.05/6))
        summary['secondary'][cond]={base:ci(losses['primary_gate'][:,j]-losses[base][:,j],n=10000) for base in ['base_probe','probe_gate','expanded_probe','expanded_qmc64_probe','expanded_decoder_primary_actions']}
    summary['matched_harm']=ci(losses['primary_gate'][:,0],losses['base_native'][:,0])
    flags=np.load(R/'expansion_flags.npz');basepol=np.load(R/'policies/base.npz');pre=basepol['pre_score'].reshape(256,6);threshold=read(R/'thresholds.json')['pre']
    for j,cond in enumerate(CONDITIONS):
        summary['detection'][cond]=dict(pre_all=float((pre[:,j]>threshold).mean()),pre_changed=float((pre[mask,j]>threshold).mean()),pre_auc_changed_vs_matched=auc(pre[mask,j],pre[:,0]),
            expansion_rates={k:float(v.reshape(256,6,16)[:,j].mean()) for k,v in flags.items()},expansion_changed={k:float(v.reshape(256,6,16)[mask,j].mean()) for k,v in flags.items()})
    for k,v in losses.items():summary['negative_control'][k]=float(abs(v[~mask,1:]-v[~mask,0,None]).max())
    for k in ['expanded_qmc16','expanded_qmc64','expanded_qmc64_seed2']:
        d=np.load(R/'evaluations'/f'{k}.npz');a=np.load(R/'policies'/f'{k}.npz')['actions'];ess=d['ess'][np.arange(len(a)),a].reshape(256,6,16)
        summary['qmc_audit'][k]={c:dict(ess_median=float(np.median(ess[:,j])),ess_q05=float(np.quantile(ess[:,j],.05))) for j,c in enumerate(CONDITIONS)}
    summary['qmc_seed_loss_difference']={c:ci(losses['expanded_qmc64_seed2_probe'][:,j]-losses['expanded_qmc64_probe'][:,j],n=10000) for j,c in enumerate(CONDITIONS)}
    gpu=read(R/'timing_gpu.json');cpu=read(R/'timing_cpu.json');summary['timing_gpu']={k:{a:b for a,b in v.items() if a!='rows'} for k,v in gpu.items()};summary['timing_cpu']={k:{a:b for a,b in v.items() if a!='rows'} for k,v in cpu.items()}
    summary['matched_gate_pass']=summary['matched_harm']['high']<=1.10
    summary['new_law_superiority_pass']=all(v['delta']['high']<0 for k,v in summary['primary'].items() if not k.startswith('viscous'))
    summary['compute_speedup_vs_expanded_qmc16_probe']=gpu['expanded_qmc16_probe']['median_s']/gpu['primary_gate']['median_s']
    write(R/'summary.json',summary)
    fig,axes=plt.subplots(1,3,figsize=(13,4.5))
    keys=['base_probe','primary_gate','probe_gate','expanded_probe','qmc16_probe','expanded_qmc16_probe','expanded_qmc64_probe']
    labels=['Compact probe','Primary gate','Probe + gate','Expanded compact probe','QMC16 probe','Expanded QMC16 probe','Expanded QMC64 probe']
    for ax,j in zip(axes,[1,3,5]):
        for k,label in zip(keys,labels):ax.scatter(gpu[k]['median_s']*1000,losses[k][:,j].mean(),label=label,s=60,marker='*' if k=='primary_gate' else 'o')
        ax.set_title(CONDITIONS[j]);ax.set_xlabel('Warm GPU latency (ms)');ax.set_ylabel('Mean decision MSE');ax.set_xscale('log');ax.grid(alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False);fig.subplots_adjust(bottom=.3,wspace=.3);fig.savefig(R/'loss_latency.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.3));xx=np.arange(6)
    for k in ['base_probe','primary_gate','expanded_probe','expanded_qmc64_probe']:axes[0].plot(xx,[summary['means'][c][k] for c in CONDITIONS],marker='o',label=k)
    axes[0].set_ylabel('Mean decision MSE');axes[0].legend(fontsize=8)
    axes[1].plot(xx,[summary['detection'][c]['pre_all'] for c in CONDITIONS],marker='o',label='Pre flag')
    axes[1].plot(xx,[summary['detection'][c]['expansion_rates']['primary_gate'] for c in CONDITIONS],marker='o',label='Any flag / expansion');axes[1].set_ylabel('Fraction of evaluated observations');axes[1].legend()
    for ax in axes:ax.set_xticks(xx,['Matched','Viscous','Sat half','Sat full','Weak half','Weak full'],rotation=30);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(R/'shift_detection.png',dpi=180);plt.close(fig)
    print(json.dumps({k:summary[k] for k in ['changed_worlds','means','primary','matched_harm','detection','new_law_superiority_pass','compute_speedup_vs_expanded_qmc16_probe']},indent=2))
if __name__=='__main__':main()
