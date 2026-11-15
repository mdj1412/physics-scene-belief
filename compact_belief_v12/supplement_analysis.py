"""Post-primary descriptive diagnostics; no fitting or policy changes."""
import os,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import R,write,private
from analyze import bootstrap

assert os.environ.get('SLURM_JOB_ID')
s=json.loads((R/'analysis/summary.json').read_text());sel=s['selected']
truth=private('test');target=truth['target'];out={}
world=np.load(R/'analysis/world_losses.npz')
for method in [sel,'qmc16384','reference61']:
    p=np.load(R/'evaluations'/f'{method}.npz')['predictive']
    loss=(.5*((p-target[:,None,None,:])**2).sum(-1)).mean(2).reshape(256,3,4)
    out[method]={}
    for j,c in enumerate(s['conditions']):
        native=world[method][:,j]
        out[method][c]=dict(fixed_means=loss[:,j].mean(0).tolist(),
            native_minus_fixed={str(a):bootstrap(native-loss[:,j,a]) for a in range(4)},
            native_minus_tree=bootstrap(native-world['qmc16384_tree'][:,j]))
out['reference_order_audit']={}
low=np.load(R/'evaluations/reference61.npz')['predictive'][:48]
high=np.load(R/'reference81_audit.npz')['predictive']
for method in [sel,'reference61']:
    a=np.load(R/'policies'/f'{method}.npz')['actions'][:48]
    diff=(.5*((high-target[:48,None,None,:])**2-(low-target[:48,None,None,:])**2).sum(-1))[np.arange(48),a].mean(1).reshape(16,3)
    out['reference_order_audit'][method]=dict(mean_by_condition=diff.mean(0).tolist(),max_abs_world_change=float(abs(diff).max()))
write(R/'analysis/supplement.json',out)

names=[sel,'qmc4096','qmc16384','qmc65536','reference61','qmc16384_tree']
labels=['Compact (selected)','QMC4k','QMC16k','QMC64k','Full reference','QMC16k + rule']
colors=['#c43d48','#4b9ca1','#286f98','#224668','#715f8c','#c18c33']
markers=['*','o','s','D','^','v']
fig,axes=plt.subplots(1,3,figsize=(13,4.5))
for j,ax in enumerate(axes):
    for k,label,col,marker in zip(names,labels,colors,markers):
        ax.scatter(s['timing'][k]['median_total_s']*1000,world[k][:,j].mean(),color=col,marker=marker,s=100 if k==sel else 55,label=label)
    ax.set_xscale('log');ax.set_xlabel('Warm GPU latency (ms, batch 1)')
    ax.set_title(['Matched','Boundary-only shift','Contact-law shift'][j]);ax.grid(alpha=.18)
    ax.ticklabel_format(axis='y',style='sci',scilimits=(-2,2));ax.set_ylabel('Mean decision MSE')
fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False)
fig.subplots_adjust(bottom=.26,wspace=.33,top=.88)
fig.savefig(R/'analysis/loss_latency_report.png',dpi=180);plt.close(fig)
print(json.dumps(out,indent=2))
