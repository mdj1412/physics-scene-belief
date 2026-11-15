import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':['AppleGothic','DejaVu Sans'],'axes.unicode_minus':False,'font.size':10,'figure.dpi':150})
root=Path('results/analysis');out=root/'figures';out.mkdir(exist_ok=True);s=json.loads((root/'summary.json').read_text());d=np.load(root/'paired_losses.npz')
methods=['standard4096','standard_fixed16384','crossfit_fixed16384','standard_budget','crossfit_budget'];labels=['Standard SNIS 4k','Standard SNIS 16k','Cross-fitted SNIS 16k','Standard SNIS budget','Cross-fitted SNIS budget'];cols=['#8c8c8c','#e0a23b','#79a65a','#9982ba','#557db6']
fig,axs=plt.subplots(1,2,figsize=(12,4.6),gridspec_kw={'width_ratios':[1,1.35]})
axs[0].barh(['Bridge',*labels],[s['means']['bridge'],*[s['means'][m] for m in methods]],color=['#c54d48',*cols]);axs[0].invert_yaxis();axs[0].set_xlabel('Actual velocity MSE');axs[0].set_title('64 independent worlds × 32 noise repeats')
for i,m in enumerate(methods):
 r=s['contrasts'][m];lo,hi=r['ci95'];axs[1].errorbar(r['mean'],i,xerr=[[r['mean']-lo],[hi-r['mean']]],fmt='o',color=cols[i],capsize=4)
 if r['ci_bonferroni']:
  lo,hi=r['ci_bonferroni'];axs[1].plot([lo,hi],[i,i],color=cols[i],alpha=.3,lw=6)
axs[1].axvline(0,color='black',lw=.8);axs[1].set_yticks(range(5),labels);axs[1].invert_yaxis();axs[1].set_xlabel('Bridge − SNIS MSE (negative: Bridge 우세)');axs[1].set_title('World-cluster CI: 95% / adjusted 98.75%');fig.tight_layout();fig.savefig(out/'decision_loss.png',bbox_inches='tight');plt.close(fig)
policies=[json.loads((Path('results/acquisition')/f'case_{c:03d}'/'policy.json').read_text()) for c in range(64)]
fig,axs=plt.subplots(1,3,figsize=(13,4))
axs[0].scatter([p['bridge_budget'] for p in policies],[p['snis_budget_seconds'] for p in policies],c='#557db6',s=22);mx=max(p['bridge_budget'] for p in policies)*1.05;axs[0].plot([0,mx],[0,mx],'k--',lw=1);axs[0].set(xlabel='Bridge seconds',ylabel='Budget SNIS seconds',title='같은 GPU의 시간 한도')
axs[1].hist([p['budget_utilization']*100 for p in policies],bins=np.arange(60,102,3),color='#557db6');axs[1].axvline(95,color='black',ls='--');axs[1].set(xlabel='Budget utilization (%)',ylabel='Worlds',title='완료된 checkpoint만 사용')
axs[2].scatter([p['snis_budget_n'] for p in policies],[s['contrasts']['crossfit_budget']['per_world'][c] for c in range(64)],color='#557db6',s=22);axs[2].axhline(0,color='black',lw=.8);axs[2].set(xlabel='Budget bank particles',ylabel='Bridge − cross-fitted SNIS MSE',title='World별 효과와 bank 크기');axs[2].ticklabel_format(axis='x',style='sci',scilimits=(0,0));fig.tight_layout();fig.savefig(out/'compute_budget.png',bbox_inches='tight');plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(12,4.5));ms=['bridge',*methods];ls=['Bridge',*labels]
for vals,color,label in [('decisive','#557db6','Decisive top-1'),('practical','#79a65a','Practically stable')]:
 off=-.18 if vals=='decisive' else .18;axs[0].barh(np.arange(len(ms))+off,[s['ranking'][m][vals]/s['ranking'][m]['active']*100 for m in ms],height=.35,color=color,label=label)
axs[0].set_yticks(range(len(ms)),ls);axs[0].invert_yaxis();axs[0].set(xlim=(0,105),xlabel='Active queries (%)',title='Ranking identity와 practical stability');axs[0].legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncol=2,frameon=False,fontsize=8)
for m,col in [('standard4096','#8c8c8c'),('crossfit_fixed16384','#79a65a'),('crossfit_budget','#557db6')]:
 vals=[s['noise'][str(r)][m] for r in [1,4,8,16,32]];xs=np.arange(5);v=np.array([t['mean'] for t in vals]);ci=np.array([t['ci95'] for t in vals]);axs[1].plot(xs,v,'o-',label=labels[methods.index(m)],color=col);axs[1].fill_between(xs,ci[:,0],ci[:,1],alpha=.12,color=col)
axs[1].axhline(0,color='black',lw=.8);axs[1].set_xticks(range(5),[1,4,8,16,32]);axs[1].set(xlabel='Observation noise repeats',ylabel='Bridge − SNIS MSE',title='World-cluster 95% CI');axs[1].legend(fontsize=8);fig.tight_layout();fig.savefig(out/'ranking_noise.png',bbox_inches='tight');plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(12,4.3))
for ax,m,title in zip(axs,['standard4096','crossfit_budget'],['Bridge − Standard SNIS 4k','Bridge − Cross-fitted SNIS budget (확대)']):
 v=np.array(s['contrasts'][m]['per_world']);order=np.argsort(v);ax.bar(np.arange(64),v[order],color=np.where(v[order]<0,'#557db6','#c54d48'));ax.axhline(0,color='black',lw=.8);ax.set(xlabel='Independent worlds (효과순 정렬)',ylabel='Paired MSE difference',title=title)
fig.tight_layout();fig.savefig(out/'world_heterogeneity.png',bbox_inches='tight');plt.close(fig)
print(str(out.resolve()))
fig,axs=plt.subplots(1,2,figsize=(11,4.5))
labels2=['Bridge','Standard 16k','Cross-fitted 16k','Standard budget','Cross-fitted budget','RB standard 16k','RB cross-fitted 16k'];values=[s['means'][k] for k in ['bridge','standard_fixed16384','crossfit_fixed16384','standard_budget','crossfit_budget']]+[s['nuisance'][k]['mse'] for k in ['standard','crossfit']]
axs[0].barh(labels2,values,color=['#c54d48','#aaaaaa','#79a65a','#9982ba','#557db6','#db9c42','#578b85']);axs[0].invert_yaxis();axs[0].set(xlabel='Actual velocity MSE',title='Model-aware nuisance marginalization');axs[0].ticklabel_format(axis='x',style='sci',scilimits=(0,0))
for i,k in enumerate(['standard','crossfit']):
 v=s['nuisance'][k];lo,hi=v['ci95'];axs[1].errorbar(v['bridge_minus'],i,xerr=[[v['bridge_minus']-lo],[hi-v['bridge_minus']]],fmt='o',capsize=5,color=['#db9c42','#578b85'][i])
axs[1].axvline(0,color='black',lw=.8);axs[1].set_yticks([0,1],['RB standard 16k','RB cross-fitted 16k']);axs[1].invert_yaxis();axs[1].set(xlabel='Bridge − RB MSE (negative: Bridge 우세)',title='Supplementary world-cluster 95% CI');fig.tight_layout();fig.savefig(out/'nuisance_control.png',bbox_inches='tight');plt.close(fig)
