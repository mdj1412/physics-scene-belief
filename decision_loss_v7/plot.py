import json
from pathlib import Path
import numpy as np
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/physics_scene_belief_v7_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=Path('results/analysis');s=json.loads((r/'summary.json').read_text());out=r/'figures';out.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':['AppleGothic','DejaVu Sans'],'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':170})
fig,axes=plt.subplots(1,2,figsize=(12,4.6),sharex=True)
for ax,prec in zip(axes,['native','precise']):
 rows=[x for x in s['comparisons'] if x['precision']==prec]
 for i,x in enumerate(rows):
  ci=x['ci95'];ci2=x['ci_bonferroni12'];col='#186a8c' if x['baseline']=='snis4096' else '#dc7c2d'
  ax.plot(np.array(ci2)*1000,[i,i],color=col,alpha=.3,lw=5)
  ax.plot(np.array(ci)*1000,[i,i],color=col,lw=2);ax.scatter(x['mean']*1000,i,color=col,zorder=3)
 ax.axvline(0,color='black',lw=.8,ls='--');ax.set_yticks(range(len(rows)),[x['cost']+' / '+x['baseline'].replace('snis','SNIS ') for x in rows]);ax.invert_yaxis();ax.set_title(prec+' view precision');ax.set_xlabel('Bridge − SNIS decision loss (×10⁻³)\n음수는 bridge에 유리')
fig.suptitle('새로운 worlds: common full-SMC decoder, observation noise 32회 반복\n48 world clusters · 진한 선: 95% CI · 옅은 선: 12-comparison Bonferroni interval',fontsize=12);fig.tight_layout();fig.savefig(out/'decision_loss.png');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10,4.5),sharey=True)
for ax,prec in zip(axes,['native','precise']):
 for m,col in [('snis4096','#186a8c'),('snis16384','#dc7c2d')]:
  rows=[x for x in s['noise_prefix'] if x['precision']==prec and x['cost']=='equal' and x['baseline']==m];xx=np.array([x['repeats'] for x in rows]);yy=np.array([x['mean'] for x in rows])*1000;ci=np.array([x['ci95'] for x in rows])*1000
  ax.plot(xx,yy,'o-',color=col,label=m.replace('snis','SNIS '));ax.fill_between(xx,ci[:,0],ci[:,1],color=col,alpha=.12)
 ax.axhline(0,color='black',lw=.8,ls='--');ax.set_xscale('log',base=2);ax.set_xticks([1,4,8,16,32],[1,4,8,16,32]);ax.set_title(prec+', equal cost');ax.set_xlabel('추가 observation noise 반복 횟수');ax.set_ylabel('Bridge − SNIS loss (×10⁻³)');ax.legend()
fig.suptitle('Noise를 반복해도 independent worlds는 48개로 유지');fig.tight_layout();fig.savefig(out/'noise_convergence.png');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11,4.6),sharex=True)
labels={'inactive':'Inactive','bridge_unresolved':'Bridge unresolved','snis_unresolved':'SNIS unresolved','stable_agreement':'Stable agreement','stable_near_tie_disagreement':'Stable near-tie','stable_material_disagreement':'Stable material difference'}
for ax,prec in zip(axes,['native','precise']):
 rows=[x for x in s['ambiguity'] if x['precision']==prec and x['cost']=='equal'];ax.barh([labels[x['category']] for x in rows],[x['n_queries'] for x in rows],color=['#c3c8cc','#d6705c','#dca958','#6c9ca7','#8bbbc2','#547c96']);ax.invert_yaxis();ax.set_title(prec+', equal cost');ax.set_xlabel('Queries (48 worlds × 6 queries; descriptive)')
fig.suptitle('Acquisition만으로 분류 · actual outcomes를 사용하지 않음');fig.tight_layout();fig.savefig(out/'ranking_ambiguity.png');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11.6,4.7))
for method,color in [('bridge','#186a8c'),('snis4096','#dc7c2d'),('snis16384','#5c986d')]:
 vals=[]
 for prec in ['native','precise']:
  row=next(x for x in s['families'] if x['family']=='corner' and x['precision']==prec and x['cost']=='equal' and x['baseline']==('snis4096' if method=='bridge' else method))
  vals.append(row['bridge_loss'] if method=='bridge' else row['baseline_loss'])
 axes[0].plot([0,1],np.array(vals)*1000,'o-',label=method.replace('snis','SNIS '),color=color,lw=2)
axes[0].set_xticks([0,1],['native (σ × 1)','precise (σ × 0.25)']);axes[0].set_ylabel('Mean MSE (×10⁻³)');axes[0].set_title('Corner: view precision에 따른 MSE\n16 worlds · descriptive subgroup');axes[0].legend();axes[0].set_ylim(bottom=0)
with np.load(r/'paired_losses.npz') as d:world=d['native_equal_bridge_minus_snis4096'].mean((1,2))*1000
axes[1].bar(np.arange(48),world,color=['#709bad']*16+['#d99a5f']*16+['#80a18c']*16,width=.85);axes[1].axhline(0,color='black',lw=.7)
axes[1].set_xticks([7.5,23.5,39.5],['corner (0–15)','gate (16–31)','deflection (32–47)']);axes[1].set_ylabel('Bridge − SNIS4096 loss (×10⁻³)');axes[1].set_title('Native / equal cost: world별 R32 mean');axes[1].set_ylim(world.min()*1.3,world.max()*1.2)
for i in [int(world.argmin()),int(world.argmax())]:
 axes[1].annotate('world '+str(i),(i,world[i]),(i+4,world[i]+(-5 if world[i]<0 else -4)),arrowprops={'arrowstyle':'-','lw':.7},fontsize=8)
fig.tight_layout();fig.savefig(out/'condition_and_worlds.png');plt.close(fig)
(r/'plot_environment.json').write_text(json.dumps(dict(numpy=np.__version__,matplotlib=matplotlib.__version__)))
print(out)
