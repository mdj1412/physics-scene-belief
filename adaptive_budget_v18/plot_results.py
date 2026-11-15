from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
require();s=read(R/'summary.json');names=['adaptive','balanced','component_only','outer_only','fixed32_512','fixed128_512','fixed128_4096','global128_4096','native','nested512','nested2048','nested4096','query','probe'];tasks=['contact_event','slip_event']
fig,axs=plt.subplots(2,3,figsize=(17,11))
for k,t in enumerate(tasks):
    for col in range(3):
        ax=axs[k,col]
        if col==0:vals=[s['means']['matched_nominal'][t][m] for m in names];label='Mean Brier loss'
        elif col==1:vals=[1000*s['timing_gpu'][m]['task_medians_s'][t] for m in names];label='GPU online median (ms)'
        else:vals=[100*s['ranking']['matched_nominal'][m].get('seed2_agreement',[np.nan]*2)[k] for m in names];label='Independent-seed Top-1 agreement (%)'
        ax.barh(range(len(names)),vals,color=['#b9582b' if m=='adaptive' else '#447fa0' for m in names]);ax.set_yticks(range(len(names)),names if col==0 else []);ax.set_ylim(len(names)-.5,-.5);ax.set_xlabel(label);ax.set_title(t);hi=np.nanmax(vals);ax.set_xlim(0,110 if col==2 else hi*1.24);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
        for j,v in enumerate(vals):
            if np.isfinite(v):ax.text(v+hi*.015,j,f'{v:.4f}' if col==0 else f'{v:.1f}',va='center',fontsize=8)
            else:ax.text(3,j,'deterministic',va='center',fontsize=8,color='#777777')
fig.tight_layout();fig.savefig(R/'tradeoff.png',dpi=160);plt.close(fig)
p=np.load(R/'policies/matched_nominal.npz');fig,axs=plt.subplots(2,2,figsize=(10,8));counts=[]
for name in ['adaptive','balanced']:
    for k in range(2):
        mm=p[name+'_meta'][:,k];counts.append(np.array([[np.sum((mm[:,0]==K)&(mm[:,1]==m)) for m in [128,256,512,1024]] for K in [16,32,64,128]]))
top=max(c.max() for c in counts)
for row,name in enumerate(['adaptive','balanced']):
    for k,t in enumerate(tasks):
        c=counts[row*2+k];ax=axs[row,k];im=ax.imshow(c,vmin=0,vmax=top,cmap='Blues');ax.set_xticks(range(4),[512,1024,2048,4096]);ax.set_yticks(range(4),[16,32,64,128]);ax.set_xlabel('Total outer samples M');ax.set_ylabel('Components per region K');ax.set_title(name+' / '+t)
        for i in range(4):
            for j in range(4):ax.text(j,i,str(c[i,j]),ha='center',va='center',color='white' if c[i,j]>top*.55 else 'black')
fig.suptitle('Final budgets on 512 fresh matched worlds');fig.tight_layout();fig.savefig(R/'budgets.png',dpi=180);plt.close(fig);print('Figures generated')
