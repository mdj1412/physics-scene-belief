import os
os.environ['BELIEF_DEVICE']='cpu'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import TASKS,SIZES
from common import *
from run_policy import CELLS
METHODS=['region32','global32','region64','region_cf32','native','pair','nested128','nested512','qmc64_native','none','query','factual','probe','region32_compact']
def bootstrap(a,b=None,n=10000,alpha=.05):
    rng=np.random.default_rng(94197);vals=[]
    for start in range(0,n,1000):
        ix=rng.integers(len(a),size=(min(1000,n-start),len(a)));v=a[ix].mean(1)
        if b is not None:v=v/b[ix].mean(1)
        vals.append(v)
    vals=np.concatenate(vals);return dict(mean=float(a.mean() if b is None else a.mean()/b.mean()),low=float(np.quantile(vals,alpha/2)),high=float(np.quantile(vals,1-alpha/2)))
def action(name,p,n):
    if name in ['none','query','factual','probe']:return np.full((n,5),['none','query','factual','probe'].index(name),int)
    return p['region32' if name=='region32_compact' else name].astype(int)
def group_loss(pred,target):
    err=(pred-target)**2;return np.stack([.5*v.sum(-1) for v in np.split(err,np.cumsum(SIZES)[:-1],axis=-1)],-1)
def choose_loss(la,a):return la[np.arange(len(a))[:,None],a,np.arange(5)[None,:]]
def run():
    require();truth=np.load(R/'data/test/private.npz');n=len(truth['theta']);L={};A={};summary=dict(means={},actions={},primary={},secondary={},noninferiority={},audit={},diagnostics={})
    for cell,cond,law in CELLS:
        p=np.load(R/'policies'/f'{cell}.npz');ev=np.load(R/'evaluations'/f'{cell}_16384.npz');la=group_loss(ev['pred'],truth['target'][:,cond,None,None,:]).mean(2)
        la64=group_loss(np.load(R/'evaluations'/f'{cell}_65536.npz')['pred'],truth['target'][:,cond,None,None,:]).mean(2) if law=='nominal' else None
        for m in METHODS:
            if m=='qmc64_native' and law!='nominal':continue
            a=action(m,p,n);A[cell+'__'+m]=a
            if m=='region32_compact':ls=group_loss(np.load(R/'evaluations'/f'{cell}_compact.npz')['pred'],truth['target'][:,cond,None,:]).mean(1)
            else:ls=choose_loss(la64 if m=='qmc64_native' else la,a)
            L[cell+'__'+m]=ls
        summary['means'][cell]={task:{m:float(L[cell+'__'+m][:,k].mean()) for m in METHODS if cell+'__'+m in L} for k,task in enumerate(TASKS)}
        summary['actions'][cell]={task:{m:np.bincount(A[cell+'__'+m][:,k],minlength=4).tolist() for m in METHODS if cell+'__'+m in A} for k,task in enumerate(TASKS)}
        selectedess=ev['ess'][np.arange(n)[:,None],p['region32'].astype(int),:]
        summary['diagnostics'][cell]=dict(factual_ess_median=float(np.median(p['ess'])),selected_ess_medians=np.median(selectedess,axis=(0,2)).tolist(),components_min=np.min(p['mixture_meta'][:,:,0],axis=0).tolist(),components_max=np.max(p['mixture_meta'][:,:,0],axis=0).tolist(),omitted_mass_max=float(p['mixture_meta'][:,:,1].max()),payload_max=p['mixture_meta'][:,:,2].max(0).tolist())
        audit=np.load(R/'audits'/f'{cell}.npz');nn=len(audit['region32_Q8']);ar={}
        for name in ['region32_Q8','region32_seed2','nested2048']:
            main='nested512' if name=='nested2048' else 'region32';aa=audit[name];loss=choose_loss(la[:nn],aa)
            ar[name]=dict(agreement=(aa==p[main][:nn]).mean(0).tolist(),loss_change=(loss-L[cell+'__'+main][:nn]).mean(0).tolist())
        ref=audit['nested2048_risk'];ar['reference_regret']={}
        for m in ['region32','global32','region64','region_cf32','native','nested128','nested512','pair']:
            aa=A[cell+'__'+m][:nn];rr=ref[np.arange(nn)[:,None],aa,np.arange(5)[None,:]]-ref[:,1:].min(1);ar['reference_regret'][m]=rr.mean(0).tolist()
        summary['audit'][cell]=ar
        for k,task in enumerate(TASKS):
            for a,b in [('region32','native'),('region32','nested512'),('region32','global32'),('region32','pair'),('region32','query'),('region32','probe'),('region64','region32'),('region_cf32','region32'),('region32_compact','region32')]:
                summary['secondary'][f'{cell}/{task}/{a}-{b}']=bootstrap(L[cell+'__'+a][:,k]-L[cell+'__'+b][:,k])
    for k in [2,4]:
        a=L['matched_nominal__region32'][:,k]
        for m in ['native','nested512','global32','pair','query','probe']:
            b=L['matched_nominal__'+m][:,k];summary['primary'][TASKS[k]+'/region32-'+m]=dict(delta=bootstrap(a-b,n=30000,alpha=.05/12),relative_mean_change=float(a.mean()/b.mean()-1))
        summary['noninferiority'][TASKS[k]]=bootstrap(a,L['matched_nominal__nested512'][:,k],n=30000)
    summary['changed_worlds']=int((truth['theta'][:,1]==2).sum());summary['controls']=int((truth['theta'][:,1]!=2).sum())
    summary['changed_means']={cell:{task:{m:float(L[cell+'__'+m][truth['theta'][:,1]==2,k].mean()) for m in ['region32','global32','native','nested512','pair','probe']} for k,task in enumerate(TASKS)} for cell,_,_ in CELLS}
    for dev in ['gpu','cpu']:summary['timing_'+dev]=read(R/f'timing_{dev}.json')
    ratio=summary['timing_gpu']['nested512']['median_s']/summary['timing_gpu']['region32']['median_s'];summary['approximation_screen']=dict(loss_criteria=all(v['high']<=1.05 for v in summary['noninferiority'].values()),gpu_speedup=ratio,passes=all(v['high']<=1.05 for v in summary['noninferiority'].values()) and ratio>=2)
    save(R/'world_losses.npz',**L);save(R/'decision_actions.npz',**A);write(R/'summary.json',summary)
    names=['region32','global32','region64','region_cf32','native','nested128','nested512','pair','query','probe'];fig,axs=plt.subplots(2,2,figsize=(12,9))
    for row,task in enumerate(['contact_event','slip_event']):
        for col in [0,1]:
            ax=axs[row,col];values=[summary['means']['matched_nominal'][task][m] if col==0 else summary['timing_gpu'][m]['task_medians_s'][task]*1000 for m in names]
            ax.barh(range(len(names)),values,color=['#b9582b' if m=='region32' else '#447fa0' for m in names]);ax.set_yticks(range(len(names)),names if col==0 else []);ax.invert_yaxis();ax.set_xlim(0,max(values)*1.22)
            for j,v in enumerate(values):ax.text(v+max(values)*.015,j,f'{v:.4f}' if col==0 else f'{v:.2f}',va='center',fontsize=8)
            ax.set_xlabel('Brier loss' if col==0 else 'GPU latency (ms)');ax.set_title(task+(' — loss' if col==0 else ' — online compute'));ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    fig.tight_layout();fig.savefig(R/'tradeoff.png',dpi=180);plt.close(fig)
    print(json.dumps(dict(primary=summary['primary'],noninferiority=summary['noninferiority'],approximation_screen=summary['approximation_screen']),indent=2),flush=True)
if __name__=='__main__':run()
