import os
os.environ['BELIEF_DEVICE']='cpu'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import TASKS,SIZES
from common import *
from run_policy import CELLS

METHODS=['native','pair','unfiltered','agnostic','nested128','nested512','variance512','none','query','factual','probe','qmc64_native','qmc64_pair','qmc64_probe','qmc64_query']
def ci(v,n=10000,alpha=.05):
    v=np.asarray(v);rng=np.random.default_rng(84197);vals=[]
    for _ in range(0,n,1000):vals.append(v[rng.integers(len(v),size=(min(1000,n-len(vals)*1000),len(v)))].mean(1))
    bs=np.concatenate(vals);return dict(mean=float(v.mean()),low=float(np.quantile(bs,alpha/2)),high=float(np.quantile(bs,1-alpha/2)))
def contrast(a,b,primary=False):
    diff=ci(a-b,n=30000 if primary else 10000,alpha=.005 if primary else .05)
    return dict(delta=diff,relative_mean_change=float(a.mean()/b.mean()-1) if b.mean()>0 else None)
def action_for(name,p,n):
    fixed={'none':0,'query':1,'factual':2,'probe':3,'qmc64_query':1,'qmc64_probe':3}
    return np.full((n,5),fixed[name],int) if name in fixed else p[name].astype(int)
def by_action_loss(pred,target):
    raw=(pred-target[:,None,None,:])**2
    return np.stack([.5*raw[...,sl].sum(-1) for sl in np.split(np.arange(8),np.cumsum(SIZES)[:-1])],-1).mean(2)
def select_loss(la,a):return la[np.arange(len(a))[:,None],a,np.arange(5)[None,:]]
def run():
    require();priv=np.load(R/'data/test/private.npz');n=len(priv['theta']);losses={};actions={};peraction={};summary=dict(means={},actions={},primary={},secondary={},mismatch_decomposition={},audit={},diagnostics={})
    for cell,condition,law in CELLS:
        p=np.load(R/'policies'/f'{cell}.npz');la=by_action_loss(np.load(R/'evaluations'/f'{cell}_16384.npz')['pred'],priv['target'][:,condition]);peraction[cell]=la
        la64=by_action_loss(np.load(R/'evaluations'/f'{cell}_65536.npz')['pred'],priv['target'][:,condition]) if law=='nominal' else None
        for name in METHODS:
            if name.startswith('qmc64') and law!='nominal':continue
            a=action_for(name,p,n);actions[cell+'__'+name]=a;losses[cell+'__'+name]=select_loss(la64 if name.startswith('qmc64') else la,a)
        summary['means'][cell]={task:{name:float(losses[cell+'__'+name][:,k].mean()) for name in METHODS if cell+'__'+name in losses} for k,task in enumerate(TASKS)}
        summary['actions'][cell]={task:{name:np.bincount(actions[cell+'__'+name][:,k],minlength=4).tolist() for name in METHODS if cell+'__'+name in actions} for k,task in enumerate(TASKS)}
        summary['diagnostics'][cell]=dict(factual_ess_median=float(np.median(p['ess'])),factual_ess_p05=float(np.quantile(p['ess'],.05)),eligible_mean=float(np.mean(p['eligible'])),fallback_count=np.sum(p['fallback'],axis=0).tolist(),contact_rate=float((priv['target'][:,condition,4]>0).mean()),stick_rate=float((priv['target'][:,condition,7]>0).mean()))
        audit=np.load(R/'audits'/f'{cell}.npz');na=len(audit['pair512']);reference=audit['nested2048_b_risk'];entry={}
        for key in ['pair512','nested2048','nested2048_b']+(['qmc64_nested512'] if law=='nominal' else []):
            base='pair' if key=='pair512' else 'nested512';a=audit[key];decode=la64[:na] if key.startswith('qmc64') else la[:na]
            ls=select_loss(decode,a);old=losses[cell+'__'+base][:na]
            entry[key]=dict(agreement_per_task=(a==p[base][:na]).mean(0).tolist(),loss_change_per_task=(ls-old).mean(0).tolist())
        entry['reference_regret']={}
        for name in ['native','pair','nested128','nested512','query','probe']:
            a=actions[cell+'__'+name][:na];selected=reference[np.arange(na)[:,None],a,np.arange(5)[None,:]]
            regret=selected-reference[:,1:].min(1);entry['reference_regret'][name]=regret.mean(0).tolist()
        summary['audit'][cell]=entry
        for k,task in enumerate(TASKS):
            for a,b in [('pair','native'),('pair','probe'),('pair','query'),('pair','nested512'),('nested512','native'),('pair','unfiltered'),('pair','agnostic')]:
                summary['secondary'][f'{cell}/{task}/{a}-{b}']=contrast(losses[cell+'__'+a][:,k],losses[cell+'__'+b][:,k])
    for k in [2,4]:
        for a,b in [('pair','native'),('pair','probe'),('pair','query'),('pair','nested512'),('nested512','native')]:
            key=TASKS[k]+'/'+a+'-'+b;summary['primary'][key]=contrast(losses['matched_nominal__'+a][:,k],losses['matched_nominal__'+b][:,k],True)
    # Same shifted data and target. Correct decoder with nominal choices isolates the decoder effect.
    for name in ['native','pair','nested512','probe','query']:
        nom=losses['shifted_nominal__'+name];cor=losses['shifted_correct__'+name]
        swap=select_loss(peraction['shifted_correct'],actions['shifted_nominal__'+name]);losses['shifted_correct_decoder_nominal_choices__'+name]=swap
        summary['mismatch_decomposition'][name]={task:dict(total=ci(nom[:,k]-cor[:,k]),decoder=ci(nom[:,k]-swap[:,k]),selection_under_correct_decoder=ci(swap[:,k]-cor[:,k])) for k,task in enumerate(TASKS)}
        for k,task in enumerate(TASKS):
            assert np.allclose((nom[:,k]-swap[:,k])+(swap[:,k]-cor[:,k]),nom[:,k]-cor[:,k],atol=1e-14)
    changed=priv['theta'][:,1]==2
    summary['changed_worlds']=int(changed.sum());summary['control_worlds']=int((~changed).sum());summary['changed_means']={}
    for cell,condition,law in CELLS:
        summary['changed_means'][cell]={task:{name:float(losses[cell+'__'+name][changed,k].mean()) for name in ['native','pair','probe','nested512']} for k,task in enumerate(TASKS)}
    summary['failure_screen']={TASKS[k]: bool(summary['primary'][TASKS[k]+'/nested512-native']['delta']['high']<0 and summary['primary'][TASKS[k]+'/nested512-native']['relative_mean_change']<=-.10) for k in [2,4]}
    summary['pair_screen']={TASKS[k]:all(summary['primary'][TASKS[k]+'/pair-'+b]['delta']['high']<0 for b in ['native','probe','query','nested512']) for k in [2,4]}
    for device in ['gpu','cpu']:
        if (R/f'timing_{device}.json').exists():summary['timing_'+device]=read(R/f'timing_{device}.json')
    save(R/'world_losses.npz',**losses);save(R/'decision_actions.npz',**actions);write(R/'summary.json',summary)
    # Static scientific figures: report CIs remain authoritative.
    names=['native','pair','nested512','query','probe'];fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ax,k in zip(axs,[2,4]):
        task=TASKS[k];x=np.arange(3);width=.15
        for j,name in enumerate(names):ax.bar(x+(j-2)*width,[summary['means'][c][task][name] for c,_,_ in CELLS],width,label=name)
        ax.set_xticks(x,['Matched\nnominal','Shifted\nnominal','Shifted\ncorrect-law']);ax.set_title(task);ax.set_ylabel('Brier loss (lower is better)')
    handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=5);fig.tight_layout(rect=(0,0,1,.90));fig.savefig(R/'event_losses.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ax,k in zip(axs,[2,4]):
        task=TASKS[k]
        for name in ['native','pair','nested128','nested512','qmc64_native','qmc64_pair','probe','query']:
            xx=summary['timing_gpu'][name]['task_medians_s'][task]*1000;yy=summary['means']['matched_nominal'][task][name]
            ax.scatter(xx,yy);ax.annotate(name,(xx,yy),xytext=(3,5),textcoords='offset points',fontsize=8)
        ax.set_xscale('log');ax.set_title(task);ax.set_xlabel('GPU latency ms, log scale');ax.set_ylabel('Matched Brier loss');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(R/'loss_latency.png',dpi=180);plt.close(fig)
    print(json.dumps(dict(primary=summary['primary'],failure_screen=summary['failure_screen'],pair_screen=summary['pair_screen']),indent=2),flush=True)
if __name__=='__main__':run()
