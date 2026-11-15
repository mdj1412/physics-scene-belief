import os
os.environ['BELIEF_DEVICE']='cpu'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *
from run_policy import CELLS,ETASKS,EIDX,ADAPT
METHODS=ADAPT+['region32','global32','native','qmc64_native','nested128','nested512','query','probe','factual','none']
def boot(a,b=None,n=10000,alpha=.05):
    rng=np.random.default_rng(104197);v=[]
    for start in range(0,n,1000):
        ix=rng.integers(len(a),size=(min(1000,n-start),len(a)));s=a[ix].mean(1)
        if b is not None:s=s/b[ix].mean(1)
        v.append(s)
    v=np.concatenate(v);return dict(mean=float(a.mean() if b is None else a.mean()/b.mean()),low=float(np.quantile(v,alpha/2)),high=float(np.quantile(v,1-alpha/2)))
def choose(la,a):return la[np.arange(len(a))[:,None],a,np.arange(2)[None,:]]
def run():
    require();priv=np.load(R/'data/test/private.npz');L={};A={};summary=dict(means={},actions={},primary={},secondary={},ranking={},expansion={},ratios={},diagnostics={})
    for cell,cond,law in CELLS:
        pol=np.load(R/'policies'/f'{cell}.npz');aud=np.load(R/'audits'/f'{cell}.npz');truth=priv['target'][:,cond][:,EIDX];las={}
        for N in [16384,65536]:
            pred=np.load(R/'evaluations'/f'{cell}_{N}.npz')['pred'];las[N]=(.5*(pred-truth[:,None,None,:])**2).mean(2)
        for m in METHODS:
            a=np.full((len(truth),2),['none','query','factual','probe'].index(m),int) if m in ['none','query','factual','probe'] else pol[m].astype(int)
            A[cell+'__'+m]=a;L[cell+'__'+m]=choose(las[65536 if m=='qmc64_native' else 16384],a)
        summary['means'][cell]={task:{m:float(L[cell+'__'+m][:,k].mean()) for m in METHODS} for k,task in enumerate(ETASKS)}
        summary['actions'][cell]={task:{m:np.bincount(A[cell+'__'+m][:,k],minlength=4).tolist() for m in METHODS} for k,task in enumerate(ETASKS)}
        summary['expansion'][cell]={};summary['ranking'][cell]={}
        ref=aud['nested2048_risk'];nref=len(ref);refchoice=aud['nested2048'];rbase=ref.min(1)
        for m in ADAPT:
            meta=pol[m+'_meta'];hist=pol[m+'_history'];bytask={}
            for k,task in enumerate(ETASKS):
                mm=meta[:,k];h=hist[np.arange(len(meta)),k,mm[:,1].astype(int)-1];cap=mm[:,2]==1;settled=mm[:,2]==0
                bytask[task]=dict(K_counts={str(kk):int((mm[:,0]==kk).sum()) for kk in [8,16,32,64,128]},mean_K=float(mm[:,0].mean()),mean_stages=float(mm[:,1].mean()),mean_total_replica_components_processed=float((2*(mm[:,3]/mm[:,0])*np.nansum(hist[:,k,:,3],axis=1)).mean()),reason_counts=np.bincount(mm[:,2].astype(int),minlength=4).tolist(),max_payload_bytes=float(mm[:,4].max()),max_omitted_mass=float(mm[:,5].max()),cap_sampling_alone_exceeds_gap=int((cap & (h[:,5]>=h[:,4])).sum()),settled_reference_n=int(settled[:nref].sum()),settled_reference_disagreements=int(((A[cell+'__'+m][:nref,k]!=refchoice[:,k])&settled[:nref]).sum()))
            summary['expansion'][cell][m]=bytask
        for m in METHODS:
            a=A[cell+'__'+m][:nref];reg=ref[np.arange(nref)[:,None],a-1,np.arange(2)[None,:]]-rbase if m!='none' else None
            if m!='none':summary['ranking'][cell][m]=dict(reference_top1_agreement=(a==refchoice).mean(0).tolist(),reference_risk_regret=reg.mean(0).tolist())
        if cond==0:
            for m in ADAPT+['nested512']:
                aa=aud[m+'_seed2'].astype(int);L[cell+'__'+m+'_seed2']=choose(las[16384],aa)
                summary['ranking'][cell][m]['seed2_agreement']=(aa==A[cell+'__'+m]).mean(0).tolist()
                summary['ranking'][cell][m]['seed2_loss_delta']=[boot(L[cell+'__'+m+'_seed2'][:,k]-L[cell+'__'+m][:,k]) for k in range(2)]
            for m in ['adaptive','adaptive_global','fixed128','global128','native','nested512']:
                aa=aud[m+'_bank2'].astype(int);la=choose(las[16384][:len(aa)],aa)
                summary['ranking'][cell][m]['bank2_agreement']=(aa==A[cell+'__'+m][:len(aa)]).mean(0).tolist()
                summary['ranking'][cell][m]['bank2_loss_delta']=(la-L[cell+'__'+m][:len(aa)]).mean(0).tolist()
            agree_a=(aud['adaptive_seed2']==pol['adaptive']).astype(float);agree_b=(aud['fixed32_seed2']==pol['fixed32']).astype(float)
            summary['repeatability_improvement']={task:boot(agree_a[:,k]-agree_b[:,k]) for k,task in enumerate(ETASKS)}
        for k,task in enumerate(ETASKS):
            for m in METHODS:
                if m!='adaptive':summary['secondary'][cell+'/'+task+'/adaptive-'+m]=boot(L[cell+'__adaptive'][:,k]-L[cell+'__'+m][:,k])
        summary['diagnostics'][cell]=dict(factual_ess_median=float(np.median(pol['ess'])))
    for k,task in enumerate(ETASKS):
        a=L['matched_nominal__adaptive'][:,k]
        for m in ['native','nested512','adaptive_global','fixed128','global128','query' if k==0 else 'probe']:
            summary['primary'][task+'/adaptive-'+m]=boot(a-L['matched_nominal__'+m][:,k],n=30000,alpha=.05/12)
        summary['ratios'][task]={m:boot(a,L['matched_nominal__'+m][:,k],n=30000) for m in ['nested512','fixed128']}
    for dev in ['gpu','cpu']:
        summary['timing_original_'+dev]=read(R/f'timing_{dev}.json')
        summary['timing_'+dev]=read(R/f'timing_fast_{dev}.json')
    tt=summary['timing_gpu'];speed=tt['nested512']['median_s']/tt['adaptive']['median_s'];saving=1-tt['adaptive']['median_s']/tt['fixed128']['median_s']
    lossok=all(summary['ratios'][t]['nested512']['high']<=1.05 for t in ETASKS)
    repeatok=all(summary['repeatability_improvement'][t]['mean']>=.10 for t in ETASKS)
    fixedok=all(summary['ratios'][t]['fixed128']['high']<=1.05 for t in ETASKS)
    summary['screens']=dict(nested_loss=lossok,nested_speedup=speed,approximation_pass=lossok and speed>=2,repeatability_plus_10pp=repeatok,fixed128_time_saving=saving,fixed128_loss=fixedok,adaptivity_pass=repeatok and saving>=.2 and fixedok)
    summary['changed_worlds']=int((priv['theta'][:,1]==2).sum());summary['controls']=int((priv['theta'][:,1]!=2).sum())
    save(R/'world_losses.npz',**L);save(R/'decision_actions.npz',**A);write(R/'summary.json',summary)
    names=['adaptive','adaptive_global','fixed32','fixed128','global128','global32','native','qmc64_native','nested128','nested512','query','probe']
    fig,axs=plt.subplots(2,3,figsize=(16,9))
    for k,task in enumerate(ETASKS):
        for col in range(3):
            ax=axs[k,col]
            if col==0:vals=[summary['means']['matched_nominal'][task][m] for m in names];label='Brier loss'
            elif col==1:vals=[tt[m]['task_medians_s'][task]*1000 for m in names];label='GPU online latency (ms)'
            else:vals=[100*summary['ranking']['matched_nominal'][m].get('seed2_agreement',[np.nan]*2)[k] for m in names];label='Top-1 agreement across acquisition seeds (%)'
            ax.barh(range(len(names)),vals,color=['#b9582b' if m=='adaptive' else '#447fa0' for m in names]);ax.set_yticks(range(len(names)),names if col==0 else []);ax.set_ylim(len(names)-.5,-.5);ax.set_title(task);ax.set_xlabel(label);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
            hi=np.nanmax(vals);ax.set_xlim(0,hi*1.22)
            for j,v in enumerate(vals):
                if np.isfinite(v):ax.text(v+hi*.015,j,f'{v:.4f}' if col==0 else f'{v:.1f}',va='center',fontsize=8)
    fig.tight_layout();fig.savefig(R/'tradeoff.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(primary=summary['primary'],screens=summary['screens'],repeatability=summary['repeatability_improvement']),indent=2),flush=True)
if __name__=='__main__':run()
