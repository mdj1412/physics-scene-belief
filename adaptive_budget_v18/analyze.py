import os
os.environ['BELIEF_DEVICE']='cpu'
from common import *
from run_policy import CELLS,ETASKS,EIDX,ADAPT,FIXED,MIX,NEST
METHODS=MIX+['native','qmc64_native']+list(NEST)+['query','probe','factual','none']
def boot(a,b=None,n=10000,alpha=.05):
    rng=np.random.default_rng(114197);values=[]
    for start in range(0,n,1000):
        ix=rng.integers(len(a),size=(min(1000,n-start),len(a)));v=a[ix].mean(1)
        if b is not None:v=v/b[ix].mean(1)
        values.append(v)
    v=np.concatenate(values);return dict(mean=float(a.mean() if b is None else a.mean()/b.mean()),low=float(np.quantile(v,alpha/2)),high=float(np.quantile(v,1-alpha/2)))
def choose(la,a):return la[np.arange(len(a))[:,None],a,np.arange(2)[None,:]]
def run():
    require();truth=np.load(R/'data/test/private.npz');L={};A={};s=dict(means={},actions={},primary={},secondary={},ranking={},budgets={},calibration={},ratios={})
    for cell,cond,law in CELLS:
        pol=np.load(R/'policies'/f'{cell}.npz');aud=np.load(R/'audits'/f'{cell}.npz');aud2=np.load(R/'independent_audit.npz') if cond==0 else None;tar=truth['target'][:,cond][:,EIDX];la={}
        for N in [16384,65536]:
            pr=np.load(R/'evaluations'/f'{cell}_{N}.npz')['pred'];la[N]=(.5*(pr-tar[:,None,None,:])**2).mean(2)
        for m in METHODS:
            a=np.full((512,2),['none','query','factual','probe'].index(m),int) if m in ['none','query','factual','probe'] else pol[m].astype(int)
            A[cell+'__'+m]=a;L[cell+'__'+m]=choose(la[65536 if m=='qmc64_native' else 16384],a)
        s['means'][cell]={t:{m:float(L[cell+'__'+m][:,k].mean()) for m in METHODS} for k,t in enumerate(ETASKS)}
        s['actions'][cell]={t:{m:np.bincount(A[cell+'__'+m][:,k],minlength=4).tolist() for m in METHODS} for k,t in enumerate(ETASKS)}
        s['budgets'][cell]={};s['ranking'][cell]={};ref=aud['reference_risk'];nref=len(ref)
        for m in METHODS:
            if m=='none':continue
            aa=A[cell+'__'+m][:nref];reg=ref[np.arange(nref)[:,None],aa-1,np.arange(2)[None,:]]-ref.min(1)
            s['ranking'][cell][m]=dict(reference_agreement=(aa==aud['reference']).mean(0).tolist(),reference_risk_regret=reg.mean(0).tolist())
        for m in ADAPT:
            meta=pol[m+'_meta'];s['budgets'][cell][m]={}
            for k,t in enumerate(ETASKS):
                mm=meta[:,k];hist=pol[m+'_history'];last=hist[np.arange(512),k,mm[:,2].astype(int)-1]
                s['budgets'][cell][m][t]=dict(K_counts={str(v):int((mm[:,0]==v).sum()) for v in [16,32,64,128]},M_counts={str(4*v):int((mm[:,1]==v).sum()) for v in [128,256,512,1024]},reason_counts=np.bincount(mm[:,3].astype(int),minlength=4).tolist(),mean_stages=float(mm[:,2].mean()),mean_K_steps=float(mm[:,4].mean()),mean_M_steps=float(mm[:,5].mean()),mean_work=float(pol[m+'_work'][:,k].mean()),cap_compression_dominant=int(((mm[:,3]==1)&(last[:,6]>=last[:,7])).sum()),cap_outer_dominant=int(((mm[:,3]==1)&(last[:,7]>last[:,6])).sum()))
        if cond==0:
            for m in MIX+list(NEST):
                aa=aud2[m+'_seed2'].astype(int);ls=choose(la[16384],aa);L[cell+'__'+m+'_seed2']=ls
                s['ranking'][cell][m]['seed2_agreement']=(aa==A[cell+'__'+m]).mean(0).tolist()
                s['ranking'][cell][m]['seed2_loss_delta']=[boot(ls[:,k]-L[cell+'__'+m][:,k]) for k in range(2)]
            agr=(aud2['adaptive_seed2']==pol['adaptive']).astype(float);cmp=(aud2['component_only_seed2']==pol['component_only']).astype(float)
            s['repeatability_improvement']={t:boot(agr[:,k]-cmp[:,k]) for k,t in enumerate(ETASKS)}
        # Same-sample finite-bank compression discrepancy at the recorded worst competitor.
        s['calibration'][cell]={}
        for k,t in enumerate(ETASKS):
            compression=[];outer=[];ambiguous=0
            for i in range(512):
                meta=pol['adaptive_meta'][i,k];rr=pol['adaptive_risk'][i,k];h=pol['adaptive_history'][i,k,int(meta[2])-1];best=rr.argmin();g=rr-rr[best]
                pairs=[a for a in range(3) if a!=best and abs(g[a]-h[5])<1e-12]
                if len(pairs)!=1:ambiguous+=1;continue
                a=pairs[0];j=[128,256,512,1024].index(int(meta[1]));fr=pol['full_prefix_risk'][i,j,:,k];err=abs(g[a]-(fr[a]-fr[best]));compression.append([h[6],err])
                if cond==0 and i<64:
                    ar=aud2['isolated_outer_risk'][i,k];outer.append([h[7],abs(g[a]-(ar[a]-ar[best]))])
            def describe(rows):
                q=np.array(rows)
                return dict(n=len(q),median_proxy=float(np.median(q[:,0])),median_observed_discrepancy=float(np.median(q[:,1])),observed_exceeds_proxy_fraction=float((q[:,1]>q[:,0]+1e-12).mean())) if len(q) else None
            s['calibration'][cell][t]=dict(ambiguous_pair_count=ambiguous,compression=describe(compression),outer_perturbation=describe(outer))
        for k,t in enumerate(ETASKS):
            for m in METHODS:
                if m!='adaptive':s['secondary'][cell+'/'+t+'/adaptive-'+m]=boot(L[cell+'__adaptive'][:,k]-L[cell+'__'+m][:,k])
    for k,t in enumerate(ETASKS):
        a=L['matched_nominal__adaptive'][:,k]
        for m in ['fixed128_512','fixed128_4096','global128_4096','nested2048','balanced','query' if k==0 else 'probe']:s['primary'][t+'/adaptive-'+m]=boot(a-L['matched_nominal__'+m][:,k],n=30000,alpha=.05/12)
        s['ratios'][t]={m:boot(a,L['matched_nominal__'+m][:,k],n=30000) for m in ['nested2048','fixed128_4096']}
    for dev in ['gpu','cpu']:s['timing_'+dev]=read(R/f'timing_fast_{dev}.json')
    s['timing_original_gpu']=read(R/'timing_gpu.json')
    tt=s['timing_gpu'];speed=tt['nested2048']['median_s']/tt['adaptive']['median_s'];saving=1-tt['adaptive']['median_s']/tt['fixed128_4096']['median_s'];ok=all(s['ratios'][t]['nested2048']['high']<=1.05 for t in ETASKS);fixedok=all(s['ratios'][t]['fixed128_4096']['high']<=1.05 for t in ETASKS);repeat=s['repeatability_improvement']['slip_event']['mean']>=.05
    s['screens']=dict(nested_loss=ok,nested_speedup=speed,approximation_pass=ok and speed>=2,fixed_loss=fixedok,fixed_time_saving=saving,stick_repeatability_plus5pp=repeat,adaptivity_pass=fixedok and saving>=.2 and repeat)
    s['controls']=int((truth['theta'][:,1]!=2).sum());save(R/'world_losses.npz',**L);save(R/'decision_actions.npz',**A);write(R/'summary.json',s);print(json.dumps(dict(primary=s['primary'],screens=s['screens']),indent=2),flush=True)
if __name__=='__main__':run()
