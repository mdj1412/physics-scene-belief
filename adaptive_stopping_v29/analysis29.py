"""v29 equal-budget risk, allocation and ranking diagnostics from the frozen block outputs."""
import hashlib,json,itertools,os
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
P=Path(__file__).resolve().parent
R=P/'results'
CONFIG=json.loads((P/'design.json').read_text())
METHODS=['adaptive','fixed4','random','uncertainty','known']
SIZES=[2,4,8]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
def table25(common,counts):
    # Gather one hypothetical budget per environment per deployed method.
    slots={v:i for i,v in enumerate(SIZES)}
    ids=np.vectorize(slots.__getitem__)(counts)
    return np.take_along_axis(common,ids[:,None,:],axis=2)
def paired_rank(x,y):
    if len(x)<3 or np.ptp(x)<1.e-14 or np.ptp(y)<1.e-14:return np.nan
    return float(np.corrcoef(rankdata(x),rankdata(y))[0,1])
def bnds(draw,point):
    sd=draw.std(0,ddof=1);ratio=np.abs(draw-point)/np.where(sd>1.e-14,sd,1.)
    crit=np.quantile(np.max(ratio,axis=-1),.95)
    return np.stack([point-crit*sd,point+crit*sd],axis=-1)
def ci(draw):return np.quantile(draw,[.025,.975],axis=0).T.tolist()
def means(x):return np.asarray(x).mean(0).tolist()
def mask_audit(action):
    return (action==3)
def stage_parts(risks,counts):
    diffs=risks-risks[:,:,1:2]
    by=diffs.mean(1)
    return [float(by[counts==n,i].sum()/len(counts)) for i,n in enumerate(SIZES)]
def opt_cost(x):
    # Same multiset, 420 legal assignments. Rates for oracle's winning assignment reflect evaluation noise.
    if not hasattr(opt_cost,'ids'):
        opt_cost.ids=np.array(sorted(set(itertools.permutations([0,0,0,0,1,1,2,2]))))
    costs=x.mean(1)[np.arange(8)[None,:],opt_cost.ids].mean(1)
    return float(costs.min())
def npval(v):return None if not np.isfinite(v) else float(v)
def clean(obj):
    if isinstance(obj,dict):return {k:clean(v) for k,v in obj.items()}
    if isinstance(obj,(tuple,list)):return [clean(v) for v in obj]
    if isinstance(obj,np.ndarray):return clean(obj.tolist())
    if isinstance(obj,(np.bool_,)):return bool(obj)
    if isinstance(obj,(np.integer,)):return int(obj)
    if isinstance(obj,(float,np.floating)):return npval(obj)
    return obj
def quota_from_gap(gap,quota=13):
    order=np.argsort(-gap,axis=1,kind='stable')
    q=np.zeros_like(gap,dtype=np.int8)+1
    np.put_along_axis(q,order[:,:quota],3,axis=1)
    return q
def record(idx):
    path=R/f'block_{idx:03d}'
    m=load(path/'metrics.json')
    assert sha(path/'metrics.npz')==m['npz_sha256'] and sha(path/'choice_lock.npz')==m['choices_sha256']
    with np.load(path/'metrics.npz') as z:
        out={k:z[k] for k in z if k!='probabilities'}
        if 'probabilities' in z:
            pp=z['probabilities'];labels=z['label']
            assert np.max(abs(np.square(pp-labels[:,:,None,None,None]).mean(-1)-z['cost']))<1e-11
    counts=out['counts'];assert (counts[:,:4].sum(0)==32).all() and (counts[:,1]==4).all()
    assert list(out['methods'])==METHODS
    for j in (0,2,3):assert sorted(counts[:,j])==[2]*4+[4]*2+[8]*2
    a=out['action'];x=out['common_known'];cost=out['cost']
    def selected(c,a):return np.take_along_axis(c,(a==3).astype(int)[...,None],axis=-1)[...,0]
    assert np.allclose(x,selected(np.broadcast_to(cost[:,:,4:5,:],cost.shape),a))
    assert np.allclose(out['loss'],selected(cost,a))
    assert np.allclose(x[:,:,:4],table25(out['audit_common'],counts[:,:4]))
    q=quota_from_gap(out['audit_gap']);qk=quota_from_gap(out['known_gap'][:,:,None])
    auditq=selected(np.broadcast_to(cost[:,:,4:5,:],(*q.shape,2)),q)
    allq=np.concatenate([table25(q,counts[:,:4]),qk],axis=2)
    assert (np.sum(allq==3,axis=1)==13).all()
    qc=selected(np.broadcast_to(cost[:,:,4:5,:],cost.shape),allq);qn=selected(cost,allq)
    if 'quota_common_known' in out:assert np.allclose(qc,out['quota_common_known'])
    if 'quota_action' in out:assert np.array_equal(allq,out['quota_action'])
    return dict(m=m,v=out,q=allq,qc=qc,qn=qn,qaudit=auditq)
def main():
    assert os.environ.get('SLURM_JOB_ID')
    ds=[record(i) for i in range(len(CONFIG['bank_pairs']))];B=len(ds)
    risk=np.stack([[d['v']['common_known'].mean((0,1)),d['qc'].mean((0,1)),d['v']['loss'].mean((0,1)),d['qn'].mean((0,1))] for d in ds])
    rng=np.random.default_rng(29188019);bid=rng.integers(B,size=(10000,B))
    dep=risk[:,:2,0]-risk[:,:2,1];boot=dep[bid].mean(1);pt=dep.mean(0);bounds=bnds(boot,pt)
    # Streaming-only: paired hierarchical world + block sensitivity. Quota inference uses whole blocks.
    worldloss=np.stack([d['v']['common_known'][:,:,0]-d['v']['common_known'][:,:,1] for d in ds])
    libs=[]
    for envworld in worldloss:
        E,W=envworld.shape
        sampled=np.zeros(2048)
        for x in envworld:sampled+=rng.multinomial(W,np.full(W,1/W),size=2048)@x/W/E
        libs.append(sampled)
    hierarchical=np.array(libs)[bid,rng.integers(2048,size=bid.shape)].mean(1)
    block_controls=[]
    region=[]
    for d in ds:
        count=d['v']['counts'][:,0];checks=[]
        for ar in (d['v']['audit_common'],d['qaudit']):
            expectation=ar.mean((0,1))@np.array([.5,.25,.25])
            checks.append([expectation,opt_cost(ar),*stage_parts(ar,count)])
        block_controls.append(checks)
        region.append([d['v']['counts'][j].tolist() for j in range(8)])
    ctrl=np.asarray(block_controls)
    named=list(dict.fromkeys(e['family'] for e in CONFIG['environments']))
    bylaws={}
    for name in named:
        ix=[next(i for i,e in enumerate(d['m']['environments']) if e['family']==name) for d in ds]
        counts=np.array([d['v']['counts'][j,0] for j,d in zip(ix,ds)])
        bylaws[name]=dict(budget=counts.mean(),histogram=[np.sum(counts==n) for n in SIZES],
            common=means([d['v']['common_known'][j].mean(0) for j,d in zip(ix,ds)]),
            quota_common=means([d['qc'][j].mean(0) for j,d in zip(ix,ds)]))
    t=np.stack([d['v']['times_ms'] for d in ds]);E,T,H,K=t.shape[1:]
    tm=np.quantile(t.reshape(-1,K),.9,axis=0)
    setups=[d['m']['setup_ms'] for d in ds]
    allset=np.array([np.array(v['fit'])+np.array(v['score'])+np.array(v['engines']) for v in setups])
    if allset.shape!=(B,8,5):raise ValueError(allset.shape)
    am=t+allset[:,:,None,None,:]/128
    rts=[]
    for _ in range(1500):
        bi=rng.integers(B,size=B)
        w=rng.integers(T,size=(B,E,T))
        qt=np.quantile(t[bi[:,None,None],np.arange(E)[None,:,None],w].reshape(-1,K),.9,axis=0)
        rts.append(qt[0]/qt[1])
    probe=np.array([(d['v']['action']==3).mean((0,1)) for d in ds])
    switches=np.array([[np.mean(d['v']['action'][:,:,0]!=d['v']['action'][:,:,1]),np.mean(d['q'][:,:,0]!=d['q'][:,:,1])] for d in ds])
    bc_delta=risk[:,:,0:1]-risk
    random_delta=risk[:,:2,0]-ctrl[:,:,0]
    out=dict(methods=METHODS,blocks=B,worlds=B*8*128,risk=risk.mean(0),common_delta=pt,sim95=bounds,
        ci95=ci(boot),stream_hier_ci95=ci(hierarchical),LOO=np.array([np.delete(dep,i,axis=0).mean(0) for i in range(B)]),
        all_delta=bc_delta.mean(0),all_ci95=np.quantile(bc_delta[bid].mean(1),[.025,.975],axis=0).transpose(1,2,0),
        audit=ctrl.mean(0),allocation_parts_ci95=np.quantile(ctrl[bid].mean(1),[.025,.975],axis=0).transpose(1,2,0),
        vs_expected_random=random_delta.mean(0),vs_expected_random_ci95=ci(random_delta[bid].mean(1)),bylaw=bylaws,
        probe=probe.mean(0),switches=switches.mean(0),total_stream=probe.mean(0)+np.array([4,4,4,4,0])/128,
        p90=tm,p90_ratio_ci95=ci(rts),amortized_p90=np.quantile(am.reshape(-1,K),.9,axis=0),
        setup_means={key:np.array([v[key] for v in setups]).mean((0,1)) for key in ('fit','score','engines')},
        win_blocks=(dep<0).sum(0))
    (R/'summary.json').write_text(json.dumps(clean(out),indent=2,allow_nan=False))
if __name__=='__main__':main()
