"""Paired inference across independent target catalogs, with four fixed test cells."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import t, spearmanr

ROOT=Path(__file__).resolve().parent
CELLS=('shelf_weakening','shelf_strengthening','ridge_weakening','ridge_strengthening')
REGIMES=('strong','weak','null')
N=12


def pooled(x):
    """Equal cell weighting, independent catalog means within each cell."""
    components=np.var(x,axis=1,ddof=1)/(N*len(CELLS)**2)
    variance=components.sum()
    if variance==0:
        return {'mean':x.mean(),'ci':[x.mean(),x.mean()]}
    df=variance**2/np.sum(components**2/(N-1))
    delta=t.ppf(.975,df)*np.sqrt(variance)
    return dict(mean=x.mean(),ci=[x.mean()-delta,x.mean()+delta],df=df)


def cellstats(x, sign):
    mu=x.mean()
    sd=x.std(ddof=1)
    radius=t.ppf(1-.05/(2*len(CELLS)),N-1)*sd/np.sqrt(N)
    predictive=t.ppf(.975,N-1)*sd*np.sqrt(1+1/N)
    lo,hi=mu-radius,mu+radius
    verdict='uncertain' if lo<=0<=hi else ('confirmed' if np.sign(mu)==sign else 'refuted')
    return dict(mean=mu,sim95=[lo,hi],pointwise95=(mu+t.ppf(.975,N-1)*sd/np.sqrt(N)*np.array([-1,1])),
                sd=sd,predictive95=[mu-predictive,mu+predictive],verdict=verdict,
                wins=int((x<0).sum()),correct=int((np.sign(x)==sign).sum()),catalog_means=x)

def contrast(x, coeff):
    coeff=np.array(coeff)
    value=coeff @ x.mean(1)
    vs=coeff**2*x.var(1,ddof=1)/N
    df=vs.sum()**2/(vs**2/(N-1)).sum()
    rad=t.ppf(.975,df)*np.sqrt(vs.sum())
    return dict(value=value,ci=[value-rad,value+rad],df=df)


def timing(raw):
    assert raw.shape==(4,6,3,2)
    rng=np.random.default_rng(390217)
    ix=rng.integers(6,size=(16000,4,6))
    sample=raw[np.arange(4)[None,:,None],ix]
    rep=np.quantile(sample,.9,axis=(1,2,3))
    q=np.quantile(raw,.9,axis=(0,1,2))
    ratio=q[1]/q[0]
    ci=np.quantile(rep[:,1]/rep[:,0],[.025,.975])
    loo=[]
    for k in range(24):
        leave=np.delete(raw.reshape(24,3,2),k,axis=0)
        v=np.quantile(leave,.9,axis=(0,1))
        loo.append(v[1]/v[0])
    return dict(p90=q,p90_bounds=np.quantile(rep,[.025,.975],axis=0).T,ratio=ratio,ratio_bounds=ci,
                leave_one_ratio_minmax=[min(loo),max(loo)],passes_two_percent=bool(ci[1]<=1.02))


def main():
    freeze=json.loads((ROOT/'forecast39.json').read_text())
    completion=json.loads((ROOT/'results/completion.json').read_text())
    assert completion['blocks']==96 and completion['timing_clusters']==24
    loss=np.empty((4,N,2,2,3,2))
    decomposition=np.empty((4,N,2,2,3,2))
    bias=np.empty_like(loss)
    raw=np.empty((4,6,3,2))
    for c,cell in enumerate(CELLS):
        for i in range(N):
            for b in range(2):
                fpath=ROOT/'results'/cell/f'catalog_{i:02d}'/f'block_{b:02d}.npz'
                with np.load(fpath) as f:
                    assert f['risk'].shape==(2,3,2,8)
                    loss[c,i,b]=f['risk'].mean(-1)
                    pb=f['p_base'][:,None]
                    z=f['z'][:,None,:,None,None]
                    w=f['weights'][:,:,:,None,None]
                    label=f['labels'][:,:,:,:,None]
                    def pred(weight):
                        factor=np.exp(completion['beta']*z*weight)
                        return pb*factor/(1-pb+pb*factor)
                    pf=pred(.325);pc=pred(w)
                    change=pc-pf
                    mag=(change**2).mean((2,3,4))
                    align=(2*change*(label-pf)).mean((2,3,4))
                    decomposition[c,i,b]=np.stack((mag,align),axis=-1)
                    np.testing.assert_allclose(mag-align,loss[c,i,b,:,:,1]-loss[c,i,b,:,:,0],atol=2e-14)
                    bias[c,i,b]=np.stack([(p-label).mean((2,3,4)) for p in (pf,pc)],axis=-1)
                    if b==0 and i%2==0:
                        raw[c,i//2]=f['full_ms']
    assert np.isfinite(raw).all() and np.all(raw>0)
    cell_risk=loss.mean((2,3,4))
    delta=cell_risk[:,:,1]-cell_risk[:,:,0]
    observed=delta.mean(1)
    forecasts=np.array([freeze['cells'][k]['delta'] for k in CELLS])
    negative=(forecasts<0)[:,None]
    benchmark=freeze['mean_prior']
    routed=np.where(negative,cell_risk[:,:,1],cell_risk[:,:,0])
    abs_forecast=np.abs(observed-forecasts).mean()
    out=dict(cells=CELLS,number_catalogs=N*4,mean_brier=cell_risk.mean((0,1)),delta=pooled(delta),
             by_cell={k:{**cellstats(delta[j],np.sign(forecasts[j])),'loss':cell_risk[j].mean(0),
                         'forecast':forecasts[j],'squared_change_and_alignment':decomposition[j].mean((0,1,2,3))}
                      for j,k in enumerate(CELLS)},
             point_sign_count=int((np.sign(forecasts)==np.sign(observed)).sum()),
             total_correct_catalogs=int((np.sign(delta)==np.sign(forecasts[:,None])).sum()),
             point_mae=abs_forecast,baseline_mae=np.abs(observed-benchmark).mean(),
             zero_mae=np.abs(observed).mean(),rank=float(spearmanr(forecasts,observed).statistic),
             secondary_shape_diff=contrast(delta,[-.5,-.5,.5,.5]),
             secondary_interaction=contrast(delta,[1,-1,-1,1]),
             by_regime={g:pooled((loss[:,:,:,:,:,1]-loss[:,:,:,:,:,0])[:,:,:,:,ri].mean((2,3)))
                        for ri,g in enumerate(REGIMES)},bias=bias.mean((0,1,2,3,4)),
             cell_router_loss=routed.mean(),cell_router_vs_fixed=pooled(routed-cell_risk[:,:,0]),
             cell_router_vs_cont=pooled(routed-cell_risk[:,:,1]),timing=timing(raw))
    (ROOT/'stats39.json').write_text(json.dumps(out,indent=2,allow_nan=False,default=lambda x:x.tolist())+'\n')

if __name__=='__main__':
    main()
