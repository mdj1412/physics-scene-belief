"""Source-only crossed gain precision, conditioned on shared paid prefix, simulator and K256 bank."""
from pathlib import Path
import json
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results_train'


def tied_rank(x):
    # Mean rank for exact ties; stable under strictly monotonic transforms.
    return ((x[:,None]>x).sum(1)+.5*((x[:,None]==x).sum(1)-1)).astype(float)


def rho(left,right):
    left=tied_rank(left);right=tied_rank(right)
    a=left-left.mean();b=right-right.mean();denom=np.linalg.norm(a)*np.linalg.norm(b)
    return (a@b/denom) if denom else np.nan


def sufficient(g):
    # g[E,F=2,B=2]; random future and batch factors conditioned on prefix + bank.
    row=g.mean(2);col=g.mean(1);mean=g.mean((1,2))
    ms_future=2*((row-mean[:,None])**2).sum(1)
    ms_batch=2*((col-mean[:,None])**2).sum(1)
    ms_resid=((g-row[:,:,None]-col[:,None,:]+mean[:,None,None])**2).sum((1,2))
    v_future=.5*(ms_future-ms_resid);v_batch=.5*(ms_batch-ms_resid)
    v_single=v_future+v_batch+ms_resid
    v_mean=.5*(v_future+v_batch)+.25*ms_resid
    # Independent diagonal pairs share neither calibration future nor deployment batch.
    expected_sq=.25*((g[:,0,0]-g[:,1,1])**2+(g[:,0,1]-g[:,1,0])**2)
    assert np.allclose(v_single,expected_sq,rtol=1e-9,atol=1e-17)
    ranks=[rho(g[:,0,0],g[:,1,1]),rho(g[:,0,1],g[:,1,0])] if len(g)==8 else [np.nan,np.nan]
    return np.array([len(g), v_future.sum(),v_batch.sum(),ms_resid.sum(),v_single.sum(),v_mean.sum(),
                     np.nansum(ranks),np.isfinite(ranks).sum()])


def estimates(s):
    with np.errstate(divide='ignore',invalid='ignore'):
        v=s[...,1:6]/s[...,:1]
        ans=np.concatenate([v, np.sqrt(v[...,4:5]/v[...,3:4]),s[...,6:7]/s[...,7:8]],axis=-1)
    return ans


def main():
    cols=['variance_future','variance_batch','variance_interaction','single_cell_noise_variance',
          'mean_2x2_noise_variance','conditional_noise_rmse_ratio_2x2_over_1x1','within_block_diagonal_spearman']
    records=[]
    for md in sorted((OUT/'repeated').glob('block_*/meta.json')):
        with np.load(md.parent/'data.npz') as z:
            records.append([sufficient(g) for g in z['gain']])
    data=np.array(records)
    completion=json.loads((OUT/'labels_complete.json').read_text())
    assert len(records)==completion['arms']['repeated']['blocks']
    rng=np.random.default_rng(321512039)
    draw=rng.integers(len(records),size=(10000,len(records)))
    estimate=estimates(data.sum(0));boot=estimates(data[draw].sum(1))
    interval=np.nanquantile(boot,[.025,.975],axis=0)
    counts=completion['arms']
    cost_ratio=(counts['repeated']['elapsed_s']/counts['repeated']['n'])/(counts['broad']['elapsed_s']/counts['broad']['n'])
    out={}
    for s,stage in enumerate(['2_to_4','4_to_8']):
        out[stage]={col:{'estimate':float(estimate[s,i]),'percentile_95':interval[:,s,i].tolist()} for i,col in enumerate(cols)}
        out[stage]['complete_ranking_pairs']=int(data[:,s,-1].sum())
        # Noise-only variance contribution of an unweighted mean over independent prefixes
        # at matched time. Irreducible prefix heterogeneity and regression design omitted.
        ratio=cost_ratio*estimate[s,5]**2
        uncertainty=np.nanquantile(cost_ratio*boot[:,s,5]**2,[.025,.975])
        out[stage]['noise_only_variance_ratio_unweighted_mean_at_equal_time']=dict(estimate=float(ratio),percentile_95=uncertainty.tolist())
    res={'measured_seconds_per_prefix_ratio_repeated_over_broad':cost_ratio,'n_blocks':len(records),'n_prefixes':int(data[:,0,0].sum()),'output':out,
         'estimand':'Finite simulator and fixed training bank conditioned on paid prefix. Signed untruncated random effects via two-way ANOVA, independently sampled pool futures and deployment batches. Paired diagonal gains differ in both. Spearman is average over two pairs per complete eight-environment block, tied ranks averaged. Marginal 95% resampling bank blocks, descriptive and separate from held-out policy tests.'}
    (OUT/'label_precision.json').write_text(json.dumps(res,indent=2,allow_nan=False)+'\n')
    print(json.dumps(res))
if __name__=='__main__':
    main()
