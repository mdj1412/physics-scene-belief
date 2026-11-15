"""Crossed conditional gain variance and repeatability for v31; CPU analysis only."""
from __future__ import annotations

import hashlib
import itertools
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
LAWS = ("coulomb", "split_static", "speed_weak")
STAGES = ("2to4", "4to8")
SCHEMES = ("same_future_new_batch", "new_future_same_batch", "both_new_1x1",
           "both_new_1x2", "both_new_2x1", "both_new_2x2")
COMPONENTS = ("future", "batch", "interaction")
GRIDS = ((1,1), (2,1), (1,2), (2,2), (4,1), (1,4), (4,4))
N_BOOT = 10000
SEED = 317351


def variance_components(gains):
    """Unbiased crossed random-effects components, separately for every prefix."""
    futures, batches = gains.shape[-2:]
    m = gains.mean(axis=(-2, -1), keepdims=True)
    rows = gains.mean(axis=-1, keepdims=True)
    cols = gains.mean(axis=-2, keepdims=True)
    residual = gains - rows - cols + m
    ms_i = (residual**2).sum(axis=(-2, -1)) / ((futures-1)*(batches-1))
    ms_f = batches * rows[..., 0].var(axis=-1, ddof=1)
    ms_b = futures * cols[..., 0, :].var(axis=-1, ddof=1)
    return np.stack(((ms_f-ms_i)/batches, (ms_b-ms_i)/futures, ms_i), axis=-1)


@lru_cache(maxsize=None)
def replicate_design(nf, nb, scheme):
    mf, mb = {"both_new_1x2": (1,2), "both_new_2x1": (2,1),
              "both_new_2x2": (2,2)}.get(scheme, (1,1))
    grids = list(itertools.product(itertools.combinations(range(nf), mf),
                                   itertools.combinations(range(nb), mb)))
    pairs = []
    for i,(fi,bi) in enumerate(grids):
        for j,(fj,bj) in enumerate(grids):
            same_f = scheme == "same_future_new_batch"
            same_b = scheme == "new_future_same_batch"
            if ((fi == fj) if same_f else set(fi).isdisjoint(fj)) and (
                    (bi == bj) if same_b else set(bi).isdisjoint(bj)):
                pairs.append((i,j))
    indices = np.array(pairs, dtype=int)
    return grids, indices[:, 0], indices[:, 1]


def ranking_and_repeat_noise(g, scheme):
    """Enumerate legal repetitions *independently* for both prefixes in each pair."""
    e, nf, nb = g.shape
    grids, left, right = replicate_design(nf, nb, scheme)
    means = np.stack([g[:, fi, :][:, :, bi].mean((1,2)) for fi,bi in grids],axis=1)
    first,second = means[:,left], means[:,right]
    def concordance(i,j):
        sign1 = np.sign(first[i,:,None] - first[j,None,:])
        sign2 = np.sign(second[i,:,None] - second[j,None,:])
        return .5 + .5 * np.mean(sign1*sign2)
    pairs = itertools.combinations(range(e),2)
    concordance = np.mean([concordance(i,j) for i,j in pairs])
    difference = first - second
    return [concordance, np.mean(np.abs(difference)),
            np.mean(np.abs(difference)>.003), np.mean(difference**2)/2]


def block_metrics(g):
    assert g.shape == (2,8,4,4) and np.isfinite(g).all()
    c_env = variance_components(g)
    c = c_env.mean(axis=1)
    r = np.array([[ranking_and_repeat_noise(stage, scheme) for scheme in SCHEMES] for stage in g])
    # Direct squared differences of distinct independent repeats on the finite grid.
    expected = np.stack([c[:,1]+c[:,2], c[:,0]+c[:,2], c.sum(1),
                         c[:,0]+c[:,1]/2+c[:,2]/2, c[:,0]/2+c[:,1]+c[:,2]/2,
                         c[:,0]/2+c[:,1]/2+c[:,2]/4],axis=1)
    np.testing.assert_allclose(r[:,:,3],expected,rtol=2.e-9,atol=2.e-14)
    grid_var = np.stack([c[:,0]/mf+c[:,1]/mb+c[:,2]/(mf*mb) for mf,mb in GRIDS],axis=1)
    env_mean = g.mean((-2,-1))
    # Fixed eight environments within a block: finite panel, not a sample of new laws.
    observed_between = env_mean.var(axis=1,ddof=1)
    corrected_between = observed_between - grid_var[:,-1]
    return dict(components=c, ranks=r[:,:,0], abs_repeat_difference=r[:,:,1],
                practical_discrepancy=r[:,:,2], half_squared_repeat_difference=r[:,:,3],
                grid_var=grid_var, gain=g.mean((1,2,3)), between_observed=observed_between,
                between_corrected=corrected_between)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_result(folder):
    meta = json.loads((folder / "metrics.json").read_text())
    for artifact in ('metrics','choices','prefix'):
        assert sha(folder/(artifact+'.npz')) == meta[artifact+'_sha256']
    with np.load(folder/'prefix.npz') as prefix:
        for field in ('pool_seeds','future_seeds','batch_seeds','bank_pair'):
            np.testing.assert_array_equal(prefix[field], meta[field])
        assert prefix['paid_ids'].shape == (8,4) and prefix['paid_y'].shape == (8,4,2)
        assert all(len(set(row)) == 4 for row in prefix['paid_ids'])
        assert prefix['factual_pool'].shape==(8,1024,11)
        assert prefix['additional_2'].shape==(8,4,2,13) and prefix['additional_4'].shape==(8,4,4,13)
    with np.load(folder/'choices.npz') as selections:
        gaps=selections['gap']
        assert gaps.shape==(2,8,5,4,128)
        a=np.ones_like(gaps,dtype=np.int8)
        order=np.argsort(-gaps,axis=-1,kind='stable')
        np.put_along_axis(a,order[...,:13],3,axis=-1)
        assert ((a==3).sum(-1)==13).all()
        np.testing.assert_array_equal(a[:,:,0],selections['action_before'])
        np.testing.assert_array_equal(a[:,:,1:],selections['action_after'])
    with np.load(folder / 'metrics.npz') as z:
        g = z["gain"]
        before, after = z["before"], z["after"]
        labels=z['labels']
        prob=z['probabilities']
        assert labels.shape==(8,4,128) and prob.shape==(8,4,128,2,32)
        assert np.isfinite(prob).all() and prob.min()>=-1.e-10 and prob.max()<=1.+1.e-10
        cost=((prob-labels[:,:,:,None,None])**2).mean(-1)
        np.testing.assert_allclose(cost,z['cost'],atol=1.e-12,rtol=0)
        risks=np.where(a==3,cost[None,:,None,:,:,1],cost[None,:,None,:,:,0]).mean(-1)
        np.testing.assert_allclose(risks[:,:,0],before,atol=1.e-12,rtol=0)
        np.testing.assert_allclose(risks[:,:,1:],after,atol=1.e-12,rtol=0)
        if 'gain_probe_only' in z:
            np.testing.assert_allclose(g,z['gain_probe_only']+z['gain_factual_only'],rtol=0,atol=1.e-12)
        assert before.shape==(2,8,4) and after.shape==(2,8,4,4)
        assert np.all((before>=0)&(before<=1)) and np.all((after>=0)&(after<=1))
        np.testing.assert_allclose(g,before[:,:,None,:]-after,rtol=0,atol=1.e-12)
    return g


def summary(values, boots):
    return dict(estimate=values.tolist(), pointwise95=np.moveaxis(
        np.quantile(boots,[.025,.975],axis=0),0,-1).tolist())


def joint_intervals(estimated, resampled):
    std = resampled.std(axis=0, ddof=1)
    centered = (resampled-estimated) / np.where(std==0,1,std)
    cutoff = np.quantile(np.abs(centered).reshape(len(centered),-1).max(axis=1),.95)
    return np.stack((estimated-cutoff*std,estimated+cutoff*std),axis=-1), cutoff


def analyze(arrays):
    # law -> independent block -> statistic shape; resample all endpoints and stages together.
    rng = np.random.default_rng(SEED)
    ids = rng.integers(6,size=(N_BOOT,len(LAWS),6))
    data = {k:np.asarray([[item[k] for item in law] for law in arrays])
            for k in arrays[0][0]}
    estimates, samples = {}, {}
    out = dict(laws=LAWS,stages=STAGES,components_order=COMPONENTS,
               schemes=SCHEMES,grids=GRIDS,bootstrap_blocks_per_law=6,
               bootstrap_draws=N_BOOT,bootstrap_seed=SEED, per_law={},pooled={})
    for k,a in data.items():
        law_value=a.mean(1)
        law_sample=a[np.arange(3)[None,:,None],ids].mean(2)
        estimates[k]=law_value.mean(0)
        samples[k]=law_sample.mean(1)
        out["per_law"][k]=summary(law_value,law_sample)
        out["pooled"][k]=summary(estimates[k], samples[k])
    c = estimates['components']
    bc = samples['components']
    primary_values = np.stack([c[:,1]-c[:,0],estimates['ranks'][:,2]-.5],axis=1)
    primary_boots = np.stack([bc[:,:,1]-bc[:,:,0],samples['ranks'][:,:,2]-.5],axis=2)
    bands,critical = joint_intervals(primary_values,primary_boots)
    out['primary']=dict(order=['batch_minus_future_variance','independent_concordance_minus_half'],
                        estimate=primary_values.tolist(), simultaneous95=bands.tolist(),critical=float(critical))
    out['pooled']['component_shares']=summary(c/c.sum(-1,keepdims=True),bc/bc.sum(-1,keepdims=True))
    out['pooled']['added_batch_minus_added_future_variance']=summary(
        estimates['grid_var'][:,2]-estimates['grid_var'][:,1],
        samples['grid_var'][:,:,2]-samples['grid_var'][:,:,1])
    out['block_rows']={k:v.tolist() for k,v in data.items()}
    out['interpretation']='Brier gain is before minus after. Variances in squared Brier units are unrestricted MOM estimates. No target-law or between-panel extrapolation. Pointwise intervals are unconditional over sampled block/prefix draws, conditional on three law types and eight fixed coupling values per law.'
    return out


def main():
    arrays=[]
    for law in LAWS:
        arr=[]
        for block in range(6):
            arr.append(block_metrics(audit_result(RESULTS/law/f"block_{block:03d}")))
        arrays.append(arr)
    output=analyze(arrays)
    dest=RESULTS/'crossed_stats31.json'
    tmp=dest.with_suffix('.tmp')
    tmp.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    tmp.replace(dest)
    print('Primary',output['primary'],flush=True)
    print('Components',output['pooled']['components'],flush=True)
    print('Rank reproducibility',output['pooled']['ranks'],flush=True)


if __name__=='__main__':
    main()
