"""Descriptive cross-block repeatability across independently generated catalogs."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent


def group(pairs):
    left,right=pairs.T
    match=np.sign(left)==np.sign(right)
    scale=4*np.var(pairs.mean(1),ddof=1)
    # Cov estimates dependence of block effects across catalogs. It does not
    # separate catalog-specific starting-direction interactions from noise.
    cov=float(np.cov(left,right,ddof=1)[0,1])
    return dict(agreement_count=int(match.sum()),n=int(len(left)),
                mean_delta_by_start=pairs.mean(0).tolist(),
                signed_mean_block_difference=float((right-left).mean()),
                mean_absolute_block_difference=float(np.abs(right-left).mean()),
                between_start_corr=float(np.corrcoef(left,right)[0,1]),
                between_start_cov=cov,mean_variance_share_if_additive=(4*cov/scale if scale>0. else None))


def main():
    done=json.loads((ROOT/'results/completion.json').read_text())
    matrix=[]
    for cell in done['cells']:
        pairs=np.empty((12,2))
        for j in range(12):
            for b in range(2):
                path=ROOT/'results'/cell/f'catalog_{j:02d}'/f'block_{b:02d}.npz'
                with np.load(path) as f:
                    r=f['risk']
                    pairs[j,b]=(r[:,:,1]-r[:,:,0]).mean()
        matrix.append(pairs)
    x=np.stack(matrix)
    rng=np.random.default_rng(398823)
    idx=rng.integers(12,size=(16000,4,12))
    boot=x[np.arange(4)[None,:,None],idx]
    agree=np.sign(boot[:,:,:,0])==np.sign(boot[:,:,:,1])
    cell=[group(p) for p in x]
    for c,entry in enumerate(cell):
        entry['sign_agreement_bootstrap95']=np.quantile(agree[:,c].mean(1),[.025,.975]).tolist()
    mean=x.mean(1,keepdims=True)
    centered=(x-mean).reshape(-1,2)
    out=dict(cells=done['cells'],per_cell=cell,block_deltas=x.tolist(),
             overall_agreement=float((np.sign(x[:,:,0])==np.sign(x[:,:,1])).mean()),
             overall_agreement_boot95=np.quantile(agree.mean((1,2)),[.025,.975]).tolist(),
             cell_demeaned_corr=float(np.corrcoef(centered.T)[0,1]))
    (ROOT/'repeatability39.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
