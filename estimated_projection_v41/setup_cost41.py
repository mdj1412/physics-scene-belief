"""Summarize label-free fit and candidate-template construction costs in matched timing sweeps."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent


def main():
    assert (ROOT/'results/completion.json').is_file()
    rows=[]
    for fn in sorted((ROOT/'results').glob('*/catalog_*/block_*.npz')):
        with np.load(fn) as f:
            if f['fit_ms'].size:
                rows.append((int(fn.parent.name[-2:]),f['fit_ms'],f['setup_ms']))
    assert [r[0] for r in sorted(rows)]==list(range(12))
    fit=np.stack([r[1] for r in rows])
    setup=np.stack([r[2] for r in rows])
    def summarize(x):
        return dict(p50=np.quantile(x,.5,axis=(0,1)).tolist(),p90=np.quantile(x,.9,axis=(0,1)).tolist())
    est=fit[:,:,2,:].ravel()
    result=dict(methods=['fixed','optimizer','selector'],decoders=['original','oracle','estimated'],
                passive_pairs_per_fit=32,estimated_direction_fit_ms_p50=float(np.median(est)),
                estimated_direction_fit_ms_p90=float(np.quantile(est,.9)),
                estimated_direction_fit_timings=len(est),fit_ms=summarize(fit),setup_ms=summarize(setup),
                template_after_fit_ms=summarize(setup-fit))
    (ROOT/'setup_cost41.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
