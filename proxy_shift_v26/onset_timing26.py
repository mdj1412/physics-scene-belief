"""Post hoc latency at the matched-prior diagonal onset; held-out actions remain sealed."""
import os
from pathlib import Path
import numpy as np
from physics26 import Engine, torch
from run26 import BANKS, frozen, timed
from stats26 import paired_time_quantiles
from common import read, require, save, sha, write

def main():
    os.chdir(Path(__file__).resolve().parent)
    require()
    torch.set_num_threads(2)
    root = Path('results')
    matched = root/'matched'
    onset = read(matched/'summary.json')['crossing']['diagonal']['flag']['relative']['first']
    assert onset is not None and torch.cuda.is_available()
    for name, expected in read(matched/'choices_freeze.json')['files'].items():
        assert sha(name) == expected
    with np.load(matched/'cohort.npz') as cohort:
        ids, cells = cohort['ids'], cohort['cell']
        variants = cohort['variant'][:,onset]
        condition = cohort['conditions'][onset]
    with np.load(root/'test/public.npz') as public:
        x = public['x'][ids,variants]
    with np.load(root/'test/private.npz') as measurements:
        q = measurements['obs1'][ids,variants,0]
        p = measurements['obs3'][ids,variants,0]
    time_ids = np.sort(np.concatenate([np.flatnonzero(cells==cell)[:12] for cell in range(4)]))
    predictor, cut = frozen()
    measurements = []
    for bank,pair in enumerate(BANKS[:3]):
        engine = Engine(pair,*condition)
        with np.load(matched/f'choices_{bank}.npz') as f:
            acts = f['acts'][:,onset]
        measurements.append(timed(engine,predictor,cut,x,q,p,time_ids,acts,3,260881+bank))
    raw = np.stack(measurements)
    save(matched/'onset_latency_raw.npz',times=raw,local_ids=time_ids)
    p90 = np.quantile(raw.reshape(-1,2),.9,axis=0)
    draws = paired_time_quantiles(raw[:,:,:,None,:],cells[time_ids],3000)[:,0]
    ratio = np.quantile(draws[:,0]/draws[:,1],[.025,.975])
    write(matched/'onset_latency.json',dict(condition=condition.tolist(),n=int(np.prod(raw.shape[:-1])),
        p90_ms=p90.tolist(),ratio=float(p90[0]/p90[1]),pointwise_ci95=ratio.tolist(),
        design='post hoc timing at first detected diagonal onset; no new Brier comparison'))
if __name__=='__main__':main()
