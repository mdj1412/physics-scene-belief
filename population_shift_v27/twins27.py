"""Pair two different population laws with exactly the same factual rows, on allocated CPU."""
import os
import sys
import json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'proxy_shift_v26'))
from physics26 import change_proxy, variants
from physics25 import parameters, observations, labels, sigmas, torch, t

def main():
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('BELIEF_DEVICE')=='cpu'
    torch.set_num_threads(1)
    rng=np.random.default_rng(27746621)
    n=16384
    records=[]
    for ac,af in ((-.65,.85),(.35,.65),(-.2,.35),(0.,1.)):
        unit=rng.random((n,9)); flip=rng.random((n,2))
        base=parameters(t(unit))
        other=unit.copy();other[:,1]=(other[:,1]+.5)%1.
        base2=parameters(t(other))
        # Flipping both spread bit and relation bit preserves assigned restitution center.
        opposite=flip.copy();opposite[:,1]=1.-flip[:,1]
        first=change_proxy(base,torch.from_numpy(variants(flip,ac,af)))
        second=change_proxy(base2,torch.from_numpy(variants(opposite,ac,-af)))
        orig,y=observations(first),observations(second)
        eps=t(rng.normal(size=(n,11)))*sigmas()[0]
        x1=orig[0]+eps;x2=y[0]+eps
        lab=labels(first)!=labels(second)
        diff=float((x1-x2).abs().max())
        assert diff<1e-12 and torch.equal(first[:,3],second[:,3])
        records.append(dict(alpha_c=ac,opposite_alpha_f=af,n=n,max_factual_error=diff,
            max_restitution_error=float((first[:,5]-second[:,5]).abs().max()),
            max_query_view_error=float((orig[1]-y[1]).abs().max()),
            mean_probe_l2=float((orig[3]-y[3]).norm(dim=1).mean()),query_label_flip=float(lab.double().mean())))
    p=Path('results/twins.json');p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(records,indent=2))
if __name__=='__main__':main()
