"""Cross-fit using one random unobserved-offset outcome per donor object, versus full replay."""
import json
import numpy as np
from selector_audit42 import ROOT,sums,take,splits,interval


def replay(path):
    m=json.loads(path.with_suffix('.json').read_text()); masks=splits(m['deployment_seeds'])
    index=np.array([np.random.default_rng(np.random.SeedSequence([seed,420983])).integers(8,size=(4,128))
                    for seed in m['deployment_seeds']]).transpose(1,0,2)
    with np.load(path) as a:
        y=a['labels'][a['offsets_mm']!=0]; pb=a['p_base'][2]; z=a['z'];w=a['weights']
    rows=[]
    for sign in range(2):
        for reg in range(3):
            q=pb[sign]; truth=y[:,sign,reg].astype(float)
            tgt=sums(truth,q,w[sign,reg],z[sign]).mean(0,keepdims=True)
            for mask,ids in zip(masks,index):
                sampled=np.take_along_axis(truth,ids[None],axis=0)
                full=sums(sampled,q,w[sign,reg],z[sign])
                left=sums(sampled,q,w[sign,reg],z[sign],mask)
                right=full-left
                target_left=sums(truth,q,w[sign,reg],z[sign],mask).mean(0,keepdims=True)
                target_right=tgt-target_left
                risk=(take(target_right,left.argmin(-1))+take(target_left,right.argmin(-1)))/128
                rows.append(risk.mean())
    return np.mean(rows)


def main():
    meta=json.loads((ROOT/'results/completion.json').read_text())
    arr=np.array([[replay(ROOT/'results'/c/f'catalog_{p:02d}.npz') for c in meta['cells']] for p in range(12)])
    both=interval(arr.mean(1))
    (ROOT/'single_label42.json').write_text(json.dumps(dict(half_label_budget='one uniformly random hidden-offset collision per donor object',delta_vs_zero=both,delta_by_panel=arr.mean(1).tolist()),indent=2)+'\n')
    print(both,flush=True)
if __name__=='__main__':main()
