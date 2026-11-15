"""Post-outcome paired attribution across the two contact laws, using frozen predictions."""
from pathlib import Path
from itertools import permutations, product
import json
import numpy as np
from scipy.stats import t

ROOT=Path(__file__).resolve().parent
FAMILIES=('shelf','ridge')
CONTOURS=('base','hollow','bulge')
LAWS=('saturating','crossover')
FEATURES=('base_probability','signed_cue_and_weight','stick_label')
BETA=.8289760825845313

def values(arc):
    return {k:arc[k] for k in ('risk','p_base','z','weights','labels')}

def risk_difference(base, cue, labels):
    pb=base['p_base'][:,None]
    z=cue['z'][:,None,:,None,None]
    def transform(w):
        mult=np.exp(BETA*z*w)
        return pb*mult/(1-pb+pb*mult)
    fixed,opt=transform(.325),transform(cue['weights'][:,:,:,None,None])
    return float(((opt-fixed)*(opt+fixed-2*labels[...,None])).mean())
def measure(pair):
    # Select input predictions and the observed endpoint under either law; no policy is retrained.
    table={i:risk_difference(pair[i[0]],pair[i[1]],pair[i[2]]['labels']) for i in product((0,1),repeat=3)}
    for li in range(2):
        archive=pair[li]['risk'].mean((0,1,3))
        np.testing.assert_allclose(table[(li,)*3],archive[1]-archive[0],atol=1e-13,rtol=0)
    change=np.zeros(3)
    for order in permutations(range(3)):
        current=[0,0,0]
        for slot in order:
            start=table[tuple(current)]
            current[slot]=1
            change[slot]+=(table[tuple(current)]-start)/6.
    np.testing.assert_allclose(change.sum(),table[1,1,1]-table[0,0,0],atol=1e-14)
    src,sink=pair
    reverse=risk_difference(src,src,src['labels'][::-1])
    return [table[1,1,1]-table[0,0,0],*change,
            reverse-table[0,0,0],table[0,0,1]-table[0,0,0],
            float((src['labels'][::-1]!=sink['labels']).mean()),float((src['labels']!=sink['labels']).mean())]
def summary(v):
    m=v.mean(0)
    gap=t.ppf(.975,len(v)-1)*v.std(0,ddof=1)/np.sqrt(len(v))
    return dict(mean=m.tolist(),ci=np.stack((m-gap,m+gap),axis=-1).tolist())
def main():
    info=json.loads((ROOT/'results/completion.json').read_text());assert info['blocks']==288
    all_values=[]
    for family,contour in product(FAMILIES,CONTOURS):
        records=[]
        for panel in range(12):
            blocks=[]
            for b in range(2):
                files=[ROOT/'results'/f'{family}_{contour}_{law}'/f'catalog_{panel:02d}'/f'block_{b:02d}.npz' for law in LAWS]
                with np.load(files[0]) as src,np.load(files[1]) as dest:
                    blocks.append(measure([values(src),values(dest)]))
            records.append(np.mean(blocks,axis=0))
        all_values.append(records)
    arr=np.array(all_values)
    result=dict(order=['crossover_minus_saturating',*FEATURES,'reversed_saturating_label_only','true_crossover_label_only',
                       'opposite_orientation_saturating_unequal_crossover_label','same_orientation_saturating_unequal_crossover_label'],
                pooled=summary(arr.mean(0)),cells={f'{s}_{c}':summary(x) for (s,c),x in zip(product(FAMILIES,CONTOURS),arr)})
    (ROOT/'law_decomposition40.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__': main()
