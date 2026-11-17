"""Exploratory subgroup and deployment prevalence sensitivity; labels already opened."""
import itertools
import json
import numpy as np
from stats43 import ROOT, NONZERO, OBJECTS, SURFACES, LAWS, CONTOURS, HEIGHTS, interval


def main():
    deltas = np.empty((12,2,2,2,3,2,9))
    for panel, i,j,k,v,h in itertools.product(range(12),range(2),range(2),range(2),range(3),range(2)):
        filename='_'.join((OBJECTS[i],SURFACES[j],LAWS[k],CONTOURS[v],HEIGHTS[h]))+'.npz'
        with np.load(ROOT/'results'/f'panel_{panel:02d}'/filename) as a:
            risks=a['risk'][:,:,0].mean(axis=-1)
            deltas[panel,i,j,k,v,h]=risks[:,2]-risks[:,0]
    def two_strata(x):
        return np.stack((x[...,4],x[...,NONZERO].mean(axis=-1)),axis=-1)
    shape=two_strata(deltas.mean(axis=(2,3,4,5)))
    variant=two_strata(deltas.mean(axis=(1,2,3,5)))
    primary=json.loads((ROOT/'pred_stats43.json').read_text())
    risk=np.asarray(primary['panel_risk'])
    d=risk[:,:,1,0]-risk[:,:,0,0]
    nom=d[:,4]
    nz=d[:,NONZERO].mean(axis=-1)
    q=nom.mean()/(nom.mean()-nz.mean())
    idx=np.random.default_rng(430119).integers(0,len(nom),(20000,len(nom)))
    n,s=nom[idx].mean(1),nz[idx].mean(1)
    prop=n/(n-s)
    # Interval is conditional on population mean curves having an interior crossing.
    valid=(n>0)&(s<0)
    out=dict(shape_nominal_shifted_sim95=interval(shape,4).tolist(),variant_nominal_shifted_sim95=interval(variant,6).tolist(),
             weight_shifted_break_even=q, interior_bootstrap_fraction=valid.mean(),
             conditional_break_even_ci=np.quantile(prop[valid],[.025,.975]).tolist())
    (ROOT/'split43.json').write_text(json.dumps(out,indent=2)+'\n')


if __name__=='__main__':
    main()
