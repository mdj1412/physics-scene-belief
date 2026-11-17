"""Reweight reference-to-impact mismatch after evaluating frozen policies."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parent
def scenario(losses, idx):
    contrast = losses[:, :, 0] - losses[:, :, idx]
    endpoints = np.stack((contrast[:, :2].mean(axis=1), contrast[:, 2]), axis=1)
    a, b = endpoints.mean(axis=0)
    crossing = -a / (b-a) if a*b < 0 else None
    return endpoints, dict(w0=a, w1=b, w_equal=contrast.mean(),
                           crossing=crossing, catalog_lower=int((contrast.mean(axis=1)<0).sum()),
                           catalog_effects=contrast.mean(axis=1).tolist())

def main():
    with np.load(ROOT/'clusters55.npz') as dat:
        methods=list(dat['methods'].astype(str))
        losses=dat['by_e']
        rate=dat['per_catalog_rate'].mean()
    assert losses.shape == (24,3,len(methods))
    assert methods[0]=='gate'
    ref,impact,random=[methods.index(k) for k in ('fixed4R','fixed3RI','rate_matched_random')]
    # The cost of randomizing is an average of risks after committing to an action.
    np.testing.assert_allclose(losses[:,:,random], (1-rate)*losses[:,:,ref]+rate*losses[:,:,impact],atol=1e-12,rtol=0)
    # Fixed fourth actions share potential outcomes even on prefixes assigned the other action.
    # Recover both conditional mean contrasts from the paired aggregate identities.
    mean = losses.mean(axis=0)
    routed_i = (mean[:,0] - mean[:,ref])/rate
    routed_r = (mean[:,impact] - mean[:,0])/(1-rate)
    covariance = rate*(1-rate)*(routed_i - routed_r)
    np.testing.assert_allclose(covariance, mean[:,0]-mean[:,random], atol=1e-12,rtol=0)
    points=np.linspace(0,1,101)
    fig,axes=plt.subplots(1,2,figsize=(9.2,3.7),layout='constrained')
    result={'impact_fraction':float(rate), 'e':[0.,.25,.5], 'impact_minus_ref_when_routed_impact':routed_i.tolist(),
            'impact_minus_ref_when_routed_reference':routed_r.tolist(), 'covariance':covariance.tolist()}
    for offset,(idx,label,color) in enumerate(((impact,'Fixed 3R+I','#95552c'),(random,'Rate-matched random','#127976'))):
        ends,result[methods[idx]]=scenario(losses,idx)
        axes[0].scatter(np.arange(1,25)+(offset-.5)*.17,np.array(result[methods[idx]]['catalog_effects'])*1e5,s=13,color=color,alpha=.8,label=label)
        avg=ends.mean(axis=0)
        axes[1].plot(points,((1-points)*avg[0]+points*avg[1])*1e5,label=label,color=color,lw=2)
    for ax in axes:
        ax.axhline(0,color='0.3',lw=.8)
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.18)
    # Uncertainty on the fixed-impact composition crossing, conditional on this family of shifts.
    with np.load(ROOT/'bootstrap55.npz') as boot:
        weights=boot['catalog_counts']/24
    ends,_=scenario(losses,impact)
    left,right=(weights@ends).T
    cross=(left*right<0)
    result['fixed_crossings_in_5000']=int(cross.sum())
    result['crossing_pointwise95']=np.quantile(left[cross]/(left[cross]-right[cross]),[.025,.975]).tolist() if cross.any() else []
    axes[0].set(xlabel='Held-out catalog',ylabel='Gate minus comparator Brier (×10⁻⁵)',xticks=[1,6,12,18,24],title='A   Individual catalogs')
    axes[1].axvline(1/3,color='0.5',ls=':',lw=1.2)
    axes[1].set(xlim=(0,1),xlabel='Fraction of sessions with e = 0.5',title='B   Target mismatch composition')
    axes[0].legend(fontsize=8,frameon=False); fig.savefig(ROOT/'transfer55.png',dpi=180)
    (ROOT/'transfer55.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
