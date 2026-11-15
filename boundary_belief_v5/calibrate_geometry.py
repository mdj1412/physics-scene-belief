import json,argparse
from pathlib import Path
import numpy as np
from geometry import render,extract,calibration_np

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--quality',choices=['high','low','pixel_only'],required=True);args=p.parse_args()
    out=Path('results/calibration');out.mkdir(parents=True,exist_ok=True)
    quality_index=['high','low','pixel_only'].index(args.quality)
    rng=np.random.default_rng(10801+quality_index);result={};n=40;grid=np.linspace(0,.006,13)
    for family in ['corner','gate','deflection']:
        means=[];residuals=[];counts=[]
        for b in grid:
            rows=[]
            for rep in range(n):
                g=rng.uniform(-.008,.008);d=rng.uniform(.022,.034)
                image=render(g,d,b,family,args.quality,rng);f=extract(image,family,args.quality)
                rows.append(f-np.array([g*1e3,d*1e3,0]))
            rows=np.array(rows);means.append(rows.mean(0));residuals.extend(rows-rows.mean(0));counts.append(len(rows))
        means=np.array(means)
        # Bounding box does not encode b. Do not turn finite calibration noise
        # in g/d sample means into an artificial b-dependent signal.
        pooled=means[:,:2].mean(0);residuals=np.array(residuals)
        for j in range(len(grid)):residuals[j*n:(j+1)*n,:2]+=means[j,:2]-pooled
        means[:,:2]=pooled
        cov=np.cov(residuals.T)+np.diag([.05**2,.05**2,.10**2])
        result[family]={args.quality:dict(b_grid=grid.tolist(),mean=np.array(means).tolist(),cov=cov.tolist(),n_per_bin=counts)}
        print('calibrated',family,args.quality,flush=True)
    (out/f'{args.quality}.json').write_text(json.dumps(result,indent=2))
    # Independent holdout: b posterior from curvature descriptor only.
    rng=np.random.default_rng(20801+quality_index);records=[];dense=np.linspace(0,.006,1201)
    for family in result:
        bias,cov=calibration_np(result,family,args.quality,dense)
        for rep in range(80):
            g=rng.uniform(-.008,.008);d=rng.uniform(.022,.034);b=rng.uniform(0,.006)
            im=render(g,d,b,family,args.quality,rng);f,detail=extract(im,family,args.quality,True)
            ll=-.5*(f[2]-bias[:,2])**2/cov[2,2];w=np.exp(ll-ll.max());w/=w.sum();cdf=w.cumsum()
            low=dense[np.searchsorted(cdf,.05)];high=dense[np.searchsorted(cdf,.95)]
            records.append(dict(family=family,b=b,estimate=float(w@dense),lo=low,hi=high,
                feature=f.tolist(),truth_gd=[g,d],area_deficit=detail['deficits']))
            if rep==0:im.save(out/f'{family}_{args.quality}.png')
    (out/f'holdout_{args.quality}.json').write_text(json.dumps(records,indent=2))
    print('holdout',args.quality,'MAE mm',np.mean([abs(r['estimate']-r['b']) for r in records])*1e3,
          'coverage',np.mean([r['lo']<=r['b']<=r['hi'] for r in records]),flush=True)
