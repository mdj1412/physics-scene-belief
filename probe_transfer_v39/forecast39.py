"""Preregister a cell forecast and an observation-only forecast from v38 alone."""
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'probe_value_v38'
CELLS = ('shoulder_weakening','shoulder_strengthening','bead_weakening','bead_strengthening')
TARGET_SHAPES = ('shelf','ridge')
SURFACES = ('weakening','strengthening')
BETA = .8289760825845313
SHRINK = .325
PENALTY = .01


def geometry():
    u = np.array([.023,.025])/.025
    return {'shoulder':.072*u**2/(1+u**2),
            'bead':-.078*u*np.exp(-.5*(u/.80)**2),
            'shelf':.043*(1-np.cos(np.pi*u/2)),
            'ridge':-.153*u/(1+u**2)**2}


def design_matrix(z,w):
    a=z*(w-SHRINK)
    return np.stack([a,a*z,a**2],axis=-1)


def fit(x,y):
    # No intercept: equal actions have exactly zero incremental Brier loss.
    x = x.reshape(-1,3)
    scale=np.sqrt(np.mean(x*x,axis=0))
    normalized=x/scale
    reg=len(x)*PENALTY*np.eye(x.shape[1])
    return np.linalg.solve(normalized.T@normalized+reg, normalized.T@y.ravel())/scale


def historical_arrays():
    os.environ['BELIEF_DEVICE']='cpu'
    sys.path.insert(0,str(OLD))
    from continuous38 import source_optimum
    from gate37 import load_artifacts
    art = load_artifacts()[1.10]
    diffs=np.empty((4,8,2,3,8))
    x=np.empty((*diffs.shape,3))
    for c,cell in enumerate(CELLS):
        for b in range(8):
            path=OLD/'results'/cell/f'block_{b:03d}'
            for s,sign in enumerate(('plus','minus')):
                with np.load(path/f'shared_{sign}.npz') as a:
                    p=a['p_base']
                    far,y3,d=a['far_x'],a['first3_y'],a['dirs']
                    for r,reg in enumerate(('strong','weak','null')):
                        with np.load(path/f'{reg}_{sign}.npz') as arm:
                            z,w=source_optimum(art,far,y3,d,arm['fourth_y'])
                            label=arm['labels'][:,:,None]
                            risks=[]
                            for weight in (SHRINK,w):
                                odds=np.exp(BETA*z*weight)[:,None,None]
                                q=p*odds/(1-p+p*odds)
                                risks.append(((q-label)**2).mean((1,2)))
                            np.testing.assert_allclose(risks[0],arm['risk'][1,0],rtol=0,atol=2e-12)
                            diffs[c,b,s,r]=risks[1]-risks[0]
                            x[c,b,s,r]=design_matrix(z,w)
    check=json.loads((OLD/'continuous38.json').read_text())['diff'][0]
    np.testing.assert_allclose(diffs.mean(),check,atol=5e-12,rtol=0)
    return x,diffs


def route_results(y,pred):
    take=pred<0
    return {'optimizer_fraction':take.mean(), 'delta_route_minus_fixed':(y*take).mean(),
            'delta_route_minus_optimizer':-(y*(~take)).mean(),
            'optimizer_minus_fixed_forecast_win':y[take].mean() if take.any() else None,
            'optimizer_minus_fixed_forecast_loss':y[~take].mean() if (~take).any() else None,
            'rmse':np.sqrt(np.mean((y-pred)**2))}


def forecast():
    x,y=historical_arrays()
    pts=y.mean(axis=(1,2,3,4))
    block=y.mean(axis=(2,3,4))
    rng=np.random.default_rng(39003)
    # Each v38 cell has one mixture of fixed alpha-specific catalogs. Preserve starting sign.
    ids=(2*rng.integers(4,size=(16000,4,4,2))+np.arange(2)).reshape(16000,4,8)
    boot=block[np.arange(4)[None,:,None],ids].mean(2)
    ci=np.quantile(boot,[.025,.975],axis=0).T
    slope=geometry()
    predictions={}
    for new in TARGET_SHAPES:
        analog=min(('shoulder','bead'),key=lambda k:np.linalg.norm(slope[new]-slope[k]))
        for law in SURFACES:
            ref=f'{analog}_{law}'
            i=CELLS.index(ref)
            # Leave one *block* out only tests within-catalog repeatability.
            loo=(block[i].sum()-block[i])/7
            predictions[f'{new}_{law}']=dict(source_cell=ref,delta=pts[i],
                sign='win' if pts[i]<0 else 'loss',historical_block_interval=ci[i],
                old_blocks_negative=int((block[i]<0).sum()),old_loo_stable=bool(np.all(np.sign(loo)==np.sign(pts[i]))),
                old_sign_separated=bool(ci[i,0]>0 or ci[i,1]<0))
    coef=fit(x,y)
    cv=np.empty_like(y)
    for i in range(len(CELLS)):
        keep=np.arange(len(CELLS))!=i
        cv[i]=x[i]@fit(x[keep],y[keep])
    return dict(method='v38 paired Brier nearest local-slope geometry, same law; not a physics consequence',
                quantities='optimizer minus fixed; negative means optimizer has smaller Brier',
                beta=BETA,shrink=SHRINK,speed=1.10,
                v38_catalogs_per_cell_per_alpha=1,units='per-environment mean Brier; equally weighted regimes and directions',
                geometry_slopes_at_23_25_mm=slope,cells=predictions,
                mean_prior=pts.mean(),min_useful_improvement=.003,
                observable=dict(formula='a=z*(w-.325); x=[a, a*z, a*a]; predicted_delta=x@coefficients',
                                coefficients=coef, standardized_ridge_lambda=PENALTY, weight_if_negative='frozen optimizer',
                                weight_if_nonnegative='fixed .325',v38_leave_one_cell_out=route_results(y,cv),
                                v38_leave_one_cell_out_by_cell={c:route_results(y[i],cv[i]) for i,c in enumerate(CELLS)}),
                source_manifest_sha256=hashlib.sha256((OLD/'results'/'run.json').read_bytes()).hexdigest(),
                source_policy_sha256=hashlib.sha256((OLD/'continuous38.py').read_bytes()).hexdigest())


def main():
    result=forecast()
    (ROOT/'forecast39.json').write_text(json.dumps(result,indent=2,allow_nan=False,
                              default=lambda v:v.tolist())+'\n')
    print('cell signed Brier predictions',[(k,v['delta']) for k,v in result['cells'].items()])
    print('observed-feature slopes',result['observable']['coefficients'])


if __name__=='__main__':
    main()
