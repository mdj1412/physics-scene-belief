"""Predict v40 without accessing any v40 samples. Replay historical v39 measurements only."""
from pathlib import Path
import os
os.environ['BELIEF_DEVICE']='cpu'
import sys
import json
import hashlib
from itertools import product
import numpy as np

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'probe_transfer_v39'
sys.path.insert(0,str(OLD))
sys.path.insert(0,str(ROOT.parent/'probe_value_v38'))
import physics39 as historical
import run38 as policy

SHAPES=('shelf','ridge')
CONTOURS=('base','hollow','bulge')
LAWS=('saturating','crossover')
OLD_LAWS=('weakening','strengthening')
BUMP=np.array([.0007111111111111111,.0007111111111111111,0.,0.,.00050625,.0009,.00050625])
BETA=.8289760825845313
def asymmetry(law,v):
    if law=='saturating':
        return .18*np.tanh(v/.62)**3
    if law=='crossover':
        return .20*np.tanh(v/.32)*np.tanh((abs(v)-.70)/.16)
    if law=='weakening':
        return .19*(-np.expm1(-v/.47))
    return .192*(2*v/.75)/(1+(v/.75)**2)
def retention(source,law):
    return float(np.clip(asymmetry(law,.425)/asymmetry(source,.425),-1.,1.))

def replay_block(path, shape, old_law, cid):
    """Full archived base result; paired finite intervention on a systematic subset of old objects and noise."""
    meta=json.loads(path.with_suffix('.json').read_text())
    with np.load(path) as f:
        alpha,z,w=f['alpha'],f['z'],f['weights']
        labels=f['labels']
        pb=f['p_base']
        actions=f['actions']
    array=np.zeros((3,2))
    # Full historical label-recombination baseline plus paired subset contour increment.
    # Interpolate risks (squared conditional-risk difference is affine in the event indicator).
    old_asym=asymmetry(old_law,.425)
    selection=np.arange(meta['block']*2,128,4)
    noises=[0,16]
    shifts=np.array([-1.,1.])
    for e,aa in enumerate(alpha):
        decoder=policy.Engine(meta['pair'],*aa).inner
        for s,sgn in enumerate((1,-1)):
            obs,old_y,_,sen=historical.deployment(meta['deployment_seeds'][e],aa,(shape,old_law,sgn,cid),'strong',repeats=32)
            np.testing.assert_array_equal(labels[s,0,e],old_y)
            selected=selection[actions[s,e,selection]==policy.Q]
            prior=decoder.fit(policy.t(obs[selected]))[:,None,:]
            observed=sen[policy.Q][selected][:,noises,:]
            ref=pb[s,e,selected][:,noises]
            pold=(decoder.update(prior,policy.Q,policy.t(observed))@decoder.label[:,0]).numpy()/np.sqrt(2.)
            np.testing.assert_allclose(ref,pold,rtol=0.,atol=2.e-10)
            shifted=[]
            for direction in shifts:
                prob=(decoder.update(prior,policy.Q,policy.t(observed+direction*BUMP))@decoder.label[:,0]).numpy()/np.sqrt(2.)
                shifted.append(prob)
            def difference(base,mean_label,weight):
                def adjust(v):
                    odds=np.exp(BETA*z[s,e]*v)
                    return base*odds/(1-base+base*odds)
                fixed,opt=adjust(.325),adjust(weight)
                return (opt-fixed)*(opt+fixed-2*mean_label[:,None])
            for li,law in enumerate(LAWS):
                stay=(1+retention(old_law,law))/2
                for ri in range(3):
                    ybar=stay*labels[s,ri,e]+(1-stay)*labels[1-s,ri,e]
                    qref=difference(pb[s,e],ybar,w[s,ri,e]).mean()
                    array[:,li]+=qref
                    subset=difference(ref,ybar[selected],w[s,ri,e])
                    for ci,q in enumerate(shifted):
                        increment=(difference(q,ybar[selected],w[s,ri,e])-subset).sum()/len(noises)/len(selection)
                        array[ci+1,li]+=increment
    return array/(8*2*3)
def main():
    policy.torch.set_num_threads(1)
    arr=np.empty((2,2,12,2,3,2))
    for si,shape in enumerate(SHAPES):
        for oi,source in enumerate(OLD_LAWS):
            for c in range(12):
                for b in range(2):
                    p=OLD/'results'/f'{shape}_{source}'/f'catalog_{c:02d}'/f'block_{b:02d}.npz'
                    arr[si,oi,c,b]=replay_block(p,shape,source,c)
    old=json.loads((OLD/'stats39.json').read_text())
    tangent={shape:np.mean([old['by_cell'][f'{shape}_{law}']['mean'] for law in OLD_LAWS]) for shape in SHAPES}
    prior=old['delta']['mean']
    points=arr.mean((1,2,3))
    # Sampling uncertainty of source catalogs only. The 12 replicates under two source laws are independent.
    boot=np.random.default_rng(40401)
    per_cat=arr.mean(3)
    draws=boot.integers(12,size=(10000,2,2,12))
    draws=per_cat[np.arange(2)[None,:,None,None],np.arange(2)[None,None,:,None],draws].mean((2,3))
    ci=np.quantile(draws,[.025,.975],axis=0)
    cells={}
    for si,ci_idx,li in product(range(2),range(3),range(2)):
        shape,contour,law=SHAPES[si],CONTOURS[ci_idx],LAWS[li]
        point=float(points[si,ci_idx,li])
        interval=ci[:,si,ci_idx,li].tolist()
        cells[f'{shape}_{contour}_{law}']={'delta':point,'sign':'win' if point<0 else 'loss','historical95':interval,
            'historical_contains_zero':interval[0]<=0<=interval[1],'pooled_prior':prior,'tangent_only':float(tangent[shape]),
            'law_replay_no_contour':float(points[si,0,li]),'geometry_increment':point-float(points[si,0,li])}
    result={'quantity':'optimizer minus fixed Brier','cells':cells,'reference_pooled_v39':prior,'reference_tangent_v39':tangent,
            'historical_query_direction_retention':{o:{t:retention(o,t) for t in LAWS} for o in OLD_LAWS},
            'historical_raw':arr.tolist(),'historical_catalogs_per_source_shape_law':12,
            'historical_runtime_sha256':hashlib.sha256((OLD/'results/completion.json').read_bytes()).hexdigest(),
            'sensor_offset_m':BUMP.tolist(),'historical_geometry_covariate':'quarter systematic objects, 2 paired sensor replicates; full base archive',
            'unvalidated':'nominal signed friction proxy; no new-law calibration simulation, static/kinetic offsets or unseen labels'}
    (ROOT/'forecast40.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for a,v in cells.items():print(a,v['delta'],v['historical95'])
if __name__=='__main__':
    main()
