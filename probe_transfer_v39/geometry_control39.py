"""Change only the contour perturbation on the target catalogs, paired to the main run."""
from pathlib import Path
from unittest.mock import patch
import json
import numpy as np
from scipy.stats import t as student_t

import run39 as primary
import physics38

ROOT=Path(__file__).resolve().parent
previous=primary.previous
TARGET_SHAPES={'shelf':'shoulder','ridge':'bead'}


def legacy_contour(y,shape):
    return physics38.perturbation(y,TARGET_SHAPES[shape])


def run_old(cell,catalog,meta,old,detector,artifact):
    shape,law=cell.split('_')
    base=meta['base_seed']
    pair=meta['pair']
    alpha=old['alpha']
    decs=[previous.Engine(pair,*a) for a in alpha]
    risks=np.empty_like(old['risk'])
    max_base_difference=0.
    choice_change=[]; label_change=[]
    for s,sign in enumerate(primary.SIGNS):
        context=shape,law,sign,catalog
        for r,reg in enumerate(primary.REGIMES):
            cal=[primary.world.calibration(seed,a,context,reg,meta['start'])
                 for a,seed in zip(alpha,meta['calibration_seeds'])]
            records=[entry[0] for entry in cal]
            x,y,d=previous.observations(records)
            xx,labels,_,obs=zip(*[primary.world.deployment(seed,a,context,reg)
                                 for a,seed in zip(alpha,meta['deployment_seeds'])])
            xx=np.stack(xx);labels=np.stack(labels)
            # At zero height all four contour profiles have zero perturbation
            # and slope, hence the first three measured velocities must be identical.
            np.testing.assert_array_equal(y[:,:3],old['first3_y'][s])
            np.testing.assert_array_equal(d[:,:3],old['dirs'][s])
            np.testing.assert_array_equal([entry[1] for entry in records],old['ids'][s])
            if r==0:
                choices=previous.prefix_actions(detector,records,pair,xx)
                p=previous.paired_decode(decs,xx,choices,obs)
                choice_change.append(float((choices!=old['actions'][s]).mean()))
                max_base_difference=max(max_base_difference,float(np.abs(p-old['p_base'][s]).max()))
            label_change.append(float((labels!=old['labels'][s,r]).mean()))
            z,w=primary.continuous38.source_optimum(artifact,x,y[:,:3],d[:,:3],y[:,3])
            for m,multiplier in enumerate((previous.SHRINK,w)):
                pred=previous.apply(p,(z*multiplier)[:,None,None],previous.BETA)
                risks[s,r,m]=((pred-labels[:,:,None])**2).mean((1,2))
    return risks,np.mean(choice_change),np.mean(label_change),max_base_difference


def contrast(r):
    return (r[:,:,1]-r[:,:,0]).mean()


def summary(coeff,x):
    """Different cell catalogs are independent; both contours are paired within a catalog."""
    n=x.shape[-1]
    parts=x.var(1,ddof=1)*np.square(coeff)/n
    se=float(np.sqrt(parts.sum()))
    df=float(parts.sum()**2/(np.square(parts).sum()/(n-1))) if se else n-1
    center=float(coeff@x.mean(1))
    half=float(student_t.ppf(.975,df)*se)
    return dict(point=center,pointwise95=[center-half,center+half],df=df)


def main():
    if not previous.torch.cuda.is_available() or previous.torch.cuda.device_count()!=1:
        raise RuntimeError('One assigned GPU is required for the paired geometry control')
    prior=json.loads((primary.OUT/'completion.json').read_text())
    primary.source_hashes()
    prevhash=previous.digest(ROOT/'geometry_control39.py')
    nblocks=48
    dst=ROOT/'geometry_results'
    dst.mkdir(exist_ok=True)
    previous.torch.set_num_threads(2)
    with previous.torch.inference_mode():
        artifact=previous.load_artifacts()[previous.SPEED]
        detector=previous.Detector()
        for c,cell in enumerate(primary.CELLS):
            for j in range(12):
                b=j%2
                origin=primary.OUT/cell/f'catalog_{j:02d}'/f'block_{b:02d}.npz'
                meta=json.loads(origin.with_suffix('.json').read_text())
                if previous.digest(origin)!=meta['sha256'] or meta['fingerprint']!=prior['fingerprint']:
                    raise RuntimeError('Invalid paired main archive')
                path=dst/f'{cell}_{j:02d}.npz'
                if path.exists():
                    with np.load(path) as archive:
                        if str(archive['source_hash'])!=meta['sha256'] or str(archive['control_hash'])!=prevhash:
                            raise RuntimeError('Unmatched secondary checkpoint')
                    continue
                with np.load(origin) as arc:
                    keys=('alpha','risk','first3_y','dirs','ids','actions','p_base','labels')
                    raw={k:arc[k] for k in keys}
                # Process-local substitution only; the public primary generator and files remain frozen.
                with patch.object(primary.world,'perturbation',legacy_contour):
                    risk,actions,labels,base=run_old(cell,j,meta,raw,detector,artifact)
                previous.atomic_npz(path,risk_old=risk,risk_new=raw['risk'],source_hash=meta['sha256'],
                                    control_hash=prevhash,action_change=actions,label_change=labels,
                                    max_base_probability_change=base)
                print('geometry',cell,j,b,flush=True)
    x=np.empty((4,12,2));switch=np.empty((4,12,2))
    for c,cell in enumerate(primary.CELLS):
        for j in range(12):
            with np.load(dst/f'{cell}_{j:02d}.npz') as a:
                x[c,j]=contrast(a['risk_old']),contrast(a['risk_new'])
                switch[c,j]=a['action_change'],a['label_change']
    change=x[:,:,1]-x[:,:,0]
    interactions={
        'shelf_strengthening_minus_weakening':np.array([-1,1,0,0]),
        'ridge_strengthening_minus_weakening':np.array([0,0,-1,1]),
        'law_by_shape_on_perturbation':np.array([1,-1,-1,1]),
        'pooled_contour_change':np.full(4,.25)}
    output=dict(raw_blocks=nblocks,cells=primary.CELLS,contours=TARGET_SHAPES,
                old_vs_new_delta_by_catalog=x.tolist(),
                cells_old=[summary(row,x[:,:,0]) for row in np.eye(4)],
                cells_new=[summary(row,x[:,:,1]) for row in np.eye(4)],
                cells_new_minus_old=[summary(row,change) for row in np.eye(4)],
                change_contrasts={k:summary(v,change) for k,v in interactions.items()},
                mean_action_change_and_label_change=switch.mean(1).tolist(),
                fingerprint_primary=prior['fingerprint'],control_hash=prevhash)
    previous.atomic_json(ROOT/'geometry39.json',output)


if __name__=='__main__':
    main()
