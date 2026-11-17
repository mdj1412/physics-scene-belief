"""Audit location metrics on paired bona-fide raster data, with catalog as replicate."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

ROOT=Path(__file__).resolve().parent
HEIGHTS=(19,31)
EFFECTS=('visual','drop_last','bias_minus035','bias_plus035','nominal')
GRID=np.array([18.5,19.,19.5,30.5,31.,31.5])
PROJ=np.array([False,False,False,True,True,False])


def binof(x):
    return np.abs(x[...,None]-GRID).argmin(axis=-1)


def ci(x):
    n=len(x)
    mean=x.mean(0)
    hw=t.ppf(.975,n-1)*x.std(0,ddof=1)/np.sqrt(n)
    return np.stack([mean,mean-hw,mean+hw],axis=-1).tolist()


def run():
    panel_rows=[]; errors=[]; measured=[]; totals=np.zeros((5,6,6),np.int64)
    digests=[]
    for panel in range(12):
        byheight=[]; allerr=[]; allmeasurement=[]
        for h in HEIGHTS:
            arr=[np.load(ROOT/'camera45'/f'panel_{panel:02d}'/f'{shape}_{v}_h{h}.npz')
                 for shape in ('fold','ripple') for v in ('base','hollow','bulge')]
            xy=np.stack([a['x_hat_mm'] for a in arr]); txy=np.stack([a['oracle_x_mm'] for a in arr])
            drop=np.stack([a['x_drop_mm'] for a in arr])
            for a in arr:
                assert np.array_equal(a['visual_project'],PROJ[binof(a['x_hat_mm'])])
                assert np.array_equal(a['drop_project'],PROJ[binof(a['x_drop_mm'])])
                assert np.array_equal(a['oracle_project'],PROJ[binof(a['oracle_x_mm'])])
                digests.extend(a['raster_sha256'].tolist())
                a.close()
            tru=binof(txy)
            items=[]
            for i,x in enumerate((xy,drop,xy-.35,xy+.35,np.full_like(xy,h))):
                b=binof(x)
                totals[i]+=np.bincount((6*tru+b).ravel(),minlength=36).reshape(6,6)
                er=x-txy
                metrics=np.stack([abs(er),er,er*er,(b==tru),(PROJ[b]==PROJ[tru])],axis=-1)
                items.append(metrics)
            group=np.stack(items)
            # methods, (2 shape*3 variant), 3 offset, 4 env,128 object, 5 metric
            byheight.append(group.mean(axis=(1,3,4)))
            allerr.append(xy-txy)
            allmeasurement.append(group.reshape(5,-1,5).mean(axis=1))
        panel_rows.append(np.stack(byheight,axis=1))
        errors.append(np.ravel(allerr))
        measured.append(np.mean(allmeasurement,axis=0))
    errs=np.stack(errors)
    stats=dict(n_unique_clips=int(errs.size),frame_shape=[3,32,96],seed_rule='45032026+128*panel+environment',
               method_order=EFFECTS,by_cell_axes=['method','height=19/31','offset=-.5/0/+.5','metric','mean,lower,upper'],
               metric_order=['MAE_mm','bias_mm','MSE_mm2','nearest_bin_accuracy','projection_choice_accuracy'],
               pooled_ci=ci(np.array(measured)),by_cell_ci=ci(np.array(panel_rows)),
               pooled_abs_err_p50_p90_p99_mm=np.percentile(abs(errs),[50,90,99]).tolist(),
               # absolute position agreement for a future setup requires new camera data.
               max_abs_err_mm=float(abs(errs).max()), confusion_true_rows_pred_cols=totals.tolist(),
               archive_sampling='Each of 432 SHA-256 groups covers the raw raster bytes of 512 clips in object order.',
               sha256_of_ordered_raster_group_digests=hashlib.sha256(''.join(digests).encode()).hexdigest())
    (ROOT/'camera_audit45.json').write_text(json.dumps(stats,indent=2)+'\n')
    return stats


if __name__=='__main__':
    r=run();print(r['n_unique_clips'],r['pooled_ci'],r['pooled_abs_err_p50_p90_p99_mm'])
