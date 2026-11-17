"""Paired uncertainty, label sensitivity, and flat point-contact impulse diagnostic."""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read_predictions():
    empirical = list(csv.DictReader((ROOT/"analysis"/"empirical_predictions.csv").open()))
    rows = list(empirical)
    with (ROOT/"solver"/"predictions.csv").open() as f:
        for r in csv.DictReader(f):
            if r["split"] == "test":
                rows.append(dict(target=r["target"], row=r["row"], model=r["method"], true_vx=r["vx_out"],
                    pred_vx=r["vx_pred"], true_u=float(r["vx_out"])+float(r["rim_out"]),
                    pred_u=r["u_pred"], in_u=float(r["vx_in"])+float(r["rim_in"]),
                    p050=r["p050"], p025=r["p025"], p100=r["p100"], p_rev=r["p_rev"]))
    return rows


def paired_interval(d, size, rng):
    groups = np.arange(len(d)) // size
    counts = np.bincount(groups)
    sums = np.bincount(groups, weights=d)
    idx = rng.integers(len(counts), size=(4000,len(counts)))
    draws = sums[idx].sum(1) / counts[idx].sum(1)
    return np.quantile(draws,[.025,.975]).tolist()


def assess(rows):
    effects, geometry = {}, {}
    for target, source in [('B','A'),('A','B')]:
        methods = {}
        for name in sorted(set(r["model"] for r in rows)):
            subset = sorted((r for r in rows if r["model"]==name and r["target"]==target),
                            key=lambda r:int(r["row"]))
            assert [int(r["row"]) for r in subset] == list(range(len(subset)))
            methods[name] = {k:np.array([float(r[k]) for r in subset]) for k in
                             ('true_vx','pred_vx','true_u','pred_u','in_u','p050','p_rev','p025','p100')}
        pairs = {}
        for left in ['compliant_coulomb','compliant_memory','compliant_memory_refined','two_regime','source_frequency']:
            a,b = methods[left],methods['pybullet']
            y,ui = b['true_u'], b['in_u']
            np.testing.assert_allclose(y,a['true_u'])
            np.testing.assert_allclose(b['true_vx'],a['true_vx'])
            d = {'vx_mae':abs(a['pred_vx']-a['true_vx'])-abs(b['pred_vx']-b['true_vx']),
                 'u_mae':abs(a['pred_u']-y)-abs(b['pred_u']-y)}
            for code,thresh in [('p025',.25),('p050',.50),('p100',1.),('p_rev',None)]:
                event = ui*y<0 if thresh is None else abs(y)<=thresh
                d[code]=(a[code]-event)**2-(b[code]-event)**2
            e = {'effects': {k: {'mean':float(v.mean()),'ci_by_rows_per_block':{str(m):paired_interval(v,m,np.random.default_rng(6900+m)) for m in (12,24,48)}} for k,v in d.items()}}
            for delta in [.1,.25,.5]:
                ambiguous = abs(abs(y)-.5)<=delta
                flipped = (a['p050']-(abs(y)>.5))**2-(b['p050']-(abs(y)>.5))**2
                lo=np.where(ambiguous,np.minimum(d['p050'],flipped),d['p050'])
                hi=np.where(ambiguous,np.maximum(d['p050'],flipped),d['p050'])
                e[f'margin_{delta}']=dict(n=int(ambiguous.sum()),min_delta=float(lo.mean()),max_delta=float(hi.mean()))
            pairs[left]=e
        effects[target]=pairs
        def raw(c):
            return np.loadtxt(ROOT/'data69'/'ESM_2'/f'data_campaign{c}.csv',delimiter=',')*.02135
        st=raw(source)
        st=st[(np.arange(len(st))//24)%4!=1]
        xt=raw(target)
        du, dw = st[:,3]-st[:,0], st[:,5]-st[:,2]
        src_k = (du @ dw) / (dw @ dw)
        vt, wt=xt[:,3]-xt[:,0],xt[:,5]-xt[:,2]
        rb = methods['pybullet']
        # Orthogonal distance to the admissible delta-vx,delta-rim line for each trial.
        tests={}
        for k in [.3,.4,.5,2/3,float(np.clip(src_k,0,2/3))]:
            defect=vt-k*wt
            floor=np.mean(defect**2)/(2*(1+k*k))
            tests[str(k)]={'median_abs_defect':float(np.median(abs(defect))),
                            'paired_component_rmse_floor':float(np.sqrt(floor)),
                            'share_abs_defect_gt_1':float(np.mean(abs(defect)>1))}
        rmserr=(rb['pred_u']-rb['pred_vx'])-(rb['true_u']-rb['true_vx'])
        err2=(np.mean((rb['pred_vx']-rb['true_vx'])**2)+np.mean(rmserr**2))/2
        geometry[target]={'source_fit_k':float(src_k),'k_panel':tests,'bullet_joint_rmse':float(np.sqrt(err2)),
                          'floor_fraction_joint_mse_at_k0.4':float(tests['0.4']['paired_component_rmse_floor']**2/err2)}
    return {'paired':effects,'geometry':geometry}

if __name__=='__main__':
    (ROOT/'analysis'/'paired_audit.json').write_text(json.dumps(assess(read_predictions()),indent=2)+'\n')
