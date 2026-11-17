"""Aggregate common draw-level results without treating shots/draws as independent rubbers."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
N={'hexer_duro':506,'power_3':321,'grass_dtecs':299,'diamant':168,'blowfish':171,'abs_ii_soft':158}
BUDGETS=(0,1,2,4,8,16,32)
def run():
    tape=np.load(ROOT/'core'/'draw_indices.npz')
    for g in N:
        assert tape[g].shape==(256,32)
        assert all(len(set(r))==32 for r in tape[g])
    m={}
    for p in ['core/draw_metrics.csv','baselines/draw_metrics.csv','baselines/gated_eb_draw_metrics.csv','analysis/elastic_draws.csv','analysis/cov_draws.csv']:
        with (ROOT/p).open() as f:
            for r in csv.DictReader(f):
                name=r['method'];g=r['rubber'];k=int(r['budget']);draw=int(r['draw'])
                a=m.setdefault((name,g,k),np.full((256,2),np.nan))
                assert not np.isfinite(a[draw,0])
                a[draw]=[float(r['velocity']),float(r['spin'])]
    assert all(np.isfinite(x).all() for x in m.values())
    methods=sorted({key[0] for key in m})
    out={'methods':methods,'budgets':BUDGETS,'group_counts':N,'overall':{},'folds':{},'paired':{}}
    for name in methods:
        out['overall'][name]={}
        for k in BUDGETS:
            panel=np.stack([m[name,g,k] for g in N])
            macro=panel.mean(0); micro=np.average(panel,axis=0,weights=list(N.values()))
            out['overall'][name][str(k)]={'macro':macro.mean(0).tolist(),'micro':micro.mean(0).tolist(),
                                            'draw_macro_p10_p90':np.quantile(macro,[.1,.9],axis=0).tolist()}
            for g,v in zip(N,panel):
                fold=out['folds'].setdefault(g,{}).setdefault(name,{})
                fold[k]={'mean':v.mean(0).tolist(),'p10p90':np.quantile(v,[.1,.9],axis=0).tolist()}
    for a,b in [('gate_expert','gate_only'),('gate_expert','routed_expert_only'),('gate_expert','inclined_rigid'),
                ('gate_expert','single_affine_ridge'),('gate_expert','single_affine_eb'),
                ('gate_expert_cov','single_affine_eb'),('gated_expert_eb','single_affine_eb'),('elastic_response','gated_expert_eb'),('elastic_response','single_affine_eb')]:
        deltas={}
        for k in BUDGETS:
            panel=np.stack([m[a,g,k]-m[b,g,k] for g in N])
            macro=panel.mean(0);grp=panel.mean(1)
            loo=np.stack([np.delete(grp,g,axis=0).mean(0) for g in range(6)])
            deltas[k]={'mean':macro.mean(0).tolist(),'paired_draw_p10p90':np.quantile(macro,[.1,.9],axis=0).tolist(),
                       'mc_se':(macro.std(0,ddof=1)/np.sqrt(len(macro))).tolist(),'winning_products':(grp<0).sum(0).tolist(),
                       'leave_one_product_range':[loo.min(0).tolist(),loo.max(0).tolist()]}
        out['paired'][a+' minus '+b]=deltas
    (ROOT/'analysis'/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':run()
