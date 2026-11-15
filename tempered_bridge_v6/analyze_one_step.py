import json,argparse
from pathlib import Path
import numpy as np
from analyze import oldcase,scores,selected,ci,load

assert len(list(Path('results/one_step').glob('case_*.json')))==72
p=argparse.ArgumentParser();p.add_argument('--main-only',action='store_true');args=p.parse_args()
summary=json.loads(Path('results/analysis/summary.json' if args.main_only else 'results/analysis_adaptive/summary.json').read_text());cases=[oldcase(i) for i in range(72)];result={}
for q in ['high','pixel_only']:
    rows=[];dgs=[];conds=[0 if q=='high' else 1,2,3,4]
    for d in cases:
        case=d['case'];bank=np.load(Path('results/one_step')/f'case_{case:03d}.npz');h=summary[q]['budgets'][str(case)]
        folder=Path('results/main' if h==64 else f'results/outer{h}')/f'case_{case:03d}'
        full=load(folder,q,'full');gm=bank['gain'][conds,:h].mean(1).T;g512=bank['gain'][conds].mean(1).T;gf=full['samples'].mean(0)
        active=gf.max(1)>1e-4;choice=gm.argmax(1)
        pred=bank['mean'][conds,:h].transpose(1,2,3,0)
        dgs.append(dict(case=case,split=d['split'],active=active.tolist(),agreement=(choice==gf.argmax(1)).tolist(),
            rmse=float(np.sqrt(np.mean((pred-full['means'].mean(0))**2))),ess_median=float(np.median(bank['ess'][conds,:h])),ess_min=float(np.min(bank['ess'][conds,:h])),
            gain_mae=float(np.mean(np.abs(gm-gf))),regret=(gf.max(1)-gf[np.arange(6),choice]).tolist()))
        if d['split']!='test':continue
        sc=scores(d,q);m=selected(sc,choice);b=selected(sc,np.array(next(r for r in summary[q]['diagnostics'] if r['case']==case)['bridge_choice']));m512=selected(sc,g512.argmax(1))
        rows.append(dict(case=case,snis_mse=float(np.mean(m['mse'])),snis512_mse=float(np.mean(m512['mse'])),bridge_mse=float(np.mean(b['mse']))))
    ds=[d for d in dgs if d['split']=='test'];mask=np.array([d['active'] for d in ds]);agree=np.array([d['agreement'] for d in ds])
    result[q]=dict(matched_mse=float(np.mean([d['snis_mse'] for d in rows])),max512_mse=float(np.mean([d['snis512_mse'] for d in rows])),
        bridge_minus_snis=ci([d['bridge_mse']-d['snis_mse'] for d in rows]),agreement_active=float(agree[mask].mean()),
        prediction_rmse_median=float(np.median([d['rmse'] for d in ds])),prediction_rmse_max=float(max(d['rmse'] for d in ds)),
        diagnostics=dgs,per_case=rows)
Path('results/analysis/one_step_main.json' if args.main_only else 'results/analysis/one_step.json').write_text(json.dumps(result,indent=2));print(json.dumps({q:{k:v for k,v in d.items() if k not in ['diagnostics','per_case']} for q,d in result.items()},indent=2))
