import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

rows=[];fig,axs=plt.subplots(3,3,figsize=(13,10),sharex=True)
for ci,case in enumerate([1,25,49]):
    fn=f'case_{case:03d}.json'
    main=json.loads((Path('results/main')/fn).read_text())
    pixel=json.loads((Path('results/pixel_only')/fn).read_text())
    main['policies']['pixel_only']=pixel['policies'];main['records'].extend(pixel['records'])
    audit=json.loads((Path('results/audit')/fn).read_text())
    small=json.loads((Path('results/conditional_view')/fn).read_text())['records']
    large=json.loads((Path('results/conditional_convergence')/fn).read_text())['records']
    record=dict(case=case,family=main['family'],posterior={},conditional={},bank={})
    for qi,q in enumerate(['high','low','pixel_only']):
        a=next(r for r in main['records'] if r['name']=='view' and r['quality']==q and r['mode']=='boundary')
        bs=[r for r in audit['records'] if r['name']=='view_refit' and r['quality']==q]
        record['posterior'][q]=dict(main_b_mm=a['parameters']['mean'][4]*1e3,audit_b_mm=[b['parameters']['mean'][4]*1e3 for b in bs],
            main_mse=float(np.mean(a['metrics']['mse'])),audit_mse=[float(np.mean(b['metrics']['mse'])) for b in bs],
            prediction_rmse=[float(np.sqrt(np.mean((np.array(a['metrics']['mean'])-np.array(b['metrics']['mean']))**2))) for b in bs],
            main_acceptance=a['cost']['acceptance'],audit_acceptance=[b['cost']['acceptance'] for b in bs])
        record['conditional'][q]={}
        for mode in ['bbox','boundary']:
            gain=np.array(main['policies'][q][mode]['gains']);sg=gain.copy();lg=gain.copy()
            sg[:,0]=small[q][mode]['gain'];lg[:,0]=large[q][mode]['gain']
            record['conditional'][q][mode]=dict(small_view_gain=sg[:,0].tolist(),large_view_gain=lg[:,0].tolist(),
                small_choice=sg.argmax(1).tolist(),large_choice=lg.argmax(1).tolist(),changes=int(np.sum(sg.argmax(1)!=lg.argmax(1))),
                small_ess_min=small[q][mode]['ess_min'],large_ess_min=large[q][mode]['ess_min'],
                small_eig=small[q][mode]['eig'],large_eig=large[q][mode]['eig'])
        ax=axs[qi,ci];g=np.array(main['policies'][q]['boundary']['gains']);x=np.array([-30,-17,-7,4,15,28])
        ax.plot(x,g[:,0],'o--',label='view: bank');ax.plot(x,small[q]['boundary']['gain'],'s-',label='view: conditional 64x512')
        ax.plot(x,large[q]['boundary']['gain'],'^-',label='view: conditional 128x2048');ax.plot(x,g[:,1:].max(1),'k:',label='best probe: bank')
        ax.set_title(main['family']+' / '+q);ax.set_ylabel('Expected squared-velocity reduction');ax.set_xlabel('Query source offset [mm]')
        if qi==0 and ci==0:ax.legend(fontsize=7)
        if q!='pixel_only':
            aa=[r['policies'][q] for r in audit['records'] if r['name']=='acquisition']
            record['bank'][q]={mode:dict(main_choice=np.array(main['policies'][q][mode]['gains']).argmax(1).tolist(),
                audit_choice=[np.array(p[mode]['gains']).argmax(1).tolist() for p in aa]) for mode in ['bbox','boundary']}
    rows.append(record)
out=Path('results/analysis');out.mkdir(parents=True,exist_ok=True)
(out/'convergence.json').write_text(json.dumps(rows,indent=2))
fig.tight_layout();fig.savefig(out/'acquisition_convergence.png',dpi=160)
print('audit analysis complete')
