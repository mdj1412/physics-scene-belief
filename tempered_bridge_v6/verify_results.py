"""Read-only integrity checks of completed inference outputs, no simulation."""
import json
from pathlib import Path
import numpy as np

root=Path('results');phases={'main':(64,1024,12),'outer128':(128,1024,12),'outer256':(256,1024,12),'outer512':(512,1024,12),'budget_audit':(128,2048,18)}
conditions=['high_view','pixel_only_view','high_left','high_right','high_center']
counts={};populations=0;batch_files=0;min_fraction=1.;prefix_checks=0
for phase,(h,n,moves) in phases.items():
    folders=sorted((root/phase).glob('case_*/complete.json'));counts[phase]=len(folders)
    if phase=='main':assert len(folders)==72
    elif phase=='budget_audit':assert len(folders)==5
    else:
        plan=json.loads((root/'adaptive_plans'/f'{phase}.json').read_text())
        assert {int(p.parent.name.split('_')[-1]) for p in folders}==set(plan['cases'])
    for marker in folders:
        folder=marker.parent;meta=json.loads(marker.read_text())
        assert (meta['h'],meta['n'],meta['moves'],meta['reps'])==(h,n,moves,2)
        for condition in conditions:
            observation=np.load(folder/f'{condition}_observation.npy')
            assert len(observation)==h and np.isfinite(observation).all()
            if phase!='main':
                original=np.load(root/'main'/folder.name/f'{condition}_observation.npy')
                assert np.array_equal(original,observation[:64]),(phase,folder.name,condition,'outer prefix mismatch')
                prefix_checks+=1
            for method in ['bridge','full']:
                for rep in [0,1]:
                    stem=folder/f'{condition}_{method}_{rep}';stat=json.loads(stem.with_suffix('.json').read_text())
                    assert (stat['h'],stat['n'],stat['moves'],stat['case'],stat['method'])==(h,n,moves,meta['case'],method)
                    with np.load(stem.with_suffix('.npz')) as data:
                        assert data['mean'].shape==(h,6,2) and data['contact'].shape==(h,6)
                        for key in data.files:assert np.isfinite(data[key]).all(),(str(stem),key,'nonfinite')
                        assert ((data['contact']>=0)&(data['contact']<=1)).all()
                    beta=np.zeros(h);stages=np.zeros(h,dtype=int)
                    for entry in stat['history']:
                        ids=np.array(entry['populations'],dtype=int);bb=np.array(entry['beta']);ess=np.array(entry['ess'])
                        assert len(set(ids.tolist()))==len(ids) and ((ids>=0)&(ids<h)).all()
                        assert np.isfinite(bb).all() and np.isfinite(ess).all()
                        assert ((bb>beta[ids])&(bb<=1)).all(),(str(stem),'temperature schedule')
                        assert ((ess>=.75*n-1e-4)&(ess<=n+1e-4)).all(),(str(stem),'stage ESS')
                        beta[ids]=bb;stages[ids]+=1
                    assert np.array_equal(beta,np.ones(h)) and np.array_equal(stages,stat['stages'])
                    assert np.isfinite(stat['acceptance']).all() and np.isfinite(stat['one_step_ess']).all()
                    min_fraction=min(min_fraction,min(stat['min_stage_ess'])/n)
                    populations+=h;batch_files+=1
result=dict(status='passed',counts=counts,smc_population_batches=batch_files,conditional_smc_populations=populations,
    exact_outer_prefix_checks=prefix_checks,min_stage_ess_fraction=min_fraction,
    checks=['completion plan and budget','finite summaries/log evidence','mean/contact shapes','contact bounds','same first 64 hypothetical observations',
        'strictly increasing temperatures ending at one','pre-resampling ESS threshold','history/stage count agreement'])
(root/'validation/final_integrity.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
