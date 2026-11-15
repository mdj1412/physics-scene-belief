"""Post-freeze fidelity audit for rank-change forecasts, held-out labels used here only."""
from pathlib import Path
import json,os
import numpy as np

ROOT=Path(__file__).resolve().parent
R=ROOT/'results'


def row_summary(p,real):
    # Columns: probability forecast and observed flip rate, value forecast,
    # realized common-decoder gain, and continuation allocation indicator.
    vals=np.stack([np.array([d['flip_probability'], h[0],d['priority'],h[1],h[2],h[3]])
                   for d,h in zip(p,real)])
    # Concordance scores report the chance two environments are correctly
    # ordered in a single competitive block, giving exact ties half credit.
    i,j=np.triu_indices(len(vals),1)
    def concordance(score,target):
        order=np.sign(vals[i,score]-vals[j,score])*np.sign(vals[i,target]-vals[j,target])
        return np.mean(.5+.5*order)
    return np.r_[vals.mean(0)[:5],concordance(0,1),concordance(2,3),
                 vals[vals[:,5]>0,1:5].mean(0)-vals[vals[:,5]==0,1:5].mean(0)]


def main():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Run audit on allocated CPU')
    blocks=[]
    for path in sorted(R.glob('block_*/metrics.json')):
        d=json.loads(path.read_text());st=d['stages']['adaptive']
        with np.load(path.with_name('metrics.npz')) as f:
            act=f['audit_action'];cost=f['audit_common'];pred_gap=f['audit_gap']
            sim=[]
            assert [z['budget'] for z in st]==[2,4]
            for j,stage in enumerate(st):
                rows=[]
                for e in stage['active']:
                    flip=act[e,:,j]!=act[e,:,j+1]
                    gain=cost[e,:,j]-cost[e,:,j+1]
                    # Internal post-update estimated gap at changed actions vs observed outcome gain.
                    fv=(np.abs(pred_gap[e,:,j+1])*flip).mean()
                    rows.append([flip.mean(),gain.mean(),fv,e in stage['retained']])
                sim.append(row_summary(stage['details'],rows))
        blocks.append(sim)
    data=np.asarray(blocks);count=len(blocks)
    cfg=json.loads((ROOT/'design.json').read_text())
    assert count==len({e['rep'] for e in cfg['environments']})
    rng=np.random.default_rng(29960119);ids=rng.integers(count,size=(10000,count));b=data[ids].mean(1)
    names=['forecast_flip','observed_flip','forecast_priority','observed_common_gain','updated_model_flip_value',
        'concordance_flip','concordance_common_gain','continued_minus_stopped_observed_flip',
        'continued_minus_stopped_priority','continued_minus_stopped_common_gain','continued_minus_stopped_model_flip_value']
    out=dict(blocks=count,stages=[{k:dict(mean=float(x),ci95=ab.tolist()) for k,x,ab in zip(names,points,bounds)}
                    for points,bounds in zip(data.mean(0),np.quantile(b,[.025,.975],axis=0).transpose(1,2,0))],
            note='Exploratory per-stage pointwise block bootstrap; chance concordance 0.5. Stage two conditions on four survivors.')
    (R/'score_audit.json').write_text(json.dumps(out,indent=2,allow_nan=False))
if __name__=='__main__':main()
