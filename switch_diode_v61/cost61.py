"""Matched acquisition costs, independent of instrumentation outcomes."""
from pathlib import Path
import json
import os
import numpy as np

ROOT=Path(__file__).resolve().parent
assert os.getenv('SLURM_JOB_ID') and os.getenv('CUDA_VISIBLE_DEVICES')==''
with np.load(ROOT/'fixtures'/'panel_00.npz') as f:
    wave_size=f['wave'].dtype.itemsize
    frame_size=f['frames'].dtype.itemsize
score=json.loads((ROOT/'score61.json').read_text())
p_impact=dict(zip(score['methods'],score['uniform']['independent']['deployment_final_impact_rate']))
rows={}
for method,read,reset in [('dual_risk_reset',2,1),('dual_risk_sham',2,0),('gate',-1,0),
                        ('led_risk_reset',1,1),('led_risk_sham',1,0),('nominal_risk_reset',0,1)]:
    clips=16*int(read>=0)
    traces=32*int(read==2)
    rows[method]=dict(impacts=8,references=8,objects=8,sessions=4,mechanical_resets=16,
                      electronic_resets=4*reset,camera_clips_read=clips,
                      camera_frames=8*clips,raw_camera_bytes=15*27*8*clips*frame_size,
                      adc_traces=traces,adc_samples=128*traces,raw_adc_bytes=128*traces*wave_size,
                      contact_switch_edges_requested=16*int(read==2),photodiode_edges_requested=16*int(read==2))
    prob=float(p_impact[method.replace('_risk_', '_')])
    rows[method]['final_impact_purchase_probability']=prob
    prices=[]
    for h in (1,2,8):
        for price in (1.04,2.08):
            pilot=8*(1.04+price)
            # Baseline 4.88 s includes the fixed shared deployment steps.
            deployment=4.88+(1-prob)*1.04+prob*price
            prices.append(dict(horizon=h,seconds_impact=price,
                               padded_seconds_per_session=pilot/h+4.88+max(1.04,price),
                               expected_seconds_per_session=pilot/h+deployment))
    rows[method]['modeled_sensing_reset_cost']=prices
(ROOT/'cost61.json').write_text(json.dumps(rows,indent=2)+'\n')
