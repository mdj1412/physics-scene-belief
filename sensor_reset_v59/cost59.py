"""Incremental contact, reset, acquisition, and optical capture ledger."""
from pathlib import Path
import json
import os

ROOT=Path(__file__).resolve().parent
assert os.getenv('SLURM_JOB_ID') and os.getenv('CUDA_VISIBLE_DEVICES')==''
s=json.loads((ROOT/'score59.json').read_text())
cost={}
for method,r in zip(s['methods'],s['final_deployment_impact_rate']):
    if method=='oracle':continue
    # Unconditional policies are charged the sign-only no-reset blind matched pilot.
    need_reset='r1' in method
    need_cam='c1' in method
    records=[]
    for horizon in (1,2,8):
        for impact_slot in (1.04,2.08):
            pilot_physical=8*(1.04+impact_slot)
            acquired=3*1.04+4*.44+(1-r)*1.04+r*impact_slot
            cap=3*1.04+4*.44+max(1.04,impact_slot)
            records.append(dict(sessions=horizon,impact_plus_reset_s=impact_slot,
                                unpadded_s=pilot_physical/horizon+acquired,
                                common_padded_physical_s=pilot_physical/horizon+cap,
                                additional_session_setup_multiplier=4/horizon,
                                additional_sensor_reset_multiplier=(4 if need_reset else 0)/horizon,
                                additional_nonoverlapped_camera_overhead_per_clip=(16 if need_cam else 0)/horizon))
    cost[method]=dict(deployment_I=r,paid_pilot_impacts=8,paid_pilot_contacts=16,paid_pilot_mechanical_resets=16,
                      distinct_objects=8,pilot_sessions=4,pilot_electronic_sensor_resets=4 if need_reset else 0,
                      pilot_camera_clips=16 if need_cam else 0,pilot_camera_frames=96 if need_cam else 0,
                      electronic_signed_readings_offered=32,examples=records)
(ROOT/'cost59.json').write_text(json.dumps(cost,indent=2)+'\n')
