"""Separate cost of physical contact, electronic reset, optical capture, and terminal decision."""
from pathlib import Path
import json
import os
ROOT=Path(__file__).resolve().parent

def main():
    assert os.getenv('SLURM_JOB_ID') and os.getenv('CUDA_VISIBLE_DEVICES')==''
    report=json.loads((ROOT/'score60.json').read_text())
    names=report['methods']
    result={}
    for population,data in report['uniform_delay'].items():
        result[population]={}
        for idx,(method,prob) in enumerate(zip(names,data['deployment_final_impact_rate'])):
            if idx>=10:continue
            # First eight methods are four layout flags with loss-weighted and MAP variants.
            sync,reset=report['arms_sync_reset'][idx%4] if idx<8 else (0,0)
            uses_camera=idx<8
            table=[]
            for horizon in (1,2,8):
                for impact in (1.04,2.08):
                    pilot=8*(1.04+impact)
                    post=4.88+(1-prob)*1.04+prob*impact
                    table.append(dict(target_sessions=horizon,seconds_per_impact_with_reset=impact,
                                      seconds_worst_case_budget=pilot/horizon+4.88+max(1.04,impact),
                                      seconds_expected=pilot/horizon+post))
            clips=16 if uses_camera else 0
            result[population][method]=dict(impacts=8,references=8,objects=8,mechanical_resets=16,
                                electronic_resets=int(reset)*4,independent_session_setups=4,optical_clips=clips,
                                raw_buffer_frames=8*clips,raw_uint8_bytes=15*27*8*clips,
                                marker_alignment=bool(sync),deployment_I_probability=prob,physical_cost=table)
    (ROOT/'cost60.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__': main()
