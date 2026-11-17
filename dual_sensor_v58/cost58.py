"""Count contacts and resets, separately from sensor readouts, for common budget."""
from pathlib import Path
import json
import os
ROOT=Path(__file__).resolve().parent
assert os.getenv("SLURM_JOB_ID") and os.getenv("CUDA_VISIBLE_DEVICES")==""
d=json.loads((ROOT/'score58.json').read_text())
arms=d['methods']
rates=d['deployment_final_impact_rate']
out={}
for m,r in zip(arms,rates):
    if m=='oracle':continue
    g=8 if m=='dual8_map' else 4
    # For constants these are matched physical pilot costs, with 4 session setups.
    rows=[]
    for deployments in [1,2,8]:
        for impact_reset in [1.04,2.08]:
            ref_reset=1.04
            pilot=8*(ref_reset+impact_reset)
            deployment=3*ref_reset+4*.44+ref_reset*(1-r)+impact_reset*r
            cap=pilot/deployments+3*ref_reset+4*.44+max(ref_reset,impact_reset)
            rows.append(dict(deployments=deployments,impact_plus_reset_sec=impact_reset,
                 contacts_total=16+deployments*8,resets_total=16+deployments*8,
                 allocated_elapsed_sec_per_deployment=cap,
                 unpadded_expected_elapsed_sec_per_deployment=pilot/deployments+deployment,
                 additional_setup_coefficient=g/deployments))
    out[m]={'bought_impact_rate':r,'physical_pilot_contacts':16,'physical_pilot_resets':16,
            'allocated_simultaneous_readings':32,'offered_specimens':8,'pilot_session_count':g,'rows':rows}
(ROOT/'cost58.json').write_text(json.dumps(out,indent=2)+'\n')
