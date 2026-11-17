"""Terminal-label-blind 17-site history replay with the v68 frozen predictor."""
import json
import os
import sys
import time
from pathlib import Path
import hashlib
import numpy as np
import torch
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H.parent))
import predict68 as main
F=main.frozen
def run():
    assert torch.cuda.device_count()==1 and os.environ.get('SLURM_JOB_ID') and not (H/'FORECASTS_SEALED').exists()
    end=H/'forecasts'/'compliant'
    end.mkdir(exist_ok=True,parents=True)
    axes,heads=main.load_source()
    engine=F.inf.make_bank()
    temp={h:F.inf.make_template(engine,h/1000) for h in F.HEIGHTS}
    artifact=F.inf.r.v38.load_artifacts()[1.1]
    torch.set_num_threads(1)
    begin=time.time()
    source={};output={}
    with torch.inference_mode():
        for k in range(12):
            nwait=0
            while not (H/f'tape_ready_{k}').is_file():
                time.sleep(1);nwait+=1
                if nwait>900:raise TimeoutError(k)
            parts=[];s3=[];s4=[];cells=[];acts=[];cache={}
            folder=H/'tapes'/'compliant'/F.panel(k)
            paths=sorted(folder.glob('*.npz'))
            assert len(paths)==48
            for path in paths:
                source[str(path.relative_to(H))]=main.digest(path)
                with np.load(path) as z:
                    x,q,p=z['xx'],z['query'],z['probe']
                    cue3,cue4=main.shifts(z['cx'],z['cy'],z['dirs'],artifact)
                h=int(path.stem.rsplit('_h',1)[1])
                key=hashlib.sha256(x.tobytes()).digest()
                if key not in cache:cache[key]=F.inf.actions43(x,engine)
                act=cache[key]
                q[act!=F.world.Q]=np.nan
                p[act!=F.world.P]=np.nan
                parts.append(main.predict(x,q,p,act,h,engine,temp[h],axes[h],heads))
                cells.append(path.name);s3.append(cue3);s4.append(cue4);acts.append(act)
            target=end/(F.panel(k)+'.npz')
            np.savez_compressed(target,p=np.stack(parts),shift=s4,shift3=s3,cells=cells,actions=acts)
            output[str(target.relative_to(H))]=main.digest(target)
            print('fine forecast',k,flush=True)
    for k,sha in output.items():assert main.digest(H/k)==sha
    (H/'tapes.sha256').write_text(''.join(f'{sha}  {path}\n' for path,sha in source.items()))
    hashes={str(p.relative_to(H.parent)):main.digest(p) for p in (H.parent/'predict68.py',H/'PLAN_KO.md',H/'fine_tapes.py',H/'fine_predict.py',H/'score_fine.py')}
    seal=dict(outputs=output,inputs=main.digest(H/'tapes.sha256'),hashes=hashes,main_seal=main.digest(H.parent/'FORECASTS_SEALED'),
              device=torch.cuda.get_device_name(),job=os.environ['SLURM_JOB_ID'],duration_s=time.time()-begin,utc=time.time())
    (H/'FORECASTS_SEALED').write_text(json.dumps(seal,indent=2)+'\n')
if __name__=='__main__':run()
