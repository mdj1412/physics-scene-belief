"""Source-frozen full-cohort sensing, then predictions for the defined external holdout."""
import json, os, sys, time
from itertools import product
from pathlib import Path
import numpy as np
import torch

B=Path(__file__).resolve().parent
H=B.parent
sys.path.insert(0,str(H))
import predict68 as main
frozen=main.frozen

def run_panel(k,axes,heads,engine,temp,artifact):
    sub=frozen.panel(k);directory=B/'tapes'/sub
    pdir=B/'forecasts';pdir.mkdir(exist_ok=True)
    if not (directory/'diagnostics.json').exists(): return False
    cells=[];pred=[];shifts=[];actions=[];changes=0
    memo={}
    for key in sorted(directory.glob('*.npz')):
        with np.load(key) as z:
            x=z['xx']; cx=z['cx']; cy=z['cy']; ds=z['dirs']; p=z['probe8']
        shift=main.shifts(cx,cy,ds,artifact)[1]
        cache=main.hashlib.sha256(x.tobytes()).hexdigest()
        if cache not in memo:memo[cache]=frozen.inf.actions43(x,engine)
        a=memo[cache]
        assert a.shape==(4,128) and np.all((a==frozen.world.P).sum(1)==13)
        mask=a[:,:8] == frozen.world.Q
        p[mask]=np.nan
        for h in (19,31):
            fn=key.name.replace('h19.npz','h%d.npz'%h)
            with np.load(H/'tapes'/'prescribed'/sub/fn) as z:q=z['query'][:,:8]
            q[~mask]=np.nan
            probs=main.predict(x[:,:8],q,p,a[:,:8],h,engine,temp[h],axes[h],heads)
            pred.append(probs);cells.append(fn);shifts.append(shift);actions.append(a)
    assert len(cells)==48 and len(set(cells))==48
    np.savez_compressed(pdir/(sub+'.npz'),p=pred,shift=shifts,cells=cells,actions=actions)
    print('external GPU',sub,flush=True)
    return True

def run():
    assert torch.cuda.device_count()==1 and os.getenv('SLURM_JOB_ID')
    assert not (B/'forecasts_sealed.json').exists()
    start=time.monotonic();axes,heads=main.load_source();engine=frozen.inf.make_bank()
    temps={h:frozen.inf.make_template(engine,h/1000.) for h in frozen.HEIGHTS}
    artifact=frozen.inf.r.v38.load_artifacts()[1.1];torch.set_num_threads(1)
    pending=set(range(12))
    with torch.inference_mode():
        while pending:
            complete=set()
            for k in sorted(pending):
                if run_panel(k,axes,heads,engine,temps,artifact):complete.add(k)
            pending-=complete
            if pending and not complete:
                if time.monotonic()-start>1800:raise TimeoutError(sorted(pending))
                time.sleep(3)
    out=dict(utc=time.time(),duration_s=time.monotonic()-start,job=os.environ['SLURM_JOB_ID'],
           device=torch.cuda.get_device_name(),plan=main.digest(B/'PLAN_KO.md'),code=main.digest(Path(__file__)),
           forecasts={p.name:main.digest(p) for p in sorted((B/'forecasts').glob('*.npz'))},
           tapes={str(p.relative_to(B)):main.digest(p) for p in sorted((B/'tapes').glob('*/*.npz'))})
    assert len(out['forecasts'])==12 and len(out['tapes'])==288
    (B/'forecasts_sealed.json').write_text(json.dumps(out,indent=2))
if __name__=='__main__':run()
