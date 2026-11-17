"""Check the extended time window against its exact matched primary objects and record file hashes."""
from pathlib import Path
from hashlib import sha256
import json
import numpy as np
from sensitivity44 import ROOT,bounds


def main():
    rows=[]; nloaded=0;nflip=0;nflip_loaded=0; delta=[]
    for panel in range(12):
        stats=[];dif=[]
        for f in sorted((ROOT/'labels44_horizon'/f'panel_{panel:02d}').glob('*.npz')):
            sub=f.relative_to(ROOT/'labels44_horizon')
            with np.load(ROOT/'labels44'/sub) as ar:
                v=ar['slip_speed'][:,:,:,0,:2]; loaded=ar['loaded_end'][:,:,:,0,:2]
            with np.load(f) as ar:
                vnew=ar['slip_speed']
            with np.load(ROOT/'results'/sub) as ar:
                p=ar['pred'][:,0,0,:2]
            y0=v<=.02;y1=vnew<=.02
            nloaded+=int(loaded.sum());nflip+=int((y0!=y1).sum());nflip_loaded+=int(((y0!=y1)&loaded).sum())
            stats.append([(y0!=y1).mean(),abs(vnew-v).mean(),loaded.mean()])
            # Within-object change in projected minus original Brier.
            dif.append(-2*np.mean((p[1]-p[0]).mean(axis=-1)[None,None,None,:]*(y1.astype(float)-y0)))
        assert len(stats)==48
        rows.append(np.mean(stats,axis=0));delta.append(np.mean(dif))
    out=dict(horizon_ms=[.4,.8],subset_n=13824,prior_still_loaded_n=nloaded,
             event_change_count=nflip,event_changes_within_loaded=nflip_loaded,
             change_velocitymae_loaded_ci=bounds(rows),overall_geometry_gap_change=bounds(delta))
    (ROOT/'horizon44.json').write_text(json.dumps(out,indent=2,default=lambda x:x.tolist())+'\n')
    files=[]
    for directory in ('labels44','labels44_halfdt','labels44_halfmesh','labels44_full_halfdt','labels44_full_halfmesh','labels44_horizon'):
        current=sorted((ROOT/directory).glob('panel_*/*.npz'))
        assert len(current)==576
        files.extend(current)
    (ROOT/'solver_outputs44.sha256').write_text(''.join(f'{sha256(f.read_bytes()).hexdigest()}  {f.relative_to(ROOT)}\n'
                                                       for f in files))
    print(out)

if __name__=='__main__':
    main()
