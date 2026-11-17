"""Post-primary full-sample convergence of labels, losses and signs (same held-out worlds)."""
import json
from pathlib import Path
import numpy as np
from sensitivity44 import ROOT,bounds

def main():
    folders=('labels44','labels44_full_halfdt','labels44_full_halfmesh')
    all_main=[]; all_delta=[]; gap=[]; subset=[]; loaded=[]
    for panel in range(12):
        files=sorted((ROOT/'labels44'/f'panel_{panel:02d}').glob('*.npz'))
        assert len(files)==48
        per=[]; dd=[]; ss=[]; ld=[]
        arms={tag:{sh:[] for sh in ('fold','ripple')} for tag in folders}
        for f in files:
            path=f.relative_to(ROOT/'labels44')
            with np.load(ROOT/'results'/path) as ar:
                p=ar['pred'][:,0,:,:8]
            rates=[]; errors=[]; matches=[]; late=[]
            for k,folder in enumerate(folders):
                with np.load(ROOT/folder/path) as ar:
                    speed=ar['slip_speed'];y=ar['labels'];valid=ar['valid']
                    assert valid.all()
                    late.append(ar['loaded_end'].mean(axis=(2,3,4)))
                score=((p[:,None,None,None]-y[None,...,None])**2).mean(axis=(-1,-2,-3))
                d=(score[1]-score[0])
                arms[folder][f.name.split('_')[0]].append(np.stack((d[:,:,1], d[:,:,[0,2]].mean(axis=-1)),axis=-1))
                rates.append(score.mean(axis=(1,2,3)))
                if k==0:
                    orig_y=y;orig_v=speed
                else:
                    errors.append(np.stack([(y!=orig_y).mean(axis=(2,3,4)),
                              np.abs(speed-orig_v).mean(axis=(2,3,4))],axis=-1))
                    with np.load(ROOT/folder.replace('_full_','_')/path) as pre:
                        matches.append([int((y[:,:,:,0,:2]!=pre['labels']).sum()),
                                float(np.max(abs(speed[:,:,:,0,:2]-pre['slip_speed'])))])
            ss.append(matches);per.append(rates);dd.append(errors);ld.append(late)
        all_main.append(np.mean(per,axis=0));all_delta.append(np.mean(dd,axis=0));subset.extend(ss);loaded.append(np.mean(ld,axis=0))
        gap.append([[np.mean(arms[tag][shape],axis=0) for shape in ('fold','ripple')] for tag in folders])
    g=np.array(gap)
    chk=np.array(subset)
    out=dict(conditions=folders,number_each=221184,primary_halfdt_halfmesh_abs_brier=bounds(all_main),
             gap_nominal_shifted=bounds(g.mean(axis=(3,4))),
             change_vs_primary_gap=bounds((g[:,1:]-g[:,0:1]).mean(axis=(3,4))),
             label_flip_and_velocity_mae=bounds(all_delta),
             overall_flip_mae=bounds(np.mean(all_delta,axis=(2,3))),
             still_in_contact=bounds(np.mean(loaded,axis=(2,3))),
             prespecified_subset_replay_label_mismatch=chk[:,: ,0].sum(axis=0),
             prespecified_subset_replay_max_velocity_error=chk[:,:,1].max(axis=0))
    (ROOT/'full_numerical44.json').write_text(json.dumps(out,indent=2,default=lambda x:x.tolist())+'\n')
    for k in ('overall_flip_mae','prespecified_subset_replay_label_mismatch','prespecified_subset_replay_max_velocity_error','gap_nominal_shifted','still_in_contact'):
        print(k,out[k])

if __name__=='__main__':
    main()
