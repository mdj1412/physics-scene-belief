"""Mesh/timestep paired replay and directly measured contact manifold characterization."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import t

ROOT=Path(__file__).resolve().parent


def bounds(x):
    x=np.asarray(x)
    u=x.mean(axis=0)
    r=t.ppf(.975,len(x)-1)*x.std(axis=0,ddof=1)/np.sqrt(len(x))
    return np.stack((u,u-r,u+r),axis=-1)


def main():
    changes={k:[] for k in ('halfdt','halfmesh')}
    physical=[]
    losses={k:[] for k in ('primary','halfdt','halfmesh')}
    gaps={k:[] for k in losses}
    pair_counts=np.zeros((2,2,2,2),int)
    for i in range(12):
        files=sorted((ROOT/'labels44'/f'panel_{i:02d}').glob('*.npz'))
        assert len(files)==48
        phys=[];ch={k:[] for k in changes}
        rr={k:[] for k in losses};gg={k:[[],[]] for k in losses}
        for file in files:
            sub=file.relative_to(ROOT/'labels44')
            with np.load(file) as raw:
                v=raw['valid']; assert v.all()
                primary=raw['slip_speed'][:,:,:,0,:2]
                y=raw['labels']
                x=np.stack((v,raw['npoints']>1,raw['span_mm']>.01,raw['span_mm'],raw['loaded_end'],y),axis=-1)
                phys.append(x.mean(axis=(2,3,4)))
                for a in range(2):
                    for j in range(2):
                        pair_counts[a,j]+=np.bincount((y[a,j].ravel().astype(int)*2+y[0,0].ravel().astype(int)),minlength=4).reshape(2,2)
            with np.load(ROOT/'results'/sub) as raw:
                p=raw['pred'][:,0,0,:2]
            speeds={'primary':primary}
            for tag in changes:
                with np.load(ROOT/('labels44_'+tag)/sub) as raw:
                    assert raw['valid'].all()
                    s=speeds[tag]=raw['slip_speed']
                dy=(s<=.02)!=(primary<=.02)
                ch[tag].append(np.stack((dy,np.abs(s-primary)),axis=-1).mean(axis=(2,3)))
            fam=0 if file.name.startswith('fold') else 1
            for tag,sp in speeds.items():
                bits=sp<=.02
                # Each arm's loss on two identical objects across all engine conditions.
                score=((p[:,None,None,None]-bits[None,...,None])**2).mean(axis=(-1,-2))
                rr[tag].append(score.mean(axis=(1,2,3)))
                d=score[1]-score[0]
                gg[tag][fam].append([d[:,:,1].mean(),d[:,:,[0,2]].mean()])
        physical.append(np.mean(phys,axis=0))
        for tag in changes: changes[tag].append(np.mean(ch[tag],axis=0))
        for tag in losses:
            losses[tag].append(np.mean(rr[tag],axis=0))
            gaps[tag].append(np.mean(gg[tag],axis=1))
    out=dict(primary_count=221184,sensitivity_impacts_per_arm=13824,
        fields=['valid','multicontact','span_gt_0p01mm','mean_span_mm','still_loaded','stick_fraction'],
        physical=bounds(physical),paired_tool_class_tables=pair_counts,
        sensitivity={tag:dict(gap_vs_primary=bounds(np.asarray(gaps[tag])-np.asarray(gaps['primary'])),
                     absolute_geometry_gap=bounds(gaps[tag]),flip_speed_mae=bounds(changes[tag]),
                     overall_flip_speed_mae=bounds(np.asarray(changes[tag]).mean(axis=(1,2))),
                     abs_brier=bounds(losses[tag])) for tag in changes},
        main_subset_gaps=bounds(gaps['primary']),main_subset_brier=bounds(losses['primary']))
    (ROOT/'sensitivity44.json').write_text(json.dumps(out,default=lambda x:x.tolist(),indent=2)+'\n')
    print('physical',np.asarray(out['physical'])[...,0])
    for tag,row in out['sensitivity'].items(): print(tag,'flip/mae',row['overall_flip_speed_mae'],'gap',np.asarray(row['gap_vs_primary'])[...,0])

if __name__=='__main__':
    main()
