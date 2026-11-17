"""Independent endpoint audit from held-out per-draw outputs, with a common truth mask."""
from pathlib import Path
import argparse
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
MODELS=('calibrated','majority','elastic','single_affine_eb','source_only','always_reverse')
CUT=8.079025256745311

def interval(per_impact,rng):
    n=len(per_impact)
    idx=rng.integers(n,size=(4096,n))
    lo,hi=np.quantile(per_impact[idx].mean(1),[.025,.975])
    return [float(per_impact.mean()),float(lo),float(hi)]

def evaluate(path):
    with np.load(path) as bank:
        x, y=bank['incident'],bank['target']
        assert len(x)==len(y)
        fast=np.abs(x[:,1])>=CUT
        target=y[fast]
        slip=target[:,0]-target[:,2]
        out={}
        for k in (4,8):
            stats={}; raw={}
            resolved=np.abs(slip)>.2
            pos=slip>0
            for name in MODELS:
                if name=='always_reverse':
                    p=bank[f'elastic_k{k}'][:,fast].copy()
                    sign=np.where((x[fast,0]-x[fast,2])>0,-1.,1.)
                    before=p[:,:,0]-p[:,:,2]
                    delta=np.where((before>0)!=(sign>0), .20*sign-before,0.)
                    p[:,:,0]+=.4*delta
                    p[:,:,2]-=.6*delta
                else:
                    p=bank[f'{name}_k{k}'][:,fast]
                assert np.isfinite(p).all() and p.shape==(256,len(target),3)
                s=p[:,:,0]-p[:,:,2]
                errors=p-target
                hits=(s>0)==pos
                v=np.linalg.norm(errors[:,:,:2],axis=-1)
                w=np.abs(errors[:,:,2])/.02
                neg_recall=hits[:,resolved&~pos].mean()
                pos_recall=hits[:,resolved&pos].mean()
                stats[name]=dict(velocity=float(v.mean()),spin=float(w.mean()),accuracy=float(hits[:,resolved].mean()),balanced=float((neg_recall+pos_recall)/2),negative_recall=float(neg_recall),positive_recall=float(pos_recall))
                raw[name]=(hits[:,resolved].mean(0),v.mean(0),w.mean(0),(s>0).mean(0),hits[:,resolved].mean(1),s)
            a=raw['calibrated']
            comparisons={}
            for name in MODELS[1:]:
                b=raw[name]
                rng=np.random.default_rng(740095)
                d=a[0]-b[0]
                ambiguous=np.abs(a[3][~resolved]-b[3][~resolved]).sum()
                accdiff=a[4]-b[4]
                comparisons[name]=dict(accuracy=interval(d,rng),velocity=interval(a[1]-b[1],rng),spin=interval(a[2]-b[2],rng),
                    unknown_sign_delta_range_full_fast=[float((d.sum()-ambiguous)/len(slip)),float((d.sum()+ambiguous)/len(slip))],
                    paired_draw_wlt=[int((accdiff>1e-12).sum()),int((np.abs(accdiff)<=1e-12).sum()),int((accdiff< -1e-12).sum())],
                    impact_win_tie_loss=[int((d>1e-12).sum()),int((abs(d)<=1e-12).sum()),int((d< -1e-12).sum())])
            retained={}
            for v in ('velocity','spin'):
                adv=stats['majority'][v]-stats['elastic'][v]
                retained[v]=float((stats['majority'][v]-stats['calibrated'][v])/adv) if adv>0 else None
            flip=(raw['elastic'][5]>0)!=(a[5]>0)
            nflip=int(flip.sum())
            coherent=dict(fast_total=nflip,rate=float(flip.mean()),residual_to_boundary_if_changed=float(np.max(abs(np.abs(a[5][flip])-.20))) if nflip else None)
            coherent['source_vs_always_reverse_disagreement']=float(((raw['source_only'][5]>0)!=(raw['always_reverse'][5]>0)).mean())
            coherent['source_vs_always_reverse_disagreement_resolved']=float(((raw['source_only'][5][:,resolved]>0)!=(raw['always_reverse'][5][:,resolved]>0)).mean())
            observed_rev=(pos!=(x[fast,0]-x[fast,2]>0))
            head_ok=(a[5]>0)==pos
            r=resolved&observed_rev
            q=resolved&~observed_rev
            hit_r=float(head_ok[:,r].mean())
            hit_q=float(head_ok[:,q].mean())
            denominator=1-hit_r+hit_q
            subgroups=dict(reversed=int(r.sum()),preserved=int(q.sum()),accuracy_when_reversed=hit_r,accuracy_when_preserved=hit_q,reversal_fraction_break_even=hit_q/denominator if denominator>0 else None)
            out[k]=dict(reversal_groups=subgroups,n=len(target),resolved=int(resolved.sum()),pos=int((pos&resolved).sum()),measures=stats,delta_primary_minus=comparisons,retained_fraction=retained,flips=coherent)
        return out

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('predictions',type=Path,nargs=2);a=ap.parse_args()
    result={key:evaluate(p) for key,p in zip(('official','secondary'),a.predictions)}
    (ROOT/'analysis/paired.json').write_text(json.dumps(result,indent=2)+'\n')
