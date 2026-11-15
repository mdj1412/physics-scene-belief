"""Post-primary diagnostics from frozen arrays; no policy changes."""
import os
import numpy as np
from common import *
from analyze import ci,CONDITIONS
assert os.environ.get('SLURM_JOB_ID')
truth=np.load(R/'data/test/private.npz');target=truth['target'].reshape(-1,2)
aa=np.load(R/'decision_actions.npz')['primary_gate'];idx=np.arange(len(aa));flags=np.load(R/'expansion_flags.npz')['primary_gate']
base=np.load(R/'evaluations/base.npz');expanded=np.load(R/'evaluations/expanded.npz')
bl=.5*((base['predictive'][idx,aa]-target[:,None,:])**2).sum(-1)
el=.5*((expanded['predictive'][idx,aa]-target[:,None,:])**2).sum(-1)
gain=(bl-el).reshape(256,6,16);ff=flags.reshape(256,6,16);pre=np.load(R/'policies/base.npz')['pre_score'].reshape(256,6)>read(R/'thresholds.json')['pre']
scores=base['post_score'][idx,aa].reshape(256,6,16);actions=aa.reshape(256,6)
out={}
for j,c in enumerate(CONDITIONS):
    g=gain[:,j];f=ff[:,j];fields=dict(expansion_fraction_ci=ci(f.mean(1),n=10000),pre_fraction_ci=ci(pre[:,j].astype(float),n=10000),
        mean_expansion_benefit_when_flagged=float(g[f].mean()) if f.any() else None,
        mean_expansion_benefit_when_unflagged=float(g[~f].mean()) if (~f).any() else None,
        loss_reduction_captured=float((g*f).mean()),loss_reduction_missed=float((g*(~f)).mean()),
        action_counts=np.bincount(actions[:,j],minlength=4).tolist())
    fields['by_action']={str(a):dict(worlds=int((actions[:,j]==a).sum()),postscore_median=float(np.median(scores[actions[:,j]==a,j])),expansion_rate=float(ff[actions[:,j]==a,j].mean())) for a in [1,2,3] if (actions[:,j]==a).any()}
    out[c]=fields
cal=np.load(R/'calibration_scores.npz');out['calibration_action_counts']=np.bincount(cal['primary_actions'],minlength=4).tolist()
write(R/'diagnostics.json',out);print(out)
