"""Condition on one specimen and the observed calibration bank; resample held impacts only."""
from pathlib import Path
import argparse
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("majority", "continuous_loss_gate", "single_affine_eb", "elastic_impulse")
EDGES = (("elastic_impulse", "single_affine_eb"), ("continuous_loss_gate", "majority"))

def interval(values, indices):
    means = values[indices].mean(axis=1)
    return [float(values.mean()), *np.percentile(means, [2.5, 97.5]).tolist()]

def main(cutoff):
    bank = np.load(ROOT/'gpu/control/predictions.npz')
    # Historical outputs use the official 158 rows, then an original-only diagnostic row.
    target = bank['target'][:158]
    normal = abs(bank['incident'][:158, 1])
    valid = abs(target[:, 0] - target[:, 2]) > .20
    positive = target[:, 0] > target[:, 2]
    rng = np.random.default_rng(73021)
    groups = {}
    for label, keep in {'all':np.ones(len(target), dtype=bool), 'fast':normal>=cutoff, 'other':normal<cutoff}.items():
        ids=np.where(keep)[0]
        # Each index is a distinct held-out impact. Calibration-draw average is computed first.
        boot = rng.integers(0, len(ids), (4096, len(ids)))
        errors = {}
        performance = {}
        for k in (4,8):
            for method in MODELS:
                pred = bank[f'{method}_k{k}'][:, :158, :]
                diff = pred - target
                velocity = np.linalg.norm(diff[:, :, :2], axis=2).mean(axis=0)
                spin = np.abs(diff[:, :, 2]).mean(axis=0)/.020
                errors[method,k] = np.column_stack([velocity[ids],spin[ids]])
                signed = pred[:,:,0]-pred[:,:,2]
                same = ((signed>0)==positive).mean(axis=0)
                eligible = keep & valid
                recall = [float(same[eligible & (positive==s)].mean()) if (eligible & (positive==s)).any() else None for s in (False,True)]
                performance[f'{method}_k{k}'] = dict(velocity=float(velocity[ids].mean()),spin=float(spin[ids].mean()),accuracy=float(same[eligible].mean()),negative_recall=recall[0],positive_recall=recall[1],balanced_accuracy=np.mean(recall).item() if None not in recall else None,mean_abs_release=float(np.abs(signed[:,eligible]).mean()),observed_mean_abs_release=float(abs(target[eligible,0]-target[eligible,2]).mean()),pred_inside_margin_on_resolved=float((np.abs(signed[:,eligible])<=.2).mean()),release_mae=float(np.abs(signed[:,eligible]-(target[eligible,0]-target[eligible,2])).mean()))
        contrasts={}
        for a,b in EDGES:
            for k in (4,8):
                d=errors[a,k]-errors[b,k]
                contrasts[f'{a} minus {b}, K{k}']=[interval(d[:,c],boot) for c in (0,1)]
        for m in MODELS:
            d=errors[m,8]-errors[m,4]
            contrasts[f'{m} K8 minus K4']=[interval(d[:,c],boot) for c in (0,1)]
        groups[label] = dict(n=len(ids),pos=int((keep & valid & positive).sum()),neg=int((keep & valid & ~positive).sum()),performance=performance,contrasts=contrasts)
    out = dict(normal_speed_threshold=cutoff,seed=73021,replicates=4096,columns=['mean_delta','resample_p025','resample_p975'],groups=groups)
    (ROOT/'analysis/paired.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('source_fast_cutoff',type=float)
    main(ap.parse_args().source_fast_cutoff)
