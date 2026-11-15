"""Extend the fourth-action density with the same source draws and v36 generator."""
from pathlib import Path
import hashlib
import json
import os
import numpy as np
from gate37 import ROOT, SOURCE, SOURCE_HASH, SPEEDS, FIELDS, load_artifacts, predict
from gate36 import FAMILIES, SIGNS, load_artifact
from train_gate36 import components, shape, mixture_samples, STRENGTH, SOURCE_SEED, N_COMPONENT, N_PER_COMP

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    current = load_artifact()
    source = mixture_samples().reshape(-1,5)
    by_speed = {v:[] for v in SPEEDS}
    for family in FAMILIES:
        samples = shape(source, family)
        for sign in SIGNS:
            for strength in STRENGTH:
                for speed, target in by_speed.items():
                    target.append(components(samples,family,sign,strength,.023,speed))
    arrays = {}
    for i, field in enumerate(FIELDS):
        values = np.stack([[row[i] for row in by_speed[s]] for s in SPEEDS])
        np.testing.assert_allclose(values[0], current[field][2], atol=2e-11, rtol=2e-13)
        arrays[field] = values[1:]
    target = ROOT/'gate37_frozen.npz'
    tmp = ROOT/'gate37_building.npz'
    np.savez_compressed(tmp,**arrays,speeds=np.asarray(SPEEDS[1:]))
    os.replace(tmp,target)
    gates = load_artifacts()
    # Validate the complete .50 prediction on legacy (v36) observations,
    # available before v37 and never used to fit a parameter or threshold.
    history = SOURCE/'results'/'sinusoid_atan'/'block_000'/'weak_plus.npz'
    with np.load(history,allow_pickle=False) as record:
        args=(record['far_x'],record['near_y'][:,:3],record['dirs'][:,:3],record['near_y'][:,3],'alternating')
        ref = predict(current,*args)
        for speed,gate in gates.items():
            result = predict(gate,*args)
            keys=ref if speed==.50 else ('z','score3','accept3','near_strength3')
            for key in keys:
                np.testing.assert_array_equal(result[key],ref[key])
    summary=dict(speeds=SPEEDS,source_artifact_sha256=SOURCE_HASH,sha256=digest(target),
                 source_seed=SOURCE_SEED,gaussian_components=N_COMPONENT,draws_per_component=N_PER_COMP,
                 gain_and_beta='verbatim v36',height=.023,identical_predictions_at_0p50=True,
                 identical_first_three_across_speeds=True,
                 reconstructed_source_density_at_0p50_max_abs={
                     field:float(np.max(np.abs(np.stack([row[i] for row in by_speed[.50]])-current[field][2])))
                     for i,field in enumerate(FIELDS)},
                 sources={p.name:digest(p) for p in (SOURCE/'train_gate36.py', SOURCE/'gate36.py',ROOT/'gate37.py',Path(__file__))})
    (ROOT/'freeze37.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))
if __name__=='__main__':main()
