"""Load the frozen v36 gate for each selected fourth-probe speed."""
from pathlib import Path
import hashlib
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'query_matched_probe_v36'
sys.path.insert(0, str(SOURCE))
from gate36 import load_artifact, predict

SOURCE_HASH = '277a0b5d892686db7bab2274be64c2f8c87a836f709d0298b487c45b28145842'
FIELDS = ('means', 'precision', 'log_norm')
SPEEDS = (.50, .75, 1.10)

def load_artifacts(path=None):
    frozen_source = SOURCE / 'gate36_frozen.npz'
    if hashlib.sha256(frozen_source.read_bytes()).hexdigest() != SOURCE_HASH:
        raise ValueError('Wrong frozen v36 source artifact')
    original = load_artifact(frozen_source)
    gates = {SPEEDS[0]: original}
    with np.load(path or ROOT/'gate37_frozen.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['speeds'], SPEEDS[1:])
        for ix, speed in enumerate(SPEEDS[1:]):
            adapted = original.copy()
            for field in FIELDS:
                adapted[field] = np.concatenate([original[field][:2], saved[field][ix:ix+1]], axis=0)
            gates[speed] = adapted
    return gates
