"""Host microbenchmark of the optional optical sign decoder on rendered line scans."""
import json
import os
import socket
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import optical53

ROOT = Path(__file__).resolve().parent


def main():
    assert os.getenv('SLURM_JOB_ID')
    decode = optical53.read_scans
    rng = np.random.default_rng(530172)
    repeats = rng.integers(256, size=(12, 4))
    clock = np.empty((12, 120, 6))
    for panel in range(12):
        batches = []

        def capture(images):
            # Capture the actual four environment images passed by the optical experiment.
            batches.append(np.ascontiguousarray(images[np.arange(4), repeats[panel]]))
            return decode(images)

        with patch.object(optical53, 'read_scans', capture):
            optical53.readings(panel)
        assert len(batches) == 6 and all(im.shape == (4, 2, 2, 49) for im in batches)
        expected = [decode(im) for im in batches]
        for rep in range(-1, 120):
            for j in rng.permutation(6):
                start = time.perf_counter_ns()
                sign = decode(batches[j])
                stop = time.perf_counter_ns()
                np.testing.assert_array_equal(sign, expected[j])
                if rep >= 0:
                    clock[panel, rep, j] = (stop - start)*1e-6
    np.savez_compressed(ROOT / 'scan_latency53.npz', host_ms=clock, repeat_index=repeats)
    result = dict(shape=[4,2,2,49], unit='milliseconds per four sessions', ms_p90=float(np.quantile(clock,.9)),
                  ms_median=float(np.median(clock)), samples=clock.size, catalogs=12,
                  node=socket.gethostname(), job=os.getenv('SLURM_JOB_ID'))
    (ROOT / 'scan_latency53.json').write_text(json.dumps(result, indent=2))
    print(result, flush=True)


if __name__ == '__main__':
    main()
