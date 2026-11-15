"""One unscored, held-aside synthetic cohort checks the full lock-to-Brier interface on CPU."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import numpy as np
import evaluate32 as ev
from stats32 import audit_block

def main():
    assert not ev.torch.cuda.is_available()
    ev.torch.set_num_threads(2)
    with TemporaryDirectory() as folder:
        ev.OUT=Path(folder)
        # These artificial coefficients never enter training, test design, or policy comparisons.
        d={'mean':[0.]*15,'sd':[1.]*15,'beta':[.1]*15,'intercept':0.,'lam':3.}
        model={a:[d,d] for a in ('broad','repeated')}
        det=ev.ProbeShiftEstimator(seed=3003913,bank_size=8192)
        cfg=json.loads((ev.ROOT/'design_eval.json').read_text())
        # Block 99 and law index 5 are outside all three evaluation seed families.
        # Production writes CUDA device name to metadata; only this label is mocked.
        with patch.object(ev.torch.cuda,'get_device_name',return_value='CPU smoke'):
            ev.block_eval(det,model,'smoke','smoke',cfg,'directional',5,99)
        rec=audit_block(ev.OUT/'directional'/'block_099',{'models':model},'smoke','directional',99)
        assert rec['full'].shape==(0,4)
        print('Complete policy replay passed. Raw Brier reconstruction error',rec['max_brier_error'])
if __name__=='__main__':main()
