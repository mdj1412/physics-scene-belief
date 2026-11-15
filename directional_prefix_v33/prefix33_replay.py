"""Fast CPU-only recomputation with an integer-range seed audit and identical estimands."""
from pathlib import Path
import hashlib
import numpy as np
import json

ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    # The originally locked range-membership audit iterates millions of values for
    # np.int64 arguments. The integer cast uses constant-time Python range lookup.
    path=ROOT/'prefix33_stats.py'
    source=path.read_text().replace('any(s in policy_seeds','any(int(s) in policy_seeds')
    ns={'__name__':'prefix33_cpu','__file__':str(path)}
    exec(compile(source,str(path),'exec'),ns)
    result=ns['finalize']()
    print('raw archive reconstructed; B endpoints',result['B']['paired_means'])
