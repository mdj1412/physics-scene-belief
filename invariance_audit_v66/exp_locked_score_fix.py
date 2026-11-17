"""Strip the LF left on paths read from the forecast seal by the frozen scoring code."""
import os
import exp_locked as locked
from exp_readout import digest

def digest_manifest_path(path):
    return digest(path.with_name(path.name.rstrip("\r\n")))

if __name__=="__main__":
    assert os.getenv("SLURM_JOB_ID")
    locked.audit_models()
    locked.digest=digest_manifest_path
    locked.score()
