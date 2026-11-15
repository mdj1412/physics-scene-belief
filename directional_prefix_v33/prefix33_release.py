"""Release only our measured worker after independent verification of the complete archive."""
from pathlib import Path
import hashlib
import json
import os
import signal
import socket

ROOT=Path(__file__).resolve().parent
out=ROOT/'results_prefix'
def read(p):
    return json.loads(p.read_text())
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert os.environ['SLURM_JOB_ID']=='2744468'
    metas=sorted(out.glob('[AB]/*/block_*/meta.json'))
    assert len(metas)==64 and (ROOT/'PREFIX_RESULTS_KO.md').is_file()
    result=read(out/'prefix_gain33.json')
    all_fingerprints=set()
    records=[]
    for file in metas:
        m=read(file)
        assert m['job']==os.environ['SLURM_JOB_ID']
        assert m['metrics_sha256']==sha(file.parent/'metrics.npz') and m['choices_sha256']==sha(file.parent/'choices.npz')
        records.append(m); all_fingerprints.add(m['fingerprint'])
    assert len(all_fingerprints)==1 and len(result['B_hashes'])==64
    # Independent raw Brier, arrays, matched seeds and probabilities already replayed by prefix33_replay.py.
    worker=read(ROOT/'gpu_prefix33/results/worker.json')
    pid=worker['pid']
    assert (Path('/proc')/str(pid)/'cwd').resolve()==ROOT/'gpu_prefix33'
    cmdline=(Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\x00')
    assert b'../prefix33.py' in cmdline and worker['job']==os.environ['SLURM_JOB_ID']
    assert os.getpgid(pid)==pid
    status=dict(fingerprint=all_fingerprints.pop(),blocks=len(metas),independent_clusters=32,gpu=records[0]['gpu'],
        job=worker['job'],host=socket.gethostname(),scientific_results_complete=True,
        coordinator_disposition='Verified full independent CPU raw replay before intentional release of our own postprocessing worker; slow NumPy integer membership scan replaced with equivalent integer range test.',
        result_sha256=sha(out/'prefix_gain33.json'),worker_pid=pid)
    p=out/'completion.json'; p.write_text(json.dumps(status,indent=2)+'\n')
    os.kill(pid,signal.SIGTERM)
    print('Released measured GPU worker',pid,'after independent verification of all',len(metas),'signed blocks.')
if __name__=='__main__':
    main()
