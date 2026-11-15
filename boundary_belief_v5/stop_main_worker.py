"""Terminate only the identified main worker before a controlled resume."""
import argparse,json,os,signal,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--pid',type=int,required=True);a=p.parse_args()
proc=Path('/proc')/str(a.pid)
if not proc.exists():
    print('worker already exited',a.pid,flush=True);raise SystemExit(0)
assert proc.stat().st_uid==os.getuid()
assert (proc/'cwd').resolve()==Path.cwd().resolve()
cmd=(proc/'cmdline').read_bytes().decode().split('\0')
assert 'run.py' in cmd and 'queue_run.py' not in cmd,cmd
os.kill(a.pid,signal.SIGTERM)
for _ in range(100):
    if not proc.exists():break
    time.sleep(.1)
assert not proc.exists(),'worker has not exited; do not resume yet'
out=Path('results/validation')/f'stopped_main_{a.pid}.json'
out.write_text(json.dumps(dict(pid=a.pid,cmd=cmd,time=time.time(),reason='resume missing worlds with shared queue'),indent=2))
print('stopped own main worker',a.pid,flush=True)
