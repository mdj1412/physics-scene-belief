"""Assign only this project's active workers to separate allocated CPU cores.

No allocation/cgroup changes; a denied sched_setaffinity is left denied.
"""
import argparse,json,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--cpus',type=int,nargs=6,required=True);a=p.parse_args()
root=Path.cwd().resolve();rows=[]
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():continue
    try:
        if proc.stat().st_uid!=os.getuid() or (proc/'cwd').resolve()!=root:continue
        cmd=(proc/'cmdline').read_bytes().decode().split('\x00')
        scripts={Path(x).name for x in cmd}
        if 'run.py' in scripts:i=0
        elif 'pixel_only.py' in scripts:i=1
        elif 'conditional_view.py' in scripts:i=3 if 'results/conditional_refined' in cmd else 2
        elif 'audit.py' in scripts:i=4
        elif 'finish_analysis.py' in scripts:i=5
        else:continue
        before=sorted(os.sched_getaffinity(int(proc.name)))
        for task in (proc/'task').iterdir():os.sched_setaffinity(int(task.name),{a.cpus[i]})
        rows.append(dict(pid=int(proc.name),cmd=cmd,before=before,after=sorted(os.sched_getaffinity(int(proc.name)))))
    except (FileNotFoundError,ProcessLookupError):continue
out=Path('results/validation')/('cpu_affinity_'+os.environ.get('SLURM_JOB_ID','unknown')+'.json')
out.write_text(json.dumps(rows,indent=2));print(json.dumps(rows),flush=True)
