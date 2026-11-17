"""Repeat the unchanged five-arm crossover with a quiet CPU allocation."""
import json
import os
from pathlib import Path
import subprocess
import time
from runner45 import ROOT,old,infer,torch
from timing45 import run

def snapshot():
    def read(p):
        try:
            return Path(p).read_text().strip()
        except OSError:
            return None
    # cgroup mounts may expose v1 or v2 depending on the OS image.
    cgroups = read("/proc/self/cgroup")
    stats = {}
    for row in cgroups.splitlines():
        _,subsystems,scope = row.split(":",2)
        if "cpu" in subsystems.split(","):
            parent = Path("/sys/fs/cgroup/cpu")/scope.lstrip("/")
        elif not subsystems:
            parent = Path("/sys/fs/cgroup")/scope.lstrip("/")
        else:
            continue
        for name in ("cpu.stat","cpu.pressure","cpu.max"):
            value = read(parent/name)
            if value is not None:
                stats[name]=value
    return dict(time=time.time(),allowed_cpus=sorted(os.sched_getaffinity(0)),cgroups=cgroups,
                cgroup_cpu=stats,cpu_psi=read("/proc/pressure/cpu"),allocation=os.environ["SLURM_JOB_ID"],
                gpu=subprocess.check_output(["nvidia-smi","--query-gpu=uuid,clocks.sm,temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits"],text=True).strip())

def main():
    if torch.cuda.device_count()!=1 or not os.getenv("SLURM_JOB_ID"):
        raise RuntimeError("Expected an isolated single GPU allocation")
    torch.set_num_threads(1)
    data=old.OUT
    target=ROOT/"timing_repeat45"
    target.mkdir(exist_ok=True)
    for p in [data/"worlds",*sorted(data.glob("panel_*"))]:
        alias=target/p.name
        if not alias.exists():alias.symlink_to(p, target_is_directory=True)
    old.OUT=target
    axes=json.loads((ROOT/"source_axis45.json").read_text())
    shots=[]
    with torch.inference_mode():
        engine=infer.make_bank()
        for panel in range(12):
            before=snapshot()
            run(panel,engine,axes)
            shots.append(dict(panel=panel,before=before,after=snapshot()))
            (target/"resources.json").write_text(json.dumps(shots,indent=2)+"\n")
            print("repeat",panel,"complete",flush=True)
    (target/"complete.json").write_text(json.dumps(dict(panels=12,node=os.uname().nodename),indent=2)+"\n")

if __name__=="__main__":
    main()
