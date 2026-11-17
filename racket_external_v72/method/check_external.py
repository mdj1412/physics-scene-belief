"""Exercise external CSV mode with generated numbers. No empirical scores are reported."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
fields = ["impact_id","v_in_x","v_in_z","spin_before_y","v_out_x","v_out_z","spin_after_y"]
with TemporaryDirectory(prefix="v72_interface_",dir=HERE) as temp:
    folder=Path(temp)
    for split, count, offset in (("calibration",9,0),("test",3,10)):
        with (folder/(split+".csv")).open("w") as f:
            w=csv.writer(f);w.writerow(fields)
            for i in range(count):
                t=i+offset
                w.writerow([f"synthetic-{t}",.4+.1*t,-3.+.03*t,70.-t,-.3+.05*t,2.-.03*t,42.+t])
    out=folder/"check"
    subprocess.run([sys.executable,str(HERE/"bench.py"),"--external",str(folder),"--output",str(out)],check=True)
    with (out/"draws.csv").open() as f:rows=list(csv.DictReader(f))
    run=json.loads((out/"run.json").read_text())
    assert run["mode"]=="external" and len(rows)==256*2*4
    assert set((r["method"],r["k"]) for r in rows)=={(m,k) for m in ("majority","continuous_loss_gate","single_affine_eb","elastic_impulse") for k in ("4","8")}
    assert run["folds"]["external_product"]["target_test"]==3
    (HERE/"interface_check.json").write_text(json.dumps({
        "mode":"external CSV plumbing test using generated numbers, no performance estimate",
        "source_products":6,"synthetic_calibration_rows":9,"synthetic_test_rows":3,
        "metric_rows_checked":len(rows),"passed":True,"device":run["gpu_before"],
        "job":os.getenv("SLURM_JOB_ID"),"step":os.getenv("SLURM_STEP_ID")},indent=2))
