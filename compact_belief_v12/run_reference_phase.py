import os,time,json
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID')
assert Path('results/compact_stage_done.json').exists()
import run_baselines
import benchmark
run_baselines.main()
benchmark.main()
Path('results/reference_phase_done.json').write_text(json.dumps(dict(time=time.time())))
