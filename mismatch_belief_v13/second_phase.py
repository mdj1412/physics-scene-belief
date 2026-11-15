import os,time,shutil
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID')
assert Path('results/compact_done.json').exists()
import run_banks,assemble,benchmark
run_banks.main();assemble.main();benchmark.main()
