"""Wait for all authorized experiment branches, then analyze complete results."""
import json,shutil,subprocess,sys,time
from pathlib import Path

last=None
while True:
    names=['main','pixel_only','conditional_view','conditional_refined','audit','conditional_failure_audit']
    counts=[len(list((Path('results')/p).glob('case_*.json'))) for p in names]
    ready=counts==[72,72,72,72,3,2]
    if counts!=last:print(dict(zip(names,counts)),flush=True);last=counts
    if ready:
        time.sleep(2)
        # Every output must parse before processing; invalid archived runs are excluded.
        for name in names:
            for p in (Path('results')/name).glob('case_*.json'):json.loads(p.read_text())
        break
    time.sleep(5)
shutil.copyfile('results/conditional_refined/case_049.json','results/conditional_convergence/case_049.json')
Path('results/analysis').mkdir(exist_ok=True)
with open('results/analysis/analysis_stdout.json','w') as f:subprocess.run([sys.executable,'analyze.py'],stdout=f,check=True)
for script in ['analyze_audit.py','figures.py','analyze_failure.py','write_report.py']:
    subprocess.run([sys.executable,script],check=True)
print('all experiment branches and report tables complete',flush=True)
