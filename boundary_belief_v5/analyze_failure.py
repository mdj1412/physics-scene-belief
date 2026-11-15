import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

rows=[]
for case in [66,68]:
    d=json.loads((Path('results/conditional_failure_audit')/f'case_{case:03d}.json').read_text())
    for r in d['rows']:
        rows.append(dict(case=case,hypothesis=r['hypothesis'],ess=r['ess'],conditional_vs_smc=np.mean([v['disagreement_from_conditional'] for v in r['references']]),between_smc=r['between_smc_seeds_rmse']))
out=Path('results/analysis');(out/'failure_diagnostic.json').write_text(json.dumps(rows,indent=2))
fig,ax=plt.subplots(figsize=(10,4.5));x=np.arange(len(rows))
ax.scatter(x,[r['conditional_vs_smc'] for r in rows],marker='o',s=70,color='#b54a37',label='Conditional estimate vs full SMC (mean of two seeds)')
ax.scatter(x,[r['between_smc'] for r in rows],marker='s',s=50,color='#187c8d',label='Between two full-SMC seeds')
for i,r in enumerate(rows):ax.plot([i,i],[r['between_smc'],r['conditional_vs_smc']],color='#bbb',zorder=0)
ax.set_yscale('log');ax.set_xticks(x,[f"case {r['case']}, h{r['hypothesis']}\nESS={r['ess']:.1f}" for r in rows],rotation=20,ha='right',fontsize=8)
ax.set_ylabel('Posterior-mean disagreement [m/s]');ax.set_title('ESS-selected hypothetical views: numerical diagnostic, not target error');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'conditional_failure.png',dpi=180)
print('numerical failure figure complete')
