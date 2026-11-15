import os,json
from pathlib import Path
import numpy as np
assert os.environ.get('SLURM_JOB_ID')
R=Path('results');out={}
audit=json.loads((R/'numerical/audit_101.json').read_text())
for decoder in ['joint','plugin','mixture','factorized']:
    rows=[r['errors'][decoder] for r in audit]
    out[decoder]={k:float(max(r[k] for r in rows)) if k.startswith('max_') else float(np.mean([r[k] for r in rows])) for k in rows[0]}
selected={}
for name,decoder in [('joint','joint'),('plugin_aware','plugin'),('mixture_aware','mixture'),('factorized','factorized'),('rule_best','joint'),('lmmse','joint')]:
    d61=[];d81=[]
    for i in range(32):
        p=json.loads((R/'test/policies'/f'{i:04d}.json').read_text())
        a=p['actions'][name] if name!='lmmse' else json.loads((R/'test/extra_policies'/f'{i:04d}.json').read_text())['action']
        base=np.load(R/'test/evaluation_61'/f'{i:04d}.npz');mid=np.load(R/'numerical/81'/f'{i:04d}.npz');ref=np.load(R/'numerical/101'/f'{i:04d}.npz')
        d61.append(float((ref[decoder+'_loss'][a]-base[decoder+'_loss'][a]).mean()));d81.append(float((ref[decoder+'_loss'][a]-mid[decoder+'_loss'][a]).mean()))
    selected[name]=dict(mean_loss_change_61_101=float(np.mean(d61)),max_world_loss_change_61_101=float(np.max(abs(np.array(d61)))),mean_loss_change_81_101=float(np.mean(d81)))
summary=dict(decoders=out,selected=selected,worlds=32)
(R/'numerical/summary.json').write_text(json.dumps(summary,indent=2))
text=['## Numerical checks와 남은 근사 오차\n',
'Main acquisition은 float64 Gauss–Legendre order41, actual decoder는 order61이다. Independent test의 first32 worlds에서 모든 action과32 noise repeats를 order81과101로 다시 계산했다. Main choices와 main results는 덮어쓰지 않았다.\n',
'| Decoder | 최대 velocity component 차이:61→101 | 최대 차이:81→101 |\n| --- | --- | --- |\n']
for k,v in out.items():text.append(f'| {k} | {v["max_prediction"]:.8g} m/s | {v["max_prediction_81_101"]:.8g} m/s |\n')
text.append('\n| Policy/native decoder | 선택 action의 평균 MSE 변화:61→101 | 변화:81→101 |\n| --- | --- | --- |\n')
for k,v in selected.items():text.append(f'| {k} | {v["mean_loss_change_61_101"]:+.9g} | {v["mean_loss_change_81_101"]:+.9g} |\n')
text.append('\n이는 first32 worlds의 audit이며 전체384 test/stress worlds에 대한 uniform numerical error bound는 아니다. Hard plug-in은 discrete model의 argmax가 바뀌면 prediction이 불연속적으로 변할 수 있다. 매우 작은 method 차이에 대해서는 numerical error와 statistical uncertainty를 함께 봐야 한다.\n')
val=json.loads((R/'validation.json').read_text())
text.append(f'Physics validation에서는 최대 energy excess={val["energy_excess"]:.8g}, finite-difference slope error={val["slope_fd_max"]:.8g}, factorized marginal preservation error={val["factorized_marginal_error"]:.8g}였다.\n')
Path('NUMERICAL_REPORT_KO.md').write_text(''.join(text));print(json.dumps(summary,indent=2))
