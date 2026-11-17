"""Secondary timing diagnostics on archived randomized crossovers; does not filter outliers."""
import json
import time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'


def summarize(a,orders):
    n,reps,methods=a.shape
    q=np.quantile(a,.9,axis=(0,1))
    r=q/q[1]
    ratios=[]
    for omit in range(n):
        qout=np.quantile(np.delete(a,omit,axis=0),.9,axis=(0,1))
        ratios.append(qout/qout[1])
    positions=np.argsort(orders,axis=2)
    by_position=[]
    for method in range(methods):
        med=[np.median(a[:,:,method][positions[:,:,method]==i]) if (positions[:,:,method]==i).any() else None
             for i in range(methods)]
        by_position.append(med)
    return dict(p90=q.tolist(),minimum=a.min(axis=(0,1)).tolist(),median=np.median(a,axis=(0,1)).tolist(),
        maximum=a.max(axis=(0,1)).tolist(),per_panel_maximum=a.max(1).tolist(),ratio_vs_est_zero=r.tolist(),
        leave_one_panel_out_ratio_min=np.min(ratios,axis=0).tolist(),leave_one_panel_out_ratio_max=np.max(ratios,axis=0).tolist(),
        median_by_within_sweep_position=by_position,median_by_repeat=np.median(a,axis=0).tolist())


def main():
    while not (OUT/'completion.json').is_file():time.sleep(10)
    run=json.loads((OUT/'completion.json').read_text())
    arrays={k:[] for k in ('full_ms','setup_ms','cold_ms','order')}
    methods=run['timing_methods']
    for panel in range(run['panels']):
        with np.load(OUT/'timing'/f'panel_{panel:02d}.npz') as f:
            np.testing.assert_array_equal(methods,f['methods'])
            for k in arrays:arrays[k].append(f[k])
    arrays={k:np.stack(v) for k,v in arrays.items()}
    out=dict(methods=methods,replicates_per_panel=arrays['order'].shape[1],catalogs=run['panels'],
        gpu=run['gpu'],job=run['job'],
        warm=summarize(arrays['full_ms'],arrays['order']),cold=summarize(arrays['cold_ms'],arrays['order']),
        setup=summarize(arrays['setup_ms'],arrays['order']))
    (ROOT/'timing_audit42.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    w=out['warm'];c=out['cold']
    lines=['# V42 GPU timing의 분산','',
        '한 배치의 8개 환경·1,024개 물체에 대해 같은 GPU에서 각 방법을 무작위 순서로 실행했다. Catalog panel 12개 × 반복 4회의 실측값 48개가 방법별 표본이다. 아래 값은 사후 민감도 분석이다. 모든 측정값을 포함한다.','',
        '| Policy | Warm median | Warm p90 | Warm 최솟값–최댓값 | Cold p90 | 개별 panel 제외 시 warm p90 ratio 범위* |',
        '|---|---:|---:|---:|---:|---:|']
    for i,m in enumerate(methods):
        lines.append(f"| {m} | {w['median'][i]:.2f} | {w['p90'][i]:.2f} | {w['minimum'][i]:.2f}–{w['maximum'][i]:.2f} | "
                     f"{c['p90'][i]:.2f} | {w['leave_one_panel_out_ratio_min'][i]:.4f}–{w['leave_one_panel_out_ratio_max'][i]:.4f} |")
    lines+=['','시간 단위는 ms다. *Ratio의 분모는 동일 재표집에서 estimated + zero의 p90이다. 이 범위는 12회 leave-one-panel-out 결과로, 사전 지정 bootstrap confidence interval은 전체 보고서에 제시했다. '
        '매 반복 내 실행 위치별 median과 각 panel의 최댓값을 [timing 원수치](timing_audit42.json)에 공개했다. '
        'Online elapsed time은 실제 GPU 계산 외에 CPU calibration·selection 및 동기화도 포함한다. Sensor 획득과 source bank 최초 구축은 제외했다.']
    (ROOT/'TIMING_AUDIT_KO.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__': main()
