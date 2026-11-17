"""Publish compact Korean methods and numerical checks from the final JSON files."""
import json
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent
def main():
    n=json.loads((H/'controls68.json').read_text())
    s=json.loads((H/'convergence68.json').read_text())
    c=np.array(s['counts']); err=np.array(s['absolute_slip_error_mps_mean_p99_max'])
    delta=np.array(s['mean_fine_minus_standard']); stats=np.array(s['mean_standard_fine'])
    ls=['# v68 · 독립 contact solver의 구현과 수치 검산','',
       '동일한 friction curve를 두 서로 다른 계산에 넣었다. Prescribed solver에서는 위치별 압력의 분포와 시간 경로를 주고 총 impulse를 계산한다. Compliant solver에서는 질량 1 kg인 평면 shoe가 실제로 접근하면서 각 contact site가 눌린 깊이에 따라 힘이 생긴다. 매 step 그 힘으로 물체 속도를 갱신한다. 두 계산은 사고 실험이 같으면서 normal load와 마찰 충격량을 얻는 경로는 다르다. [결과 전 고정한 식과 단위](SOLVER_PLAN_KO.md) · [공통 인터페이스](mechanics68.py) · [C++ 운동 적분](compliant68.cpp)','',
       'Shoe의 회전 자유도는 없고, 각 접점의 normal은 초기 tangent plane으로 근사했다. 마찰계수는 충돌 직전 해당 접점의 속도에서 계산해 한 번의 짧은 접촉 동안 고정했다. Prescribed solver의 radius는 압력 footprint, compliant solver의 동일 수치는 shoe half-width다. 움직이며 접촉면이 갱신되거나 회전이 크게 일어나는 실물까지 이 결과를 확장하려면 별도 검증이 필요하다. 최종 slip은 두 경우 모두 최초 명령 중심의 tangent로 투영한 속도의 절댓값이다.','',
       '## 흩어짐 없는 평면의 해석해','',
       '경사, 입사 속도, restitution과 마찰을 바꾼 320개 별도 조건에서, 평면의 normal 반발과 Coulomb tangent impulse에 대한 해석해를 사용했다. 개발 단계의 별도 임의 형상 물체 48개에서는 prescribed solver의 분포 적분을 과거 v67 계산과 대조했다. [단위 검사 결과](controls68.json)','',
       '| Solver 및 해상도 | Sensor velocity의 최대 절대오차 (m/s) | 평균 절대오차 (m/s) | Frictionless tangent 최대오차 (m/s) |','|---|---:|---:|---:|']
    for k in ('prescribed','compliant','compliant_fine'):
        r=n[k]
        ls.append(f"| {k} | {r['velocity_max']:.3g} | {r['velocity_mean']:.3g} | {r['frictionless_tangent_error']:.3g} |")
    ls.extend(['',f"부호를 뒤집어 얻은 mirror 오차는 0이었다. 강한 마찰 조건에서 잔여 slip의 최대는 {n['compliant']['perfect_stick_max_slip']:.3g} m/s였다. 무작위 toy에서 prescribed terminal slip은 과거 구현과 {n['random_toy_prescribed_v67_max_slip_error']:.3g} m/s 이내로 일치했다. Held-out 비교에서는 기존 prescribed label {s['old_prescribed_count']:,}개 중 {s['old_prescribed_label_xor']}개가 달랐다.",'',
       '## 수치 해상도가 stick 판정과 손익에 미친 영향','',
       '사전에 고른 각 environment의 첫 32개 prototype, 두 radius와 세 offset에 대해 해상도를 높여 다시 계산했다. Prescribed는 공간/시간 Gauss quadrature를 12×6에서 24×12로, compliant는 site 9→17개와 time step 약 4.63→2.31 μs로 바꿨다. Near threshold는 표준 계산에서 slip 0.018–0.022 m/s인 표본이다. 아래 마지막 두 열은 같은 표본, 같은 calibration·router·forecast에서 fine label로 재채점할 때의 Brier contrast 변화다. 이 검산은 label 적분 해상도를 검사하며, history tape의 모든 관측을 fine 해상도로 다시 생성하지 않았다. [해상도별 모든 통계](convergence68.json)','',
       '| Solver | Family | Law | Fine pairs | 전체 label 불일치 | Near threshold 불일치/대상 | Slip 절대오차 p99 (m/s) | Soft−full 차이의 변화 | Restored−dropped 차이의 변화 |',
       '|---|---|---|---:|---:|---:|---:|---:|---:|'])
    for e,en in enumerate(s['engines']):
        for f,fa in enumerate(s['families']):
            for l,law in enumerate(s['laws']):
                r=c[e,f,l]; dv=delta[e,f,l]
                ls.append(f'| {en} | {fa} | {law} | {r[0]:,} | {r[1]:,} ({100*r[1]/r[0]:.3f}%) | {r[3]:,}/{r[2]:,} | {err[e,f,l,1]:.6f} | {dv[0]:+.6f} | {dv[1]:+.6f} |')
    signs=np.sign(stats[:,:,:,0,:])!=np.sign(stats[:,:,:,1,:])
    ls.extend(['',f"같은 fine subset에서 두 해상도 사이에 부호가 달라진 family×law×solver×contrast는 {int(signs.sum())}/{signs.size}개였다. 두 contrast의 최대 평균 변화는 {abs(delta[...,0]).max():.6f}, {abs(delta[...,1]).max():.6f} Brier였다. 독립 단위는 12개 catalog다. 위 수백만 개의 충돌·재채점 결과를 서로 독립적인 실제 장비 실험으로 세지 않았다.",'',
        '서버에 각 target archive의 checksum과 label 생성 시각을 보존했다. [forecast seal](FORECASTS_SEALED) · [solver code seal](solver_plan.sha256) · [label manifest](solver_labels.sha256) · [생성 코드](solver_labels68.py) · [재채점](convergence68.py)'])
    forres={}
    for file in sorted(H.glob('status_worker_*.json')):
        for key,rec in json.loads(file.read_text()).items():
            a=forres.setdefault(key,{})
            for item,n in rec.items():
                if item.startswith('max_') or item.endswith('_max_s') or item.startswith('longest'):
                    a[item]=max(a.get(item,0),n)
                else:a[item]=a.get(item,0)+n
    ls.extend(['','## 종료가 확정되지 않은 충돌','','최초 target label 생성 중, 3.33 ms 이내 모든 평면과 분리됐는지를 검사하는 guard가 일부 행에서 실행을 중지시켰다. 당시 예측은 봉인되어 있었고 최종 Brier는 계산하기 전이었다. 이 행에만 동일 방정식과 같은 step을 사용해 상한을 33.33 ms로 늘렸다. 원래 종료된 행은 재적분하지 않았다. 원 guard가 성공한 경우에만 기록한 모든 학습·평가 history tape는 바뀌지 않는다. 연장 후에도 미확정이면 finite-time 속도 및 명시적인 unresolved mask를 함께 저장했다. [연장 계획](SOLVER_EXTENSION_KO.md) · [미확정 처리](SOLVER_STATUS_KO.md) · [실행 코드](label_status68.py) · [명시적 시각 및 hash](status_plan_files.sha256)','',
        '| Compliant | 전체 충돌 | 연장 | 연장 후에도 미확정 | 마지막 하중이 32 ms 이후인 미확정 |','|---|---:|---:|---:|---:|'])
    for r,k in [('표준','24'),('Fine subset','48')]:
        z=forres[k]
        ls.append(f"| {r} | {z['total']:,} | {z['retry']:,} | {z['still_open']:,} ({100*z['still_open']/z['total']:.4f}%) | {z.get('open_force_near_end',0)} |")
    ls.extend(['','처음 진단한 두 경우에는 0.75–0.76 ms 이후 하중이 없었으나 먼 초기 tangent plane과의 상대속도로 인해 guard가 계속 미확정을 반환했다. 일부 경우에는 시한 가까이에도 하중이 있어 전부 같은 원인으로 단정할 수 없다. 본문의 loss에는 33.33 ms 속도를 표시하되 이 행들의 stick/slip을 임의로 뒤집을 수 있도록 한 bound도 함께 확인해야 한다. 위 fine 비교는 같은 finite-time 규약으로 얻었다.',''])
    (H/'unresolved_counts68.json').write_text(json.dumps(forres,indent=2)+'\n')
    (H/'SOLVER_KO.md').write_text('\n'.join(ls)+'\n')
if __name__=='__main__':main()
