"""Write the paired contour-control appendix after all checkpoints complete."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parent


def signed(value):
    return f'{value:+.6f}'


def interval(row):
    return '['+', '.join(signed(x) for x in row['pointwise95'])+']'


def main():
    d=json.loads((ROOT/'geometry39.json').read_text())
    lines=['# 같은 catalog에서 contour perturbation만 바꾼 대조','',
           '각 조건의 12개 v39 catalog에서 하나씩, 짝수 catalog는 첫 block, 홀수 catalog는 둘째 block을 선택했다. 같은 catalog, alpha, base curvature, calibration·deployment indices, sensor noise, QMC bank를 유지하면서 shelf를 v38 shoulder perturbation으로, ridge를 v38 bead perturbation으로 대체했다. 네 번째 관측, deployment 센서 배정 및 선택된 센서의 예측 확률·query stick label을 각 contour로 다시 생성했다. 첫 세 접촉 속도 응답과 paid row indices가 원래 archive와 정확히 같음을 확인했다. 물체 위치가 바뀌면 달라져야 하는 factual profile·label은 새로 계산했다. 이 비교는 v38의 curvature 배율이나 catalog를 복원하지 않는다.','',
           'Δ는 continuous Brier − fixed Brier다. Δ의 변화는 같은 catalog에서 v39 contour로 바꾼 값에서 대응하는 v38 contour의 값을 뺀 것이다. 양수는 이 교체로 optimizer의 상대 이득이 작아졌음을 뜻한다. 95% 구간은 이 추가 진단의 pointwise t interval이다.','',
           '| Cell | v38 contour Δ | v39 contour Δ | Paired 변화 [95%] | 센서 배정 변경률 | query label 변경률 |',
           '|---|---:|---:|---:|---:|---:|']
    for n,cell in enumerate(d['cells']):
        q=d['cells_new_minus_old'][n]; a,y=d['mean_action_change_and_label_change'][n]
        lines.append(f"| {cell.replace('_',' / ')} | {signed(d['cells_old'][n]['point'])} | {signed(d['cells_new'][n]['point'])} | {signed(q['point'])} {interval(q)} | {a:.2%} | {y:.2%} |")
    lines+=['','| Contrast of paired Δ changes | 추정 [95%] |','|---|---:|']
    names=('Shelf에서 strengthening − weakening','Ridge에서 strengthening − weakening','두 형상의 law interaction 차이 · ridge − shelf','네 cell 평균')
    for n,x in zip(names,d['change_contrasts'].values()):
        lines.append(f"| {n} | {signed(x['point'])} {interval(x)} |")
    lines+=['','이 비교에서 변한 것은 slope와 그 적분의 shape perturbation이다. Law나 catalog만 바꾸었을 때의 효과, 모든 가능한 형상에서의 일반화, 실제 표면에서의 원인을 추정하지 않는다. [전체 paired 수치](geometry39.json) · [실행 코드](geometry_control39.py) · [고정 설계 시점](geometry_frozen_utc.txt)']
    (ROOT/'GEOMETRY_CONTROL_KO.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()
