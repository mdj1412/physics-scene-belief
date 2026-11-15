"""Generate tables from the completed run and its immutable pre-outcome forecasts."""
from pathlib import Path
import json
import numpy as np

ROOT=Path(__file__).resolve().parent
def fmt(x):return f'{x:+.6f}'.replace('-','−')
def band(pair):return '['+', '.join(fmt(x) for x in pair)+']'
def load(name):return json.loads((ROOT/name).read_text())
def cname(c):return c.replace('_',' / ')

def main():
    s=load('stats39.json')
    p=load('forecast39.json')
    a=s['timing'];c=s['cells']
    verdicts={'confirmed':'사전 방향 확인','refuted':'반대 방향 확인','uncertain':'방향 미확정'}
    r=['## 사전 방향과 실측 결과','',
       'Δ는 같은 held-out 물체와 noise에서 계산한 **optimizer Brier − fixed 32.5% Brier**다. 음수는 optimizer에 유리하다. Historical Δ는 v39 표본을 만들기 전에 기록한 수치다. 실측 구간은 cell별 독립 catalog 12개를 기준으로 네 대조에 Bonferroni–t 보정을 적용했다.','',
       '| 조건 | 사전 Δ | 고정 32.5% Brier | Optimizer Brier | 실측 Δ [동시 95%] | 판정 | Optimizer 우세 catalog |',
       '|---|---:|---:|---:|---:|---|---:|']
    for k in c:
        x=s['by_cell'][k];risk=x['loss'];f=x['forecast']
        r.append(f'| {cname(k)} | {fmt(f)} | {risk[0]:.6f} | {risk[1]:.6f} | {fmt(x["mean"])} {band(x["sim95"])} | {verdicts[x["verdict"]]} | {x["wins"]}/12 |')
    r+=['','각 cell에서 12개 catalog 모두를 포함했다. 표의 마지막 열은 한 catalog 안의 두 block을 평균했을 때의 부호다. Catalog 하나를 새로 뽑았을 때의 변동은 cell 평균의 구간보다 크다. [사전 예측 원문](PREDICTIONS_KO.md) · [계획과 SHA-256](freeze39.sha256) · [모든 catalog 추정치](stats39.json)','',
        '| 조건 | 새 catalog 두 block 평균의 95% prediction interval¹ | 보정 이동의 제곱 평균 | 기존 잔차와의 정렬 항 |',
        '|---|---:|---:|---:|']
    for k in c:
        x=s['by_cell'][k];m,u=x['squared_change_and_alignment']
        r.append(f'| {cname(k)} | {band(x["predictive95"])} | {m:.6f} | {fmt(u)} |')
    r+=['','¹ 동일한 생성 분포에서 catalog 하나를 다시 뽑아 두 block으로 측정하는 경우의 정규 근사다. 물리 법칙이 바뀌는 실물 표면에 대한 구간은 아니다. 정렬 항은 2 E[(q_optimizer−q_fixed)(Y−q_fixed)]다. Δ는 이동의 제곱 평균에서 정렬 항을 뺀 값이다. 큰 확률 수정도 기존 잔차와 방향이 맞지 않으면 손해다. 이 항등식은 사후 분해용이며 정책에는 Y가 입력되지 않는다.','',
        f'네 수치 예측의 MAE는 {s["point_mae"]:.6f}이다. 모든 조건에 v38 전체 평균 {fmt(p["mean_prior"])}를 넣은 단순 예측은 {s["baseline_mae"]:.6f}, 차이가 없다고 한 예측은 {s["zero_mae"]:.6f}다. Spearman 순위 상관은 {s["rank"]:.2f}였다. 형상·law를 아는 실험자가 조건별 평균을 예측하는 문제에 대한 네 점의 기술 통계다.','',
        '## 전체 성능 및 손실의 단위','',
        f'전체 fixed Brier는 **{s["mean_brier"][0]:.6f}**, optimizer는 **{s["mean_brier"][1]:.6f}**였다. 네 cell을 균등 가중한 Δ는 **{fmt(s["delta"]["mean"])} {band(s["delta"]["ci"])}**다. 각 cell 안에서 catalog를 최상위 독립 단위로 사용하고 Welch–Satterthwaite 95% interval을 계산했다. 실용적 우위의 사전 기준은 Δ의 상한이 −0.003보다 작은 경우다.','',
        f'실제 표면의 방향성이 query 부근에서도 유지되는 strong에서 Δ는 {fmt(s["by_regime"]["strong"]["mean"])}, 약해지는 weak에서는 {fmt(s["by_regime"]["weak"]["mean"])}, 사라지는 local null에서는 {fmt(s["by_regime"]["null"]["mean"])}였다. 각 regime는 같은 latent draw를 공유한다.','',
        f'Ridge와 shelf의 평균 Δ 차이(ridge−shelf)는 {fmt(s["secondary_shape_diff"]["value"])} {band(s["secondary_shape_diff"]["ci"])}였다. Law에 따른 형상 대조의 변화, 즉 (ridge strengthening−weakening)−(shelf strengthening−weakening)은 {fmt(s["secondary_interaction"]["value"])} {band(s["secondary_interaction"]["ci"])}였다. 두 값은 보조 pointwise 95% 추정치다.','',
        '## GPU p90 및 시간 측정 범위','',
        '| 실제 경로 | 1,024개 물체 GPU p90 | 95% paired-cluster bootstrap 구간 |','|---|---:|---:|']
    for i,k in enumerate(('Fixed 32.5%','Frozen continuous optimizer')):
        lo,hi=a['p90_bounds'][i];r.append(f'| {k} | {a["p90"][i]:,.1f} ms | [{lo:,.1f}, {hi:,.1f}] ms |')
    lo,hi=a['ratio_bounds']
    r+=['',f'Optimizer/fixed p90 비율은 **{a["ratio"]:.5f} [{lo:.5f}, {hi:.5f}]**였다. 사전에 정한 +2% 허용 상한은 **{"통과했다" if a["passes_two_percent"] else "확인되지 않았다"}**. NVIDIA RTX PRO 6000 Blackwell Server Edition 한 장에서 24개 timing cluster × 3회씩, 두 방법을 무작위 순서로 실행했다. Common acquisition, selected-sensor decoding, source gain 최적화 또는 고정 보정 선택, host 반환을 포함한다. 실제 접촉·reset과 one-time bank 생성은 제외했다. 각 timing 호출의 예측값도 저장된 추론 결과와 대조했다. [Timing·통계](stats39.json) · [측정 코드](run39.py)']
    (ROOT/'TABLES_KO.md').write_text('\n'.join(r)+'\n')

if __name__=='__main__':
    main()
