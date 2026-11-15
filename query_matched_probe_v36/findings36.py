"""Render target estimates and distinguish statistical precision from practical gain."""
from report36 import R,data,f,d,band,db,percent
from pathlib import Path
def word(lo,hi):
    if hi<0:return 'Brier가 낮아졌다'
    if lo>0:return 'Brier가 높아졌다'
    return '개선 여부를 확정하기 어려웠다'
def phrase(v,j):
    return d(v['delta'][j])+' '+band(v['ci_simultaneous'][j])
def main():
    a=data('stats36.json');g=a['scopes']; local=g['local_equal']; ix=data('increment36.json')['scopes']['local_equal']
    sx=data('supplement36.json')['scopes']['local_equal']
    A=a['auc']['strong_vs_low']
    B=a['balanced_accuracy_strong_vs_weak_null']
    restricted=data('increment36.json')['paired_regime_assignment']
    threshold=local['ci_simultaneous'][2]
    if threshold[1]<-.003: margin='이 차이는 사전에 정한 실용 기준 0.003도 넘었다.'
    elif threshold[0]>-.003: margin='이 interval은 사전에 정한 0.003 이상의 실용적 Brier 감소를 지지하지 않는다.'
    else: margin='사전에 정한 0.003 이상의 실용적 Brier 감소 여부는 미확정이다.'
    p=local['loss']; ce=local['ci_simultaneous']
    soft=a['shrinkage']['local_equal']
    verdict=f'''**네 번째 접촉을 query 근처로 옮겼을 때, 표면의 방향 효과를 판독하는 AUROC는 {A["auc"][0]:.3f}에서 {A["auc"][1]:.3f}로 변했다** (strong 대 weak/null, 네 측정의 paired 95% interval {A["near_ci95"][0]:.3f}~{A["near_ci95"][1]:.3f}). 실제 보정을 켜는 binary gate를 같은 재료 구분 기준에 대입했을 때 balanced accuracy는 {percent(B["value"])}였다.

**네 번째 record를 보정 판단에 추가한 paired Brier 차이는 {phrase(local,2)}**였다. {margin} 세 local regime을 균등 가중했을 때 네 측정 gate의 held-out Brier는 {f(p[5])}, conventional fixed 4에서 원래 physics decoder는 {f(p[0])}, v35 correction은 {f(p[1])}였다. 이 두 end-to-end 차이에는 네 번째 동작과 방향 reader 변화도 들어간다.

Weak 표면의 Gate4−v35는 {phrase(g["weak"],1)}, query 근처에서 방향성이 사라진 local null은 {phrase(g["null"],1)}, 표면 전체에 방향성이 없는 global null은 {phrase(g["global_null"],1)}였다. Strong 표면의 Gate4−v35는 {phrase(g["strong"],1)}였다. Local null과 global null에서 원래 physics와의 차이는 각각 {phrase(g['null'],0)}, {phrase(g['global_null'],0)}로 두 interval 모두 0을 포함했다. 이 실험은 v35의 geometry·response family에서 새로 생성한 synthetic worlds를 사용했다.'''
    (R/'VERDICT_KO.md').write_text(verdict+'\n')
    cond=f'''## 논문 기여 가능성

**예측 성능과 실험 설계의 증거는 크기가 다르다.** 세 local regime에서 첫 세 접촉과 네 번째 conventional 접촉까지 완전히 같은 응답을 갖는 paired worlds를 구현했다. Query 부근 측정을 넣으면 strong과 weak/null의 posterior 순위가 달라진다. 동일 matched decoder에서 네 번째 near 측정과 네 번째 conventional 측정으로 만든 gate의 차이는 {d(sx["contrast"][2])} {band(sx["simultaneous_ci95"][2])}였다. 네 번째 factual feature는 양쪽에 주고 paid velocity만 추가한 차이는 {d(ix["means"][0])} {band(ix["pointwise95"][0])}였다. 둘은 서로 다른 비교다. 전자는 동작을 바꾼 가치, 후자는 이미 얻은 geometry 위에서 velocity가 제공한 가치를 묻는다.

같은 environment에서 세 local regime의 gate 선택만 바꿔 배정한 대조에서 실제 선택의 Δ는 {d(restricted["delta"])} {band(restricted["pointwise95"])}였다. {100*restricted["identical_decision_fraction"]:.1f}%의 짝에서는 세 표면 모두에 같은 결정을 내렸다. 이는 geometry 차이 또는 평균 보정률을 바꿔 얻은 이득과 분리되는 비교다.

Gate는 세 측정에서 네 측정으로 갈 때 전체 local 환경의 보정 적용률을 {percent(local['extras'][0])}에서 {percent(local['extras'][1])}로 높였다. Gate4는 strong {percent(g['strong']['extras'][1])}, weak {percent(g['weak']['extras'][1])}, local null {percent(g['null']['extras'][1])}, global null {percent(g['global_null']['extras'][1])}에서 보정을 켰다. 강한 환경을 더 자주 보정해 v35의 약·무방향 손해를 피하면서, strong에서 v35보다 덜 보정한 비용도 냈다. 이 비율을 보면 배포 환경에서 무방향 재질을 자동 검출하는 용도로는 아직 부족하다.

**단순한 대안과의 차이도 중요하다.** 같은 signed proposal에 source 평균 multiplier 0.325를 일률 적용한 방법의 Brier는 {f(soft["brier"][0])}이고 binary gate와의 차이(Gate4−고정 shrinkage)는 {d(soft["contrasts"][0])} {band(soft["ci"][0])}였다. 표면별로 기대되는 gain을 추정하는 모델을 기여로 내세우려면, 같은 budget의 단순 shrinkage보다 유리한 조건도 제시해야 한다. 약하거나 방향이 없는 표면에서 보정을 줄여 얻은 이득 전체를 새 probe의 효과로 산정하면 과장된다.

**현재 근거로 작성할 수 있는 결과**는 네 번이라는 작은 budget에서 서로 구분되지 않는 calibration histories를 구성하고, query 부근의 물리적 measurement로 얻는 표면 정보와 그 정보가 줄인 decision loss를 따로 정량화한 simulation 연구다. Contact geometry와 외형, 원래 decoder, source prior가 정해진 환경에서 얻은 결과다. GPU p90은 한 장치·한 session이고, 32개 독립 block은 고정한 4개 object–surface 조합 내부의 반복이다. 현재 단계에서 실제 새 재질, 다른 query 방향, physical probing을 포함한 로봇 cycle time으로 일반화할 근거는 부족하다. Source에서는 global-null gain을 directional deployment sensor로 근사했으며 correction 방향은 세 번의 far 접촉으로 고정했다.

**후속 검증은 물리 조건 하나를 먼저 바꾸는 편이 좋다.** 비용을 네 번으로 유지하고 같은 query 높이에서 (i) 지금 속도, (ii) 미끄러짐이 생기기 쉬운 속도, (iii) 기존 속도를 독립 배정한다. Noise-free identifiability 표의 후보 속도는 다음 실험 설계에만 쓴다. 다음 held-out panel에서 실제 GPU inference 및 Brier를 다시 측정하고, 강한 continuous directional physics posterior와 source-only shrinkage를 함께 비교한다. 실물에서는 동일한 표면 패치에서 probe와 별도 물체의 충돌을 짝지어 기록하고, 각 실험의 object ID와 surface session을 나누어야 한다. 높이·속도를 옮긴 비용, sensor saturation, 마모가 측정에 미친 영향도 같은 session에서 확인해야 한다.'''
    (R/'CONTRIBUTION_KO.md').write_text(cond+'\n')
if __name__=='__main__':main()
