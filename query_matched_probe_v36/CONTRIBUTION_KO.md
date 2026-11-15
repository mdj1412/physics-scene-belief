## 논문 기여 가능성

**예측 성능과 실험 설계의 증거는 크기가 다르다.** 세 local regime에서 첫 세 접촉과 네 번째 conventional 접촉까지 완전히 같은 응답을 갖는 paired worlds를 구현했다. Query 부근 측정을 넣으면 strong과 weak/null의 posterior 순위가 달라진다. 동일 matched decoder에서 네 번째 near 측정과 네 번째 conventional 측정으로 만든 gate의 차이는 -0.00112 [-0.00168, -0.00057]였다. 네 번째 factual feature는 양쪽에 주고 paid velocity만 추가한 차이는 -0.00125 [-0.00170, -0.00079]였다. 둘은 서로 다른 비교다. 전자는 동작을 바꾼 가치, 후자는 이미 얻은 geometry 위에서 velocity가 제공한 가치를 묻는다.

같은 environment에서 세 local regime의 gate 선택만 바꿔 배정한 대조에서 실제 선택의 Δ는 -0.00018 [-0.00027, -0.00009]였다. 85.7%의 짝에서는 세 표면 모두에 같은 결정을 내렸다. 이는 geometry 차이 또는 평균 보정률을 바꿔 얻은 이득과 분리되는 비교다.

Gate는 세 측정에서 네 측정으로 갈 때 전체 local 환경의 보정 적용률을 6.8%에서 35.1%로 높였다. Gate4는 strong 37.3%, weak 35.0%, local null 33.0%, global null 47.7%에서 보정을 켰다. 강한 환경을 더 자주 보정해 v35의 약·무방향 손해를 피하면서, strong에서 v35보다 덜 보정한 비용도 냈다. 이 비율을 보면 배포 환경에서 무방향 재질을 자동 검출하는 용도로는 아직 부족하다.

**단순한 대안과의 차이도 중요하다.** 같은 signed proposal에 source 평균 multiplier 0.325를 일률 적용한 방법의 Brier는 0.11127이고 binary gate와의 차이(Gate4−고정 shrinkage)는 +0.00037 [-0.00007, +0.00082]였다. 표면별로 기대되는 gain을 추정하는 모델을 기여로 내세우려면, 같은 budget의 단순 shrinkage보다 유리한 조건도 제시해야 한다. 약하거나 방향이 없는 표면에서 보정을 줄여 얻은 이득 전체를 새 probe의 효과로 산정하면 과장된다.

**현재 근거로 작성할 수 있는 결과**는 네 번이라는 작은 budget에서 서로 구분되지 않는 calibration histories를 구성하고, query 부근의 물리적 measurement로 얻는 표면 정보와 그 정보가 줄인 decision loss를 따로 정량화한 simulation 연구다. Contact geometry와 외형, 원래 decoder, source prior가 정해진 환경에서 얻은 결과다. GPU p90은 한 장치·한 session이고, 32개 독립 block은 고정한 4개 object–surface 조합 내부의 반복이다. 현재 단계에서 실제 새 재질, 다른 query 방향, physical probing을 포함한 로봇 cycle time으로 일반화할 근거는 부족하다. Source에서는 global-null gain을 directional deployment sensor로 근사했으며 correction 방향은 세 번의 far 접촉으로 고정했다.

**후속 검증은 물리 조건 하나를 먼저 바꾸는 편이 좋다.** 비용을 네 번으로 유지하고 같은 query 높이에서 (i) 지금 속도, (ii) 미끄러짐이 생기기 쉬운 속도, (iii) 기존 속도를 독립 배정한다. Noise-free identifiability 표의 후보 속도는 다음 실험 설계에만 쓴다. 다음 held-out panel에서 실제 GPU inference 및 Brier를 다시 측정하고, 강한 continuous directional physics posterior와 source-only shrinkage를 함께 비교한다. 실물에서는 동일한 표면 패치에서 probe와 별도 물체의 충돌을 짝지어 기록하고, 각 실험의 object ID와 surface session을 나누어야 한다. 높이·속도를 옮긴 비용, sensor saturation, 마모가 측정에 미친 영향도 같은 session에서 확인해야 한다.
