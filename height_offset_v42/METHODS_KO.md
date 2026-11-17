## 같은 물체에 같은 관측을 주고, 실제 접촉점만 바꾸는 설계

로봇이 모서리를 밀기 전 카메라로 본 외곽은 같더라도, 팁이 목표 지점보다 조금 위에서 닿으면 충돌 시점의 기울기가 달라진다. v41에서는 base·hollow·bulge가 명목 접촉점에서 같은 위치와 같은 local tangent를 갖게 설계했다. v42에서는 외곽과 센서를 그대로 둔 채 실제 접촉점만 옮겼다. 첫 질문은 높이 오차가 생긴 순간 주변 외곽이 물리적으로 중요해지는지, 둘째 질문은 그 외곽을 보는 것이 실제 확률 예측을 개선하는지다.

| 항목 | 구현과 범위 |
|---|---|
| 새 catalog | v41과 독립인 12개 panel에 두 shape family(flare·roll), 명목 높이 19·31 mm, 세 contour(base·hollow·bulge), 두 contact law(saturating·crossover)를 교차했다. 형상·법칙의 생성식은 이전 연구와 같다. 24개 cell × 12 panel=288개 archive다. 각 panel에서 calibration 시작 방향을 번갈아 배치했다. |
| Signed offset | δ=실제 높이−명목 높이. 물리적 query만 0, ±0.125, ±0.25, ±0.5, ±1 mm로 옮겼다. 한 번의 실험에서 환경의 배포 물체가 같은 δ를 공유한다. 위치 센서가 그 오차를 알려 주지 않는 조건이다. |
| 물리 | Query 충돌 속도는 +0.425다. 위치가 바뀌면 해당 높이의 local tangent로 상대 속도를 분해하고 static-friction impulse 한계에서 stick/slip을 판정한다. 표면 방향 2개와 near-field 마찰 방향성 strong·weak·null을 모두 평가한다. 외곽 변형이 없는 base에서도 높이에 따른 기울기와 마찰 변화가 있어 event 자체가 달라질 수 있다. |
| 입력과 구매 비용 | 환경당 네 번의 calibration(마지막 +1.10, 23 mm). 8개 환경 각각 128개 배포 물체에 frozen K256이 query view 115개와 probe 13개를 배정했다. 8개의 sensor noise draw는 평가 반복이다. 한 결정을 내릴 때는 128개 물체에서 구매한 해당 회차 관측만 쓴다. |
| Geometry | 명목 접촉점의 ±6 mm를 2 mm 간격으로 읽은 7점 profile, noise σ=0.30 mm. Estimated는 별도 같은 물체의 두 contour 상태를 물체 32개에서 읽어 paired difference의 첫 principal component를 구한다. 동일 panel·shape·명목 높이에 한 방향을 재사용한다. 방향의 amplitude·stick label을 쓰지 않지만 좌표 등록, pair 대응, 같은 변형을 공유하는 group과 sensor scale은 필요하다. |
| Decoder | Original은 7개 좌표를 모두 쓴다. Estimated와 oracle은 각각 추정한 방향과 simulator의 실제 변형 방향을 whitened likelihood에서 제거한다. 세 방법의 probe likelihood는 동일하다. Bank는 16,384개 source hypothesis와 동일한 환경별 두 prior alignment parameter를 사용한다. Acquisition은 이전 source 기준을 유지한다. 주 비교에서 observation template과 후보 stick event는 모두 알려진 명목 높이에 맞춘다. |
| Selector | 같은 base 확률의 log-odds에 βzc를 더한다. Signed cue z, β와 optimizer의 보정량 w는 기존 설정을 쓰고 c∈{0, 0.325, w}에서 고른다. Learned selector는 v41의 label로 후보별 Brier 차이에 필요한 residual만 회귀한다. 한 배치의 물체 128개에 같은 c를 적용한다. 실제 offset·law 이름·결과는 주지 않는다. |

**Source에서 이미 드러난 불확실성.** V41로 학습한 세 정책 selector는 occupancy feature와 ridge penalty 0.01을 선택했다. 과거 panel 또는 명목 높이를 통째로 가렸을 때 estimated selector−0%는 −0.000370, −0.000210이었고, shape family를 가리면 +0.002378이었다. 이 세 검증을 동일 가중하면 상수 0%가 우세했다. 따라서 v42는 **source에서 가장 좋은 비상수 후보**와 **상수를 허용한 보수적 source 선택**을 모두 새 catalog에 적용한 실험이다. 훈련한 자료에는 이번의 hidden offset label이 없었다. [선택식·학습 분할](SELECTOR_KO.md) · [동결 계수](selector_model42.json)

**어떤 차이를 ‘손해’로 판정했는가.** 각 물체의 stick 확률 p와 실제 event Y에 대해 (p−Y)²를 계산한다. 0%에서 D(δ)=Brier(estimated)−Brier(original)로 두고, 양의 동시 95% 구간을 projection 손해로 판정한다. 9개 pooled offset과 36개 shape×height×offset 대조를 각각 별도로 보정한다. Selector의 주 대조는 0을 제외한 8개 signed offset의 균등 평균이다. 모든 paired interval의 독립 단위는 12개 panel이다. 대조를 이번 장비의 위치 오차 분포나 연속적인 임계 높이로 외삽하지 않는다. [사전 계획](PLAN_KO.md) · [분석 코드](stats42.py)
