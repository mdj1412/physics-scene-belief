# v39 · 새 형상에서 언제 continuous gain optimizer를 신뢰할 수 있는가

**같은 네 번의 접촉을 사용했을 때, v38에서 동결한 optimizer는 새 `ridge` 물체에서 두 contact law 모두 고정 32.5% shrinkage를 이겼다. `shelf`에서 고정값이 이길 조건은 사전에 식별하지 못했다.** 두 형상을 합치면 held-out Brier는 0.113638에서 0.113252로 줄었다. 감소량 0.000386은 사전 실용 기준 0.003의 약 13%다. 같은 1,024개 물체의 GPU p90은 1,418.4 ms에서 1,420.8 ms로 약 2.4 ms 증가했다.

### 무엇을 언제 예측했는가

한 환경에서 calibration용 물체를 골라 총 네 번 접촉한 뒤, 같은 환경에 놓인 **별도 물체**가 주어진 충돌에서 미끄러지지 않고 붙는 사건(`stick`)의 확률을 예측한다. 고정 정책은 첫 세 접촉에서 얻은 signed cue를 항상 32.5%만 반영한다. Optimizer는 네 번째 outgoing velocity로 source posterior를 갱신하고, v38의 source gain table에서 예상 Brier가 가장 낮은 보정 비율을 0~100% 사이에서 고른다. 두 방법은 같은 23 mm·+1.10의 마지막 접촉, 같은 25 mm·+0.425의 query, 같은 K256 decoder 및 deployment observation을 받았다. 따라서 두 loss의 차이는 같은 관측을 어떻게 보정에 사용하는지에 대한 추정치다. 네 번째 접촉 자체의 인과적 가치는 이 비교에서 측정하지 않았다. [보정식과 정보 범위](POLICY_KO.md)

v39에서 바꾼 것은 **catalog와 contour**다. 두 기존 contact law를 유지한 채, 마지막 probe와 query 높이에서 local slope가 과거 물체와 가까운 새 shelf·ridge를 설계했다. v38에서 가장 가까운 형상의 조건별 Brier 차이를 v39의 사전 예측으로 삼았다. 정책·예측·형상식·판정 기준의 hash를 **target 생성 전**에 서버에 기록했다. 형상과 law는 이 예측을 작성한 실험자가 아는 조건이며, optimizer에는 주지 않았다. [예측표](PREDICTIONS_KO.md) · [형상 방정식](PHYSICS_KO.md) · [사전 계획](PLAN_KO.md)

각 shape × law에 독립 catalog 12개, catalog당 두 block을 사용했다. Calibration과 deployment의 prototype support를 분리했다. Deployment draw 98,304개가 92,544개 서로 다른 prototype을 사용했다. 각 draw에서 두 표면 방향, 세 가지 query 부근 방향성 강도, 32회 sensor noise를 대응시켜 두 정책을 비교했다. 이 반복을 독립 표본 수에 더하지 않았다. 독립 CPU 구현에서 저장된 query label 불일치는 0건이었고, 별도로 재계산한 Brier의 최대 절대 오차는 5.56×10⁻¹⁷ 미만이었다. [물리 검산](PHYSICS_AUDIT_KO.md) · [Brier 검산](audit39.json)

![사전 예측과 실측 효과. 마름모가 예측, 원이 실제 평균, 굵은 선이 동시 구간, 회색 점 하나가 독립 catalog 하나다.](v39.png)
