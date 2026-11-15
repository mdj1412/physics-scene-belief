# 한 catalog의 독립 block에서 우열이 반복되는 정도

아래는 원래의 조건별 사전 예측을 보완하는 사후 진단이다. Cell별로 12개 catalog가 있고 각 catalog의 두 block은 같은 prototype 분포, 다른 물체 draw·sensor noise·QMC bank seed를 쓴다. 첫 probe의 방향도 두 block에서 다르다. 따라서 두 block의 차이에는 표본 변동과 시작 방향 차이가 함께 포함된다.

| Cell | Source가 예측한 optimizer의 이득 | 실제 held-out 이득 | 두 독립 block에서 이득의 부호가 같은 catalog |
|---|---:|---:|---:|
| Shelf / weakening | +0.001382 | −0.000033 | 5/12 |
| Shelf / strengthening | +0.001439 | +0.000181 | 3/12 |
| Ridge / weakening | +0.000988 | +0.000478 | 9/12 |
| Ridge / strengthening | +0.001168 | +0.000920 | 10/12 |

여기서 이득은 fixed Brier − optimizer Brier로 양수가 개선이다. Source 순위는 동일 관측에서 frozen gain table의 최적 w와 고정 w=0.325의 expected gain 차이로 재계산했다. Optimizer의 최적화 후보에 고정값이 포함돼 있으므로 이 값은 설계상 항상 0 이상이다. v38의 별도 조건별 정량 forecast는 이 source gain 자체와 구분한다. Target에서 가장 작은 이득을 보인 shelf 두 cell에서 source gain은 오히려 더 컸다. 이 차이는 source에서 학습한 기대값을 target 성능 보증으로 사용할 수 없는 사례다. [독립 재계산](audit39.json)

두 block의 부호가 일치한 경우는 총 27/48개 catalog(56.25%)였다. Cell 안에서 catalog를 재표집한 서술적 95% 구간은 43.75–68.75%였다. 각 cell·시작 방향 평균을 제거한 뒤 두 block의 Δ 사이 Pearson correlation은 0.044였다. 이 결과는 한 cell에서의 평균 정책 순위가 새로운 미니배치 한 번에서도 항상 그대로 나타나지는 않는다는 점을 정량화한다. 부호 일치율은 각 cell의 우세 경향에도 의존한다. 50%를 모든 cell에 공통인 무정보 기준으로 해석하거나, 두 block으로 catalog 자체의 물성 효과가 0이라고 판정할 수 없다. [모든 catalog의 paired block 변화](repeatability39.json)
