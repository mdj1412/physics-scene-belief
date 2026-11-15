# 네 번째 관측 후 어느 환경에서 optimizer가 유리했나

예측식은 v38에서만 학습한 z, w 기반 ridge regression이며 v39의 test label로 보정하지 않았다. 원자료의 z와 w만 읽어 전 환경의 점수를 먼저 저장한 후 두 방법의 실제 paired Brier 차이를 읽었다. 이 순서는 동일한 실행으로 얻은 sensor record를 사용한 blind replay다. 새로운 물체의 후속 충돌 정답은 보조 정책의 입력에 없다. [사전 예측](PREDICTIONS_KO.md) · [예측 결과와 hash](route39.json)

Optimizer 사용을 예측한 비율은 52.8%였다. 이 집단의 실제 optimizer − fixed Brier는 -0.000567 [-0.000775, -0.000358]이고, fixed를 선택한 집단에서는 -0.000185 [-0.000434, +0.000073]였다. 두 집단 간 차이는 -0.000382 [-0.000731, -0.000034]였다. Interval은 4개 고정 cell 내부에서 catalog를 재표집한 pointwise 95% 구간이다. 이 비교는 보정 선택에 사용할 수 있는 정보의 정렬 능력을 평가하며 두 집단 간 causal effect를 추정하지 않는다.

| 비교 | Brier 차이 | 동시 95% interval |
|---|---:|---|
| Routing − fixed | -0.000299 | [-0.000427, -0.000172] |
| Routing − optimizer | +0.000087 | [-0.000048, +0.000222] |

| 형상과 law | Optimizer 선택률 | 예측 win 집단 실제 Δ | 예측 loss 집단 실제 Δ | Routing − fixed | Routing − optimizer |
|---|---:|---:|---:|---:|---:|
| shelf_weakening | 23.5% | +0.001317 | -0.000361 | +0.000310 | +0.000276 |
| shelf_strengthening | 22.4% | +0.000789 | -0.000461 | +0.000177 | +0.000358 |
| ridge_weakening | 78.0% | -0.000915 | +0.001074 | -0.000714 | -0.000236 |
| ridge_strengthening | 87.3% | -0.001111 | +0.000393 | -0.000970 | -0.000050 |

두 집단의 형상과 law 구성비가 달랐다. 두 shelf에서는 합산 결과와 이득의 순서가 반대이고 두 ridge에서는 같은 방향이었다. 결과 이후 이를 점검하기 위해 각 cell 안에서 먼저 구한 집단 차이를 각 25%로 표준화하면 -0.000141 [-0.000518, +0.000247]였다. 이 보조 분석은 모든 형상에 공통인 환경별 판별력을 확정하지 못한다.

환경별 Δ의 RMSE는 0.004476, 전체 descriptive Spearman correlation은 0.108이었다. 평가 단위는 4,608개 environment-regime-orientation 행이며 서로 노이즈와 물체를 공유하는 행들이 포함된다. 독립 단위는 48개 cell-specific catalog다. Router의 GPU p90은 별도로 측정하지 않았다.
