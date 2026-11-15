## 어느 paid prefix에서 다음 probe를 살 가치가 있었는가?

서로 다른 두 cohorts에서 각각 네 object–surface 조합마다 네 독립 blocks를 생성했다. 각 block의 여덟 환경에 대해 동일한 물체·noise를 사용해 friction orientation을 양쪽으로 설정했다. 합계는 독립 blocks 32개, 서로 다른 calibration pools 256개, orientation을 포함한 환경 512개, 2회와 4회를 모두 센 paid prefixes 1,024개다. 각 prefix에 독립적인 두 futures와 두 deployment batches를 교차해 총 4,096개의 Brier gain을 계산했다. 각 cohort에는 반대 orientation에서도 같은 난수로 생성한 32,768개 deployment 물체가 있다. 두 stage, 방향, sensor noise, 또는 같은 행·열의 gain cells로 독립 block 수를 늘리지 않았다.

이미 지불한 접촉 다음에 probe를 추가해도 128개 deployment 물체 중 probe를 받는 물체는 정확히 13개다. 양의 gain은 13곳을 새로 골랐을 때 평균 Brier가 줄었다는 뜻이다. 두 stage에서 효과는 추가 접촉을 2회 또는 4회 구매한 전체 가치다. 4→8회는 실제 첫 stage의 survivor로 제한하기 전 모든 환경에서 측정했다.

사전에 정한 규칙은 같은 block의 friction coupling posterior SD가 상위 절반인 환경이었다. 별도로 24개 규칙 중 **plus orientation, 2→4회, 관측된 tangential velocity spread 하위 절반** 조건이 탐색용 cohort A에서 선택됐다. A에서 이 조건의 실제 평균 gain은 +0.00186, 반대 절반과의 차이는 +0.00236였다. 이를 파일로 고정한 뒤 독립 cohort B를 생성했다. B에서는 추가 선택이나 threshold 변경을 하지 않았다. [사전 계획](PREFIX_PLAN_KO.md) · [선택된 규칙과 A의 모든 후보](results_prefix/discovery_lock.json)

| Cohort B의 prefix | 평균 gain [simultaneous 95%] | 반대 절반 대비 gain 차이 [simultaneous 95%] | 두 하한이 모두 0 초과 |
|---|---:|---:|---:|
| Friction coupling posterior SD 상위 절반 | +0.00032 [-0.00019, +0.00082] | +0.00021 [-0.00050, +0.00093] | 아니오 |
| A에서 선택한 조건 | -0.00022 [-0.00167, +0.00124] | -0.00040 [-0.00200, +0.00120] | 아니오 |

두 조건 모두에서 절대 gain과 반대 절반 대비 gain 차이의 하한이 동시에 0을 넘지 않았다. 주어진 관측 특징만으로 추가 probe를 우선할 집단을 독립적으로 확인하지 못했다.

두 하한이 모두 0.003을 넘은 조건은 0개였다. 같은 observed prefix의 realized gain은 미래 응답과 deployment worlds에 따라 변한다. 따라서 실험 단위는 개별 접촉이나 전체 gain cell 수가 아닌 16개 독립 confirmation blocks다.

### 이미 얻은 접촉 정보와 orientation

| Cohort B stage | Friction orientation | 높은 posterior SD의 gain | 낮은 posterior SD의 gain | 차이 |
|---|---|---:|---:|---:|
| 2→4 | plus | +0.00023 | -0.00026 | +0.00050 |
| 2→4 | minus | -0.00010 | +0.00038 | -0.00048 |
| 4→8 | plus | +0.00138 | -0.00039 | +0.00176 |
| 4→8 | minus | -0.00025 | +0.00067 | -0.00092 |

Stage와 orientation별 추정치는 고정 규칙을 구분해 보고한 보조 결과다. 해당 네 행에서 효과가 큰 행을 새 가설의 독립 검증으로 채택하지 않았다.

| A에서 선택한 규칙의 독립 측정 | 대상 gain | 반대 절반 대비 차이 |
|---|---:|---:|
| Cohort B, future 1 × batch 1 | +0.00051 | +0.00029 |
| Cohort B, future 2 × batch 2 | -0.00073 | -0.00133 |

선택된 규칙에서 추가 probe로 query view와 probe 사이의 선택이 바뀐 deployment 물체의 비율은 cohort B 대상 절반에서 10.01%, 나머지에서 9.67%였다. 이때 변화율은 128개 전체를 분모로 한다. 정책이 계속 같은 관측을 고르면 동일한 common decoder의 그 물체에 대한 gain은 0이다. 집단 간 gain 차이는 바뀐 물체 수와 그 물체에서 고른 sensor의 예측 성능 양쪽에 의존한다.

Cohort B에서 네 endpoints에 부여한 simultaneous 95% interval은 각각의 object–surface 안에서 양 방향과 여덟 환경을 하나의 묶음으로 재표집했다. B에서 처음까지 모두 16개 independent blocks를 사용했다. 32회의 sensor noise는 같은 물체를 반복 측정한 것이다. 같은 prefix에서 얻은 두 diagonal 측정도 prefix를 공유하므로 독립된 두 연구의 효과처럼 해석할 수 없다. 이 실험은 surface orientation을 사전에 지정한 합성 조건에서의 probe 효과를 측정한다. 개별 특징과 gain의 관계를 실제 장비에 옮기려면 해당 특징과 양 방향의 접촉을 함께 기록해야 한다. [Cohort B의 모든 추정치](results_prefix/prefix_gain33.json) · [사전 설계대로 gain을 측정한 코드](prefix33.py) · [독립 검산과 cluster bootstrap](prefix33_stats.py)

### 같은 집단 안에서 관측된 gain의 분포

아래 수치는 Cohort B에서 각 prefix마다 2×2 cells를 평균한 실제 관측 gain의 분포다. 각 환경의 conditional expected gain을 정확히 알 때의 양수 비율은 이 설계로 직접 추정할 수 없다.

| A에서 선택한 조건 | 양의 2×2 gain 관측 | Median [Q1, Q3] |
|---|---:|---:|
| 대상 절반 | 35/64 (54.7%) | +0.00075 [-0.00388, +0.00372] |
| 나머지 절반 | 32/64 (50.0%) | -0.00014 [-0.00310, +0.00248] |

### 한 환경에서 적용할 수 있는 고정 threshold

다른 환경의 순위가 없어도 같은 특징을 사용할 수 있는지 별도로 확인했다. Cohort A에서 선택된 특징의 median은 0.0873699였고, 이 수치를 고정해 B의 대상 71개 환경과 반대쪽 57개 환경을 나눴다.

두 관측값에 ddof=0을 적용한 spread는 두 tangential velocity 차이의 절댓값을 2로 나눈 값이다. 따라서 이 threshold는 두 값의 차이가 0.17474보다 작은 prefix를 가리킨다.

대상 gain은 -0.00051 [-0.00170, +0.00078], 반대 그룹 대비 gain 차이는 -0.00149 [-0.00313, +0.00029]였다. 네 조합에 동일한 가중치를 주었다. 유효한 bootstrap draws는 10,000/10,000회였다. 괄호는 탐색적 pointwise 95% interval이다. 단일 prefix용 threshold에 대한 이 보조 결과는 위 primary simultaneous 판정과 구분한다.
