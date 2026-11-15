# 네 번째 물리 관측의 Bayes Brier gain

확률을 추정할 사건은 calibration 물체와 분리해 추출한 deployment 물체의 query 충돌에서 stick이 일어나는가이다. Geometry family, contact law family 및 population parameter α를 고정한 cell 안에서 평가지표를 정의한다. 해당 cell과 유한 catalog를 아는 evaluator라도 실제로 뽑힌 prototype index, 표면 방향, near-field 강도, 정답은 관측할 수 없다. 실제 정책은 기존 source model만 사용한다.

### 비교할 두 예측

세 번의 보정 접촉에서 관측한 각각의 outgoing velocity와 미리 확보한 factual rows를 C라고 하자. 네 번째 물체의 factual row도 C에 들어간다. 이 물체에 +1.10의 접촉을 가한 뒤 읽는 noisy outgoing velocity를 V라고 한다. U는 평가할 **한 물체**의 factual row, 세 번의 접촉으로 정해진 sensor 종류, 그 sensor에서 읽은 값이다. 셀에 조건부로 다음의 두 확률을 계산한다.

- p3 = P(stick | C,U)
- p4 = P(stick | C,U,V)

다른 deployment 물체의 유료 sensor 값은 공유하지 않는다. Calibration 물체 선정은 prototype index와 독립인 pool row 추출이다. 배포할 때의 sensor 배정은 C 및 모든 deployment factual rows로 결정된다. 고정된 cell과 α 아래에서 선택에 동원된 다른 물체의 factual row 분포는 orientation이나 local regime에 무관하므로, 해당 사실치들을 더 조건화해도 이 물체의 두 확률은 그대로다. 다른 cell인지까지 추론하려는 문제에서는 이 단순화가 성립하지 않을 수 있다.

### 이산 합산

숨은 환경 상태 h는 두 방향 × 세 강도의 여섯 후보이며 균등 사전확률을 쓴다. Calibration 및 deployment prototype은 분리된 2,048개씩의 catalog에서 개별적으로 복원추출했다. 각 센서에 대해 알려진 diagonal Gaussian likelihood를 사용한다.

1. Calibration 물체마다 각 h에서 factual vector와 같은 물체의 paid velocity에 대한 **joint likelihood**를 2,048개 latent state에 걸쳐 평균한다. 물체별 평균 확률을 곱하면 첫 세 접촉에 대한 likelihood를 얻는다. 네 번째 factual row만의 likelihood는 h마다 같아 p3 계산에서 약분된다. p4는 네 번째 factual·velocity 쌍의 joint likelihood를 곱한다.
2. Deployment 물체도 각 h에서 그 물체의 factual과 선택된 sensor 값을 **같은 prototype 위에서 결합**한 뒤 catalog를 적분한다. 후보 h에 대한 likelihood와 그 후보 내 stick probability를 함께 계산한다. Probe형 sensor는 orientation 정보도 주므로 이 likelihood로 calibration posterior를 다시 갱신한다.
3. 여섯 환경 상태를 posterior로 가중해 p3와 p4를 얻는다. Query-view likelihood가 방향과 강도에 같더라도 해당 deployment 물체의 latent state와 stick probability를 추정하는 데는 쓰인다. 대수 계산은 float64이며 추가 Monte Carlo bank는 없다.

이 합산은 사전에 정의된 finite population에 대해서 정확하다. Catalog를 아는 evaluator의 결과를 continuous unknown-law population에서 얻을 수 있는 보편적인 최고 개선량으로 해석할 수는 없다. 유한 catalog에서 몇 개 후보만 남아 지나치게 쉬운 문제가 되었는지 살피기 위해 paid sensor별 posterior prototype effective support도 저장한다. 이 값은 cell·환경 상태별로 계산한 뒤 현재 posterior mass로 평균한 값이다.

### 분모와 회수량

조건부 평균의 직교성 때문에 두 확률에 대해 E[(Y−p3)²]−E[(Y−p4)²] = E[(p4−p3)²]가 성립한다. 왼쪽에는 독립 held-out binary label이 필요하고 오른쪽은 예측 확률만으로 평가할 수 있다. 같은 여섯 상태의 선언된 prior를 적용한 평균에서 이 관계를 확인한다. 임의로 한 강도만 골라 계산한 기대값에 같은 등식을 요구하지 않는다. 맨 오른쪽은 각 물체에서 정확하게 계산해도 전체 population의 평균은 유한한 실험 block에서 추정한다.

각 정책 j의 관측 전후 예측을 q3,j와 q4,j라 두면 주 회수량은 held-out label로 계산한 평균 Brier3−Brier4다. 두 예측을 p4와 비교한 평균 (q3,j−p4)²−(q4,j−p4)²도 같은 정책 이득의 Rao–Blackwellized estimate다. 조건부 평균에서 공통으로 사라지는 Bernoulli variance를 이용한다. 이 수치는 real-world label 대체가 아니며 지정한 true-law model과 prior가 맞는 benchmark에서만 의미가 있다. 같은 정보로 출발하더라도 정책의 세 관측 예측은 Bayes-optimal이 아닐 수 있어, 비율이 100%를 넘거나 음수가 될 수 있다. 고정 32.5%는 배포 sensor와 signed cue가 같고 네 번째 관측을 보정에 반영하지 않으므로 이 비교에서 회수량은 정확히 0이다.

### 검산과 저장 구조

CPU smoke에서 별도 seed로 만든 Q와 P 사례에서 직접 Gaussian 차이를 계산한 NumPy 합산과 torch 합산은 절대오차 1e-10 이내에서 일치했다. 서버 실행기는 각 cell의 첫 block에서도 양쪽 sensor와 모든 target 상태에서 독립 NumPy 합산을 비교하고, 원본 policy prediction을 복원해 archive Brier를 확인한다. `oracle_results/<cell>/<block>/oracle_summary.npz`의 앞 세 축은 orientation, regime, environment다. `gain`은 Bayes gain, `oracle_risk`는 두 prefix의 실제 Brier, `risk`는 prefix × policy의 실제 Brier, `rb_gain`은 policy별 Rao–Blackwellized gain, `ess`는 Q와 P에서의 posterior prototype effective support다. `posterior.npz`가 저장된 실행에서는 그 안에 두 확률의 object × repeated-noise 결과도 있다. 저장용 float32 변환 전에 모든 요약 수치를 float64로 계산한다. Reference 계산시간은 배포 GPU p90에서 제외한다.

### Cell을 숨기는 sensitivity

보조 계산은 네 cell에 균등 사전확률을 주고 α만 알려 준다. 실행 전에 정한 네 cell의 첫 block, 모든 8개 환경, 두 방향과 세 강도, 첫 deployment noise 반복을 비교한다. 각 후보 cell은 factual row 분포 자체가 다르므로, 공개된 calibration factual pool 1,024개와 deployment factual row 128개의 likelihood를 모두 곱한다. 이 likelihood에는 focal factual row도 이미 들어 있다. 따라서 유료 센서를 반영할 때에는 해당 factual row에 조건부인 paid likelihood를 사용해 factual evidence를 두 번 세지 않는다. 계산 시작 시 생성한 full calibration factual pool의 선택된 네 행이 저장된 관측값과 일치함을 확인한다. 나중에 팔별로 산출된 가족별 posterior mass는 실제 family ground truth와 비교하는 진단값이지만, 그 ground truth를 후보 가중치에 주지 않는다.

`marginal38.py`는 4 × 2 × 3개의 숨은 환경 상태를 합산한다. 모든 family를 똑같이 만든 독립 CPU smoke에서 family별 posterior가 정확히 1/4로 남고 known-family 예측과 일치하는지 확인했다. Primary의 예측값 archive가 존재하면 같은 첫 반복에서 known-family 재계산과 저장된 예측도 비교한다. 이 sensitivity는 작은 paired subset이며, 새 material population에서 family를 알아내는 일반적인 성능을 뜻하지 않는다. 유한 catalog가 서로 다른 cell에 고정됐기 때문에 1,152개 factual row가 catalog의 표본 특성으로 cell을 알아맞힐 여지도 있다.

### 독립 calibration stress test

기본 panel에서 p3−Y 평균은 −0.00286이었다. 사전 분석과 같은 percentile bootstrap 구간은 0을 제외했다. 셀과 첫 방향으로 나뉜 8개 층에 독립 block이 4개씩이므로, 추가 Welch–Satterthwaite 근사를 사용하면 95% 구간은 [−0.00582, +0.00010]이었다. 반면 p4−p3 평균은 +0.00166이며 이 층화 t 구간도 [+0.00030, +0.00302]였다. 독립 catalog의 모든 label을 열거한 전체 기대 발생률에 비해 실제 panel에서 사건이 0.00217 더 많았다. 같은 catalog 기대값 대비 p3와 p4 평균 차이는 −0.00069, +0.00097이었다. 이 결과만으로 generator 오류나 유한 표본 변동을 단정하기 어려워 fresh-seed 진단을 추가했다. [층화 검산](calibration_audit38.json)

서버 CPU에서 32,768개 새 환경을 독립 생성했다. 각 환경에서 4개 calibration 물체와 별도 deployment 물체 하나, 무작위 표면 방향·강도 및 새로운 Gaussian noise를 뽑았다. Deployment sensor는 다른 환경과 독립적으로 13/128 확률로 contact probe를 배정했다. 원 실험에서 factual evidence를 이용해 13개를 배정하는 방식과는 다르다. 독립 NumPy likelihood 합산으로 네 개 사전 지정 진단에 Bonferroni 동시 95% 구간을 적용했다. p3−Y는 +0.00117 [−0.00335, +0.00570], p4−Y는 +0.00141 [−0.00310, +0.00592], p4−p3는 +0.00024 [−0.00020, +0.00067], 실제 Brier 감소와 이상 이득 사이의 차이는 −0.00034 [−0.00074, +0.00005]였다. 마지막 항목의 개별 95% 구간은 0을 제외했다. 네 개 동시 구간에서는 체계적 편차가 검출되지 않았으나, 이 진단이 원 panel의 selection이나 calibration 잔차를 없애지는 않는다. [새 표본 및 모든 구간](stress_oracle38.json) · [독립 계산](stress_oracle38.py)
