# v48 · weak preload의 물리 설계와 mechanical state 판별력

약한 힘을 한 번 가한 상태에서 nominal contact site 양옆의 displacement 차이를 읽으면, 새 네 family에서 mechanical coupling을 구별하는 AUROC가 **0.9391**이었다. Source에서 선택된 31 mm fixed probe는 **0.8527**이었다. Strong 상태의 변형 방향을 맞히는 비율은 각각 **99.47%**, **89.51%**였다. 이 수치는 별도의 source-calibrated classifier에서 얻은 sensor diagnostic이다. 실제 projection router의 Brier는 같은 물체에서 별도로 채점한다. [Sensor 결과](sensor_identification48.json) · [Threshold와 측정 단위](diagnostic_thresholds48.json)

## 1. 새 family와 물리 개입

Source는 v47 keel·crown·channel·bead, 새 family는 **ogee, cove, reed, flute**다. 새 family는 19·31 mm에서 공통의 smooth unloaded site를 갖고 바깥 contour가 달라진다. 같은 site의 ±2.5 mm 안에서 네 family의 base profile과 derivative 차이는 모두 0이다. 각 높이의 seven-point profile에서 가장 비슷한 family 쌍도 적어도 한 channel에서 0.330 mm만큼 다르다. Under-load bending 기울기는 각각 cubic, cosine, quartic, oscillatory 함수로 설정했다. 네 family의 nominal full-load 기울기 진폭은 각각 0.135·0.158·0.120·0.143이다. 이는 서로 다른 조건을 만드는 **prescribed deformation**이며 실제 재료상수를 회귀해 얻은 constitutive law는 없다. [World 구현](world48.py) · [Geometry audit](pre_physics48.json)

Optical sign·alignment와 관계없이 mechanical state를 0, −1, +1로 교차했다. 분석 시 무변형 1/2, 두 변형 방향 각 1/4로 가중한다. 두 강한 부호의 실제 impact label이 달라지는 비율은 독립 12개 catalog 평균 34.27% [95% CI 33.00–35.55]였다. Null을 기준으로 음·양 signed state가 label을 바꾸는 비율은 각각 11.21%와 23.06%였다. Stick은 finite-pressure impact 후 residual tangential speed ≤0.020 m/s로 정의했다. Primary 884,736개 label slot에 exact null 중복 조건이 포함된다. 이 중복만 합치면 663,552개다. Catalog 0·6에 quadrature를 12×6에서 24×12로 높인 147,456개 조건에서 label 차이가 없었고 저장한 speed 차이는 최대 3.17×10⁻⁷ m/s였다. 실물 slip threshold와 contact solver validity는 여전히 별도 검증 대상이다. [Physics 채점](mechanics48.py) · [Label 검산](outcomes_physics48.json)

## 2. 한 번의 실제 paid observation으로 무엇을 근사했나

Nominal query 높이 또는 fixed station에서 0.40 N weak preload를 주고, nominal site 양옆 ±1.5 mm에 놓인 두 fiducial의 displacement를 한 snapshot에서 읽는다. Unloaded frame을 기준으로 두 displacement 간 차이 하나를 사용한다. Weak-force deformation은 2.0 N reference-load shape 변화의 20%인 선형 구간으로 설정했다. 2.0 N은 형상 변화를 선형 정규화하는 기준이며 dynamic impact의 peak normal force를 직접 계측한 값이 아니다. 이 때문에 true state에 번호를 붙인 값을 직접 sensor에 넣지 않고, loaded profile의 두 지점에서 계산한 차이가 sensor에 들어간다. 새 family의 양의 무잡음 differential은 0.06841–0.08580 mm다. 반대 부호에서는 방향이 바뀌고 null에서는 0이다. Object gain U(0.90,1.10)과 catalog gain U(0.93,1.07)을 곱하고 catalog bias U(−0.012,+0.012) mm, object bias N(0,0.007²) mm, draw noise N(0,0.028²) mm를 더했다. 한 물체의 네 noise draw는 네 개의 별도 trial로 평균했다. 한 번의 deployment decision에 네 측정치를 결합하지 않는다. [Sensor generator](sensor48.py)

두 station의 reduced-state model은 query site 상태가 s일 때 다른 site도 60% 확률로 s를 갖고, 나머지 40%에서는 같은 사전분포(0.5,0.25,0.25)에서 독립 추출하도록 설계했다. 이에 따라 off-site sign 정보가 일부 남는다. 두 site를 바꿔도 stationary joint distribution이 같다. 서로 다른 query height 각각에서 local state를 강제로 고정한 표본은 이 모델의 **conditional Monte Carlo**다. 각 물체에 대해 모든 intervention에서 영구 고정된 실물 재질 지도를 관찰한 것이 아니다. 같은 위치에 같은 동작을 실행하는 두 정책은 같은 측정값과 noise draw를 받는다. 이 공간적 상관 및 약한 힘과 강한 힘의 방향 일치는 설계 가정이다.

## 3. 변형을 실제로 구별했나

V47의 source stimulus로 각 정책의 nominal query height별 ternary classification threshold 둘을 고정했다. Source에서 state prior로 가중한 정확도를 최대화하는 lower·upper cutoff를 사전 격자(0–0.18 mm, 0.002 mm 간격)에서 택했다. Threshold와 weak measurement만으로 새 target을 분류했으며, 아래 수치는 12개 catalog에 대한 개별 paired-unit 95% t CI다. AUROC는 절대 displacement 크기로 inert와 strong을, active direction은 source 기준점을 중심으로 signed displacement 부호로 음·양을 가린다. Ternary balanced accuracy에서 세 상태는 동일 가중한다. [검산 구현](sensor_audit48.py)

| 측정 위치 | Coupling AUROC | 강한 상태의 sign 정확도 | Ternary balanced accuracy |
|---|---:|---:|---:|
| Query site | 0.9391 [0.9318, 0.9465] | 99.47% [99.37, 99.57] | 85.76% [84.62, 86.90] |
| Fixed 31 mm | 0.8527 [0.8452, 0.8602] | 89.51% [89.01, 90.01] | 75.27% [74.02, 76.53] |

| 실제 state | Local classifier에서 예측한 0 / − / + | Fixed 31 mm에서 예측한 0 / − / + |
|---|---:|---:|
| 0 | 90.93 / 3.24 / 5.83% | 84.04 / 7.37 / 8.59% |
| − | 20.03 / 79.96 / 0.003% | 25.65 / 69.66 / 4.69% |
| + | 13.62 / 0.001 / 86.38% | 23.30 / 4.58 / 72.12% |

이 비교에서 local gauge는 실제 coupling이 강할 때 sign을 뒤집는 오류보다 **변형이 없다고 판단하는 오류**를 더 자주 냈다. 별도 민감도 분석에서 두 station의 state 전달확률을 100%로 바꾸고 같은 noise·threshold를 유지하면 fixed31의 coupling AUROC는 0.9426, sign 정확도는 99.54%였다. Query-site 판별이 우세하다는 결과에는 **local mechanical heterogeneity**가 필요하다. 이 값은 target sensor generator의 통제 실험에서 나왔고 실물 surface 사이의 전이율은 아직 측정하지 않았다. [Full confusion과 sensitivity](sensor_identification48.json)

## 4. Sensing과 reset 비용

| 기본 actuator 단계 | 가정한 설정 | 물체당 시간 |
|---|---|---:|
| Approach | 2 mm를 20 mm/s로 이동 | 100 ms |
| Load ramp | 0.40 N까지 4 N/s로 가압 | 100 ms |
| Sensing dwell | 한 번의 동시 differential read | 50 ms |
| Unload | 8 N/s로 힘 해제 | 50 ms |
| Retract + settle | 2 mm를 20 mm/s로 복귀한 뒤 40 ms 대기 | 140 ms |

추가 action의 직접 비용은 weak preload와 fixed probe 모두 **one touch + one reset, sensing 250 ms + reset 190 ms = 440 ms/object**다. Always-original은 추가 0회다. 두 paid 정책은 query sensor가 이미 구매되지 않은 물체에도 budget을 동일하게 지급한다. 한 장비에서 512개 물체를 직렬 처리하는 단순 시나리오에서는 추가 시간 225.28 s이며, 이 수치는 GPU p90과 다른 단위다. 19·31 mm의 midpoint에서 출발·복귀하고 lateral speed를 60 mm/s로 두면 두 정책 모두 왕복 12 mm에 200 ms를 더해 640 ms다. 출발·복귀점이 실제 query 높이면 local 440 ms, fixed31은 반수의 물체에서 왕복 24 mm가 추가되어 평균 640 ms다. 이 예시는 acceleration, 영상 재동기화, 실물 슬립으로 인한 object 재배치, 힘 제어 실패·마모를 고려하지 않는다. 실제 장비가 없어 force·속도·시간과 재설정 성공률은 측정값이 아니다.

Gauge가 읽은 signed displacement는 두 fiducial의 차이다. Force를 가한 지점의 absolute normal indentation과 force–displacement curve가 따로 없으므로, 여기에서 실제 mechanical work나 소모 전력을 추정하지 않았다.
