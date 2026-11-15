# v34 · 같은 접촉 횟수에서 한 방향과 양방향 probing 비교

**표면 방향을 읽을 수 있다는 것과 다음 probe를 유리하게 배분할 수 있다는 것은 구분해야 한다.** 단방향 paid prefix에도 friction 방향 정보가 있었다. 한 방향으로 2·4·8회 접촉했을 때 독립 표면 방향 분류의 Brier는 0.133·0.077·0.035였고, 양방향 교대는 0.123·0.072·0.025였다. Factual-only의 0.250은 양쪽 방향에서 같은 관측을 내는 설계와 일치한다. 2회와 8회에서 교대가 유리했지만, family tag 없이 단일 readout을 옮긴 보조 실험에서는 교대의 추가 이득 구간이 모두 0을 포함했다.

충돌의 stick 여부를 채점했을 때, **estimated-sign stopping head의 교대 − 단방향 Brier는 +0.00065 [-0.00068, +0.00198]**였다. 같은 gain 모델에 sign을 추가한 효과(estimated − neutral)는 단방향 -0.00009 [-0.00090, +0.00073], 양방향 +0.00006 [-0.00074, +0.00086]였다. 각각 우열 미확정, 우열 미확정에 해당한다. 아래에서 같은 정보가 최종 확률의 calibration에는 어떤 효과가 있었는지 분리해서 보여준다.

**가장 유망한 단서는 stopping이 아니라 terminal prediction이었다.** 결과를 본 뒤 시행한 anchored control에서 fixed 4의 원래 확률에 추정한 표면 방향만 반영하자 Brier가 단방향에서 0.00596, 양방향에서 0.00535 낮아졌다. 아래에 원 decoder와의 직접 대조, 별도 물체 자료의 이전 결과 및 사후 분석의 한계를 함께 제시한다.

![방향 판독과 결정 단계별 Brier 차이](v34.png)

왼쪽 그림의 정답은 표면에서 마찰이 강한 방향이고, 오른쪽 그림의 정답은 별도 충돌에서의 stick 여부다. 왼쪽의 더 낮은 loss가 오른쪽의 개선을 보장하지 않는다. 오른쪽은 estimated sign을 추가한 head에서 똑같은 입력 구조의 neutral head를 뺀 값이며, 음수가 개선이다.

## 전체 stopping policy: 동일 예산의 성능과 시간

| 정책 | 한 방향 Brier ↓ | 양방향 Brier ↓ | 양방향 − 한 방향 [simultaneous 95%] | 한 방향 GPU p90 ms | 양방향 GPU p90 ms |
|---|---|---|---|---|---|
| Frozen gain scorer | 0.125199 | 0.125245 | +0.00005 [-0.00099, +0.00108] | 1,575.3 | 1,579.5 |
| Fixed 4 | 0.125703 | 0.125602 | -0.00010 [-0.00083, +0.00062] | 1,498.0 | 1,524.0 |
| Uncertainty-only | 0.124808 | 0.125334 | +0.00053 [-0.00055, +0.00160] | 1,577.3 | 1,606.7 |
| Estimated-sign head | 0.125081 | 0.125733 | +0.00065 [-0.00068, +0.00198] | 1,502.3 | 1,505.4 |

모든 방법은 8개 환경에 **calibration probe 32회**를 쓴 뒤, 1,024개 새 물체에 **deployment probe 104회**를 썼다. Fixed 4는 환경당 4회, 나머지는 2회에서 시작해 네 곳을 4회로 늘리고 그중 두 곳만 8회까지 간다. Frozen scorer는 v32의 1×1 모델을 그대로 쓰고, 새 estimated-sign head는 양쪽 schedule의 gain 자료에 같은 방식으로 fit한 공통 head다. Frozen scorer는 일방향 좌표계에 맞춰 양의 명령에서 tangential velocity 부호를 반사했다. 사용자가 보는 표면 orientation을 정책에 알려 주지 않았다.

GPU p90은 NVIDIA RTX PRO 6000 Blackwell Server Edition에서 1,024개 물체 전체의 posterior update, 순차 stopping, acquisition 및 선택된 관측 decoding까지의 synchronized wall time이다. 각 정책·schedule당 48회, 독립 timing clusters는 8개다. 초기화·data generation·실제 접촉 및 reset 시간은 제외했다. 새 head는 별도 timing pass에서 fixed 4 및 uncertainty-only와 순서를 섞어 실측했다. 그 pass의 baseline p90(한 방향/양방향)은 fixed 4 1430.0/1432.4 ms, uncertainty-only 1500.4/1500.9 ms였다. [기본 정책의 Brier와 GPU timing](primary_stats34.json) · [새 head와 두 baseline의 GPU timing](head_timing34.json)

| 기본 정책의 동일 schedule 내 차이 | 한 방향 [simultaneous 95%] | 양방향 [simultaneous 95%] |
|---|---|---|
| Frozen − fixed 4 | -0.00050 [-0.00144, +0.00043] | -0.00036 [-0.00141, +0.00069] |
| Frozen − uncertainty-only | +0.00039 [-0.00015, +0.00094] | -0.00009 [-0.00075, +0.00057] |
| Uncertainty-only − fixed 4 | -0.00090 [-0.00180, +0.00001] | -0.00027 [-0.00127, +0.00074] |

위 표와 기본 정책 세 개의 schedule 차이에는 넓게 잡은 9개 대비의 simultaneous interval을 적용했다. GPU runner가 독립적으로 정한 7개 대비(frozen 중심)도 [별도 검산](primary_stats34.json)에 남겼다. 각 family에서 초기 방향별로 네 block씩 함께 재표집한다. 먼저 가는 방향의 유리함과 교대 여부의 효과가 섞이지 않게 했다. Practical margin은 사전에 정한 absolute Brier 0.003이다.

## 같은 접촉 예산에서 probe 방향을 바꾼 실험

한 환경의 factual pool에는 물체 1,024개가 있다. 미리 무작위로 정한 여덟 개 물체에서 순서대로 접촉한다. 한 방향 조건에서는 같은 부호의 tangential velocity로, 교대 조건에서는 접촉마다 부호를 반전하여 충돌시킨다. 두 조건 모두 incoming tangential speed의 절댓값이 1.10이다. 한 환경 안에서 매 접촉은 새 물체에 가한다. 최초 부호는 각 object–surface 조합에서 네 block은 음수, 네 block은 양수다. 따라서 2회 prefix에서 alternating은 양쪽 방향을 한 번씩 본다. 이후 deployment 물체에서 선택할 수 있는 probe는 v33과 같은 음수 방향으로 고정했다. 이 비교에서 바뀌는 것은 calibration에서 지불한 행동이다.

추론기는 각 접촉의 측정값과 실제 사용한 행동 부호를 함께 받아 Coulomb contact model로 likelihood를 계산한다. 두 방향 모두 같은 8,192개 Sobol particles를 사용한다. Frozen gain scorer에는 v32에서 훈련한 15개 feature를 준다. 이 중 접선 방향의 측정 통계량을 만들 때, 각 paid velocity를 원래 훈련의 음수 방향 좌표로 반사한다. 모델이 새로운 방향으로 밀었다는 이유만으로 1.10의 부호 차이를 friction signal로 읽지 않게 하기 위한 고정 전처리다. 평가는 simulator가 생성한 material orientation을 양쪽으로 뒤집어 반복했고, 이 orientation은 정책에 알려주지 않았다. Source model은 방향에 대칭인 Coulomb law이므로 실제 target material anisotropy의 크기와 부호를 명시적으로 적합하지 않는다. Population coupling을 알고 있는 common decoder에 K256/M2048 acquisition이 선택한 query view 또는 probe를 넣어 물체별 stick probability를 채점한다. 독립적인 NumPy 계산으로 양쪽 행동의 likelihood 및 운동량 변화를 다시 계산했다. [설계](policy_design34.json) · [추론기](detector34.py) · [독립 검산](action_audit34.json)

각 block에는 coupling이 다른 여덟 환경이 있다. Frozen scorer와 uncertainty-only는 여덟 곳에 두 번씩 접촉한 뒤 네 곳에 두 번씩 더 주고, 그중 두 곳에 다시 네 번씩 배정한다. Fixed 4는 여덟 곳 모두 네 번씩 배정한다. 세 정책의 calibration probe는 각 32회다. 이후 각 환경의 독립 물체 128개 중 13개에 deployment probe를 배정하므로 총 104회다. Held-out latent deployment objects는 32,768개이고, 32개 독립 block을 두 material orientations와 두 calibration schedules에 대응시켜 실행했다. Brier를 계산할 때 동일 물체의 32 sensor noise draws를 평균했다.

Full GPU p90은 이미 도착한 factual readings 및 각 정책에서 구매 가능한 calibration readings를 사용해 stopping allocation, posterior, deployment observation selection, 선택한 observation의 decoding을 모두 다시 실행한 시간이다. 물체 1,024개짜리 block call의 시간을 측정했다. 공통 detector·decoder 초기화, 세계 생성 및 실제 sensor/reset 대기는 제외한다. 같은 block 안에서 schedule × policy 여섯 조합의 실행 순서를 매번 무작위화하고 warm-up 뒤 각각 세 번 측정했다. 물체·표면 조합별 최초 block 두 개의 두 material orientation을 모두 사용하므로 policy·schedule당 48회 실측, 독립 timing cluster는 8개다. [GPU 평가 코드](run34.py) · [통계 및 원확률 검산](stats_policy34.py)

## Paid prefix에 surface orientation이 담겼는가?

같은 물체를 양쪽 surface orientation에 놓고 trajectory와 sensor noise를 짝지었다. 마찰 방향을 분류하는 정답은 surface orientation이고, deployment에서 최종적으로 예측하는 정답은 별도 collision의 stick 여부다. 각 조합의 classifier는 평가할 물체와 표면을 모두 교체한 한 개 조합으로 학습했다. 예를 들어 rounded–fine에서 쓸 classifier는 ridged–coarse만 보고 학습했다. 각 policy가 배포될 평가 조합에 맞는 classifier를 미리 선택하는 조건이다. 어느 쪽 sign인지는 classifier에 제공하지 않았다.

학습에는 조합마다 독립 blocks 64개, 평가에는 48개를 사용했다. 각 block은 8개 coupling 환경을 포함한다. 전체 test는 독립 blocks 192개, 독립 calibration pools 1,536개다. 두 signs, 두 schedules, 2·4·8회 stages는 해당 pools에서 여러 번 측정한 값이다. 처음 진행 방향을 양쪽으로 균형 배정해 출발 방향과 물체 구성이 특정 orientation을 알려 주지 않게 했다. [계획과 seed 범위](DIAG_PLAN_KO.md) · [분리된 classifier와 학습 지문](diag_models.json)

| Paid probes | 한 방향 orientation Brier | 교대 orientation Brier | 교대 − 한 방향 [simultaneous 95%] | 한 방향 accuracy | 교대 accuracy |
|---|---|---|---|---|---|
| 2 | 0.13273 | 0.12283 | -0.00990 [-0.01973, -0.00007] | 80.8% | 80.6% |
| 4 | 0.07750 | 0.07178 | -0.00572 [-0.01450, +0.00306] | 89.5% | 89.6% |
| 8 | 0.03511 | 0.02535 | -0.00976 [-0.01507, -0.00445] | 95.9% | 96.9% |

2회만 한 방향으로 접촉해도 새 물체·표면에서 surface orientation을 80.8% 구별했다. 교대로 접촉하면 2회와 8회에서 확률의 Brier가 낮아졌다. 2회의 accuracy는 80.6%여서 같은 변화가 없었다. 4회의 interval에는 0이 포함됐다. 2회 차이의 upper bound는 −0.00007로 0에 가깝다. 각 stage마다 ± orientation을 동일 비중으로 계산했다. 95% interval은 고정된 네 family와 학습된 readout에 조건부이며 조합 안의 paired blocks를 16,000회 재표집했다.

Raw readout은 접촉 전후에 측정한 velocity로부터 normal·tangential impulse 비율을 근사하고, 이를 실제로 준 action 방향과 결합한 features를 쓴다. 주어진 incoming 방향, paid observations와 factual contour만 사용한다. 방향이 반대인 두 환경의 factual prefix는 같은 값이어서 factual-only orientation Brier는 0.25000이었다. Calibration을 한 뒤에도 각각의 쌍에서 target orientation을 무작위로 뒤집어 다시 학습·평가하면 0.254~0.258이었다. [입력과 proxy 식](diag34_model.py) · [여섯 조건의 원결과](results_diag/raw_analysis.json)

### 기존 scorer에서 방향을 되읽을 수 있는가?

똑같은 training·test histories에 source posterior likelihood를 다시 계산하고, 각각의 15 features와 첫 1×1 frozen gain score를 저장했다. 각 조합별 학습·test 각 8 blocks, test 전체 32 blocks를 썼다. 다음 값은 모두 orientation Brier다. Linear와 RBF 모두 start 방향을 제공하고 같은 logistic penalty를 썼다. Raw도 이 표에는 동일한 소규모 train split에서 다시 fit한 값을 사용했다.

| Schedule | Paid probes | Raw | 15 features · linear | Gain scalar · linear | 15 features · RBF | Gain scalar · RBF |
|---|---|---|---|---|---|---|
| 한 방향 | 2 | 0.14775 | 0.20608 | 0.25447 | 0.24643 | 0.25710 |
| 한 방향 | 4 | 0.10017 | 0.17686 | 0.21109 | 0.19795 | 0.20505 |
| 교대 | 2 | 0.13595 | 0.27412 | 0.25191 | 0.38833 | 0.26257 |
| 교대 | 4 | 0.09481 | 0.27363 | 0.25104 | 0.39789 | 0.25904 |

Alternating에서 15 features나 gain score만으로 새 family의 방향을 분류한 Brier는 0.25 부근 또는 그보다 높았다. 방향별 마찰 반응 차이를 알려면 어느 방향에서 어느 반응이 관측됐는지 알아야 한다. 현재 15 features는 motion 방향을 공통 좌표로 변환한 뒤 velocity의 평균·분산 등으로 요약한다. 한 방향 조건에서는 같은 scalar나 feature에서도 일부 orientation 정보를 되읽을 수 있었다. 이 비교가 정량화하는 것은 두 지정된 readout에서의 OOD decodability다. Frozen scorer는 원래 gain을 학습했으므로 orientation 분류가 나쁘다는 사실만으로 더 좋은 stopping rule의 존재가 입증되지는 않는다. [16개 비교의 simultaneous interval](results_diag/small_analysis.json)

### 양쪽 readout의 차원을 하나로 맞추면?

큰 raw feature set의 효과를 본 뒤, 다시 생성한 제3 test cohort에서 두 입력을 모두 scalar 하나로 제한했다. Physics scalar는 각 접촉의 추정 friction이 ± orientation에서 나올 확률비를 nuisance friction 후보 네 개에 대해 근사하고 이력에 걸쳐 더한다. Frozen scalar는 기존 모델이 출력한 expected gain이다. 각각 같은 13개 predictor terms와 intercept의 logistic readout을 같은 source folds로 학습했다. 32개 fresh paired blocks로 평가했다. Physics scalar는 simulator friction 분포의 근사치를 사전 지식으로 사용한다. 두 scalar의 정보 차이와 여기에 투입한 domain knowledge가 함께 반영된다. [실험 시점과 계획](DIAG_SCALAR_PLAN_KO.md) · [독립 test 실행 코드](diag34_scalar.py)

| Schedule | Paid probes | Physics scalar | Frozen gain scalar | Frozen − physics [simultaneous 95%] |
|---|---|---|---|---|
| 한 방향 | 2 | 0.13080 | 0.26521 | +0.13441 [+0.11618, +0.15264] |
| 한 방향 | 4 | 0.07032 | 0.21074 | +0.14043 [+0.12473, +0.15612] |
| 교대 | 2 | 0.11922 | 0.26442 | +0.14520 [+0.12876, +0.16164] |
| 교대 | 4 | 0.06924 | 0.26798 | +0.19875 [+0.18149, +0.21600] |

추가 접촉의 가치를 판단할 때에는 같은 예산에서 실제 observation이 달라지고 최종 stick Brier가 줄어드는지도 함께 보아야 한다. 여기서 채택한 family별 readout은 같은 geometry·surface를 직접 fit하지 않고 반대 family에서 학습했다. 다른 표면 전체나 실제 물체에 대한 orientation accuracy에는 별도 검증이 필요하다. [동일 차원 보조 분석 원자료](results_diag/equal_scalar_analysis.json)

### 물체·표면 이름 없이 같은 classifier를 쓰면?

앞의 진단을 본 뒤 family를 입력하지 않는 조건을 추가했다. 중간 geometry와 directional response를 가진 하나의 새로운 synthetic reference family에서만 classifier를 학습했다. 이 모델을 네 조합에 공통 적용하고, 이전 모든 panels와 seed가 겹치지 않는 64개 paired blocks에서 결과를 평가했다. 세 stage 모두 한 방향과 교대 양쪽에서 0.25보다 낮은 orientation Brier를 기록했다. 이 새 cohort에서 교대의 점추정은 모두 더 낮았지만 세 interval은 0을 포함했다. 어느 schedule에서든 방향을 읽을 수 있다는 결과가 유지됐고, 교대의 추가 효과 크기에는 source와 test 구성이 영향을 줄 수 있다. [추가 설계와 source 계수](DIAG_UNIVERSAL_PLAN_KO.md) · [보조 분석 전체 결과](results_diag/universal_analysis.json)

| Paid probes | 단일 source model 한 방향 | 단일 source model 교대 | 교대 − 한 방향 [simultaneous 95%] |
|---|---|---|---|
| 2 | 0.11404 | 0.10557 | -0.00847 [-0.02167, +0.00474] |
| 4 | 0.05784 | 0.05294 | -0.00489 [-0.01605, +0.00627] |
| 8 | 0.02224 | 0.01712 | -0.00512 [-0.01291, +0.00268] |

## 같은 정보가 어느 결정에서 쓸모가 있는가?

세 위치를 비교했다. **Stopping**은 32회 calibration budget을 나눈다. **Sensor selection**은 fixed 4를 유지한 채 각 환경의 128개 물체 중 probe를 줄 13개만 고른다. **Terminal prediction**은 fixed 4와 기존 sensor 선택까지 유지하고 이미 받은 관측에서 stick을 예측하는 마지막 확률만 바꾼다. 각 행의 neutral과 estimated/true는 같은 task, 같은 예산, 같은 train family와 feature 구조를 쓴다. 단계 간 절대 Brier를 직접 비교해 어느 모델이 더 우수하다고 해석하지 않는다.

| Schedule | 정보를 넣은 위치 | Estimated sign − neutral [95%] | True sign − neutral [95%] | 비교 대상 |
|---|---|---|---|---|
| 한 방향 | Stopping | -0.00009 [-0.00090, +0.00073] | +0.00003 [-0.00073, +0.00079] | neutral learner |
| 한 방향 | Sensor selection | -0.00009 [-0.00040, +0.00022] | +0.00003 [-0.00044, +0.00051] | neutral selector |
| 한 방향 | Terminal prediction | -0.00491 [-0.00607, -0.00374] | -0.00694 [-0.00754, -0.00634] | neutral decoder |
| 양방향 교대 | Stopping | +0.00006 [-0.00074, +0.00086] | -0.00004 [-0.00088, +0.00081] | neutral learner |
| 양방향 교대 | Sensor selection | -0.00051 [-0.00110, +0.00009] | -0.00066 [-0.00133, +0.00002] | neutral selector |
| 양방향 교대 | Terminal prediction | -0.00358 [-0.00516, -0.00199] | -0.00692 [-0.00753, -0.00631] | neutral decoder |

Estimated sign은 별도 calibration corpus에서 학습한 P(orientation=plus)다. True sign은 simulator의 정답을 준 privileged control이다. 매 block의 8개 환경이 같은 orientation을 공유하므로 sign을 모든 score에 상수로 더하면 stopping 순위가 바뀌지 않는다. 따라서 gain과 sensor selection 모두 cue×prefix/factual interaction을 포함한다. Terminal decoder는 원 logit·선택된 sensor와 sign 간 interaction으로 조정한다. 각 단계의 표는 서로 다른 확인 가설 family에 대해 보정한 구간이다. Stopping은 source와 held-out blocks를 함께 다시 뽑아 head까지 refit한 구간을 썼다. Selector/decoder는 학습된 모델에 조건부인 구간이며 물리 법칙 모집단 전체에 대한 구간이 아니다.

| Schedule | Stopping의 Brier 대비 | 조건부 구간 | Gain head까지 refit한 구간 |
|---|---|---|---|
| 한 방향 | Estimated − neutral | -0.00009 [-0.00060, +0.00043] | -0.00009 [-0.00090, +0.00073] |
| 한 방향 | True − neutral | +0.00003 [-0.00039, +0.00046] | +0.00003 [-0.00073, +0.00079] |
| 한 방향 | Estimated − placebo | -0.00024 [-0.00063, +0.00014] | -0.00024 [-0.00114, +0.00065] |
| 한 방향 | Estimated − fixed 4 | -0.00062 [-0.00161, +0.00037] | -0.00062 [-0.00192, +0.00068] |
| 한 방향 | Estimated − uncertainty-only | +0.00027 [-0.00050, +0.00104] | +0.00027 [-0.00079, +0.00133] |
| 양방향 교대 | Estimated − neutral | +0.00006 [-0.00040, +0.00053] | +0.00006 [-0.00074, +0.00086] |
| 양방향 교대 | True − neutral | -0.00004 [-0.00044, +0.00037] | -0.00004 [-0.00088, +0.00081] |
| 양방향 교대 | Estimated − placebo | +0.00019 [-0.00018, +0.00056] | +0.00019 [-0.00069, +0.00107] |
| 양방향 교대 | Estimated − fixed 4 | +0.00013 [-0.00070, +0.00096] | +0.00013 [-0.00110, +0.00136] |
| 양방향 교대 | Estimated − uncertainty-only | +0.00040 [-0.00048, +0.00127] | +0.00040 [-0.00077, +0.00156] |

Estimated와 neutral이 환경별로 다른 접촉 횟수를 배정한 비율은 한 방향 26.6%, 양방향 26.4%였다. Placebo는 paired +/− world의 cue를 block 단위로 뒤집어 일정 비율에서 실제 방향과의 대응을 끊는다. 같은 환경 여덟 곳을 독립적으로 뒤섞지 않았다. Placebo는 permutation distribution 전체가 아니라 사전 고정한 balanced permutation 한 번이므로 단독으로 정보 유무를 판정하지 않는다. [결정 모델 및 통계 계획](DECISION_PLAN_KO.md) · [실제 held-out Brier와 두 가지 구간](decision_stats34.json)

| Schedule | 원 fixed4/원 selector | Neutral selector | Estimated selector | True-sign selector | Placebo selector |
|---|---|---|---|---|---|
| 한 방향 | 0.125703 | 0.125859 | 0.125771 | 0.125893 | 0.125785 |
| 양방향 교대 | 0.125602 | 0.126562 | 0.126056 | 0.125905 | 0.126477 |

| Schedule | 원 fixed4/원 decoder | Neutral decoder | Estimated decoder | True-sign decoder |
|---|---|---|---|---|
| 한 방향 | 0.125703 | 0.135692 | 0.130786 | 0.128753 |
| 양방향 교대 | 0.125602 | 0.135624 | 0.132045 | 0.128706 |

Selector는 training family의 실제 potential loss 차이를 supervision으로 쓴다. Test 물체의 두 potential outcomes는 선택이 확정된 뒤에만 채점에 쓴다. Terminal 모델은 4회 prefix에서의 sign을 받고 동일하게 선택된 관측만 해석한다. 새로운 sensor나 label을 test time에 사지 않는다. 원래 고정 decoder는 참 population coupling을 아는 privileged reference이지만 orientation을 모른다. Neutral calibration과도 비교해야 단순한 재보정에서 오는 효과와 sign cue의 추가 효과를 분리할 수 있다. [Sensor selection 설계](SELECTOR_PLAN_KO.md) · [관측 정책별 결과](selector_stats34.json) · [Terminal calibration 결과](terminal_stats34.json)

### 원 physics decoder를 기준으로 해도 이득이 남는가?

처음 설계한 terminal logistic recalibration은 방향 추정치를 쓰면 neutral recalibration보다 낮은 loss를 냈지만, 둘 다 원래 physics decoder보다 나빴다. 전체 계수·온도까지 다시 fit한 비교만으로는 sign을 실제로 쓸 가치가 있다고 말하기 어렵다. 이 결과를 본 **후**, fixed 4와 기존 sensor 선택을 그대로 두고 원래 확률의 log-odds에 β·(2P(plus|prefix)−1)만 더하는 anchored control을 추가했다. β 한 개만 뺀 object·surface의 자료에서 학습했다. Cue가 0이면 원 확률과 완전히 같은 예측이다.

| 평가 자료 | 원 fixed 4 Brier | Estimated sign 보정 | True sign 보정(진단) | Estimated − 원 Brier [simultaneous 95%] |
|---|---:|---:|---:|---:|
| v34 단방향 | 0.12570 | 0.11974 | 0.11780 | −0.00596 [−0.00664, −0.00528] |
| v34 양방향 | 0.12560 | 0.12025 | 0.11779 | −0.00535 [−0.00605, −0.00464] |
| v33 별도 물체, 음의 방향만 | 0.12570 | 0.11882 | 0.11684 | −0.00688 [−0.00765, −0.00611] |

추가 probing이나 sensor 변경 없이 원래 decoder를 기준으로도 0.003보다 큰 개선이 나왔다. 실험자가 알려 준 true sign 없이 방향 추정치만으로도 이득이 있었다. 이 결과가 지지하는 후속 가설은 **다음 물체에 어떤 관측을 살지 바꾸기보다, 이미 산 관측을 해석하는 데 환경의 마찰 방향을 쓰자**는 것이다. 다만 primary 결과를 본 뒤 고른 모델이며 v33 archive의 기존 결과도 연구자가 본 적이 있다. 위 구간은 사후 분석 선택의 불확실성을 반영하지 않는다. 이 보정의 추가 GPU p90은 따로 실측하지 않았다. [보조 대조 계획과 식](ANCHORED_PLAN_KO.md) · [전체 수치와 β](anchored34.json) · [코드](anchored34.py)

### 사전 설계한 두 head를 다른 물체 집단에 그대로 적용한 결과

| Brier 대비 | V33의 별도 32개 paired blocks [simultaneous 95%] |
|---|---|
| Estimated stopping − neutral | +0.00010 [-0.00021, +0.00040] |
| True stopping − neutral | +0.00025 [-0.00019, +0.00069] |
| Estimated terminal − neutral | -0.00600 [-0.00727, -0.00473] |
| True terminal − neutral | -0.00811 [-0.00860, -0.00762] |

원 물체와 표면의 target family별로 source family 한 곳의 v34 자료만 사용한 뒤, 그 모델을 재학습 없이 v33의 별도 32,768개 물체에 적용했다. v33은 negative 방향만 probing했으므로 교대 효과를 검증하는 자료는 아니다. 이전 버전 연구자는 v33의 기본 정책 결과를 본 적이 있다. 새로운 v34 모델의 결과를 본 후 source family나 계수를 바꾸지 않았다. [외부 archive 적용 코드](external34.py) · [전이 결과](external34.json)

### Probe 시작 방향에 따른 상쇄 검사

| 정책 | 처음이 낮은 마찰 방향: 교대 − 단방향 | 처음이 높은 마찰 방향: 교대 − 단방향 | 사전 orientation을 알려 준 routing − 단방향 |
|---|---|---|---|
| Frozen gain scorer | -0.00068 [-0.00245, +0.00109] | +0.00078 [-0.00069, +0.00224] | -0.00005 [-0.00121, +0.00111] |
| Fixed 4 | -0.00008 [-0.00132, +0.00117] | -0.00013 [-0.00138, +0.00113] | -0.00008 [-0.00092, +0.00075] |
| Uncertainty-only | +0.00011 [-0.00175, +0.00197] | +0.00094 [-0.00059, +0.00247] | +0.00059 [-0.00046, +0.00163] |

마지막 열은 첫 운동이 표면의 어느 방향인지 별도로 안다고 가정한다. 어느 schedule이 유리한지는 training family에서만 고른 뒤 뺀 family에서 평가한다. 이 실험에서 측정한 조건부 관계가 실제 표면의 방향을 무료로 알 수 있음을 의미하지 않는다. [분석 계획](BALANCE_SENSITIVITY_KO.md) · [12개 대비](balance_stats34.json)

## 논문 기여 가능성과 후속 결정

**어디에서 차이가 생겼나.** 단방향에서 estimated cue를 넣었을 때 sensor selection의 Brier는 neutral 대비 -0.00009 [-0.00040, +0.00022], 이미 같은 관측을 선택한 뒤 terminal decoder의 Brier는 -0.00491 [-0.00607, -0.00374]였다. 양방향에서 estimated cue를 넣었을 때 sensor selection의 Brier는 neutral 대비 -0.00051 [-0.00110, +0.00009], 이미 같은 관측을 선택한 뒤 terminal decoder의 Brier는 -0.00358 [-0.00516, -0.00199]였다. V33의 다른 물체에 모델을 그대로 이전한 단방향 진단에서는 estimated cue의 stopping 효과가 +0.00010 [-0.00021, +0.00040], terminal 효과가 -0.00600 [-0.00727, -0.00473]였다.

**논문의 중심 주장.** 방향을 식별하는 문제와 그것을 downstream 위험을 낮추는 위치에 쓰는 문제를 분리한 점이 기여 후보다. 근거는 새 합성 물체에서 사전 고정한 intervention을 수행하고, 동결한 scorer의 정보 손실을 동일 용량 판독기로 확인한 뒤, 동일 probe quota에서 sign을 stopping·sensor selection·terminal decoding 세 위치에 따로 넣은 실험이다. 물체를 움직여 마찰을 알아내는 방법, 양방향 마찰 측정이나 decision utility를 쓰는 것 자체에는 선행연구가 있다. [실물에서의 방향별 friction](https://pmc.ncbi.nlm.nih.gov/articles/PMC5805965/) · [active pushing/pulling](https://arxiv.org/html/2411.09020v1) · [downstream decision utility](https://openreview.net/pdf?id=zBG7WogAvm)

**주장의 경계.** 표면 orientation은 합성 법칙의 binary sign이고, contact regime과 속도 크기는 고정했다. 동일 물체를 되돌려 측정한 실험이 아니며 서로 다른 calibration 물체의 friction과 restitution이 달라진다. 반복적인 reversal 그 자체가 도움이 된 것인지, 두 운동 방향을 한 번씩이라도 포함하면 충분한지는 `--++`와 `-+-+`를 따로 비교해야 한다. Gain head는 맞은편 family 하나의 독립 blocks 8개에서 학습하며, 한쪽에서 loss가 줄지 않은 것이 모든 scorer의 불가능성을 뜻하지 않는다. Family를 뺀 분석에는 fold별 classifier/head를 사용했다. Real deployment에서 family에 맞는 head를 공짜로 고를 수 있다는 주장은 하지 않는다. 단일 reader의 transfer 확인은 방향 판독에 관한 결과지 모든 downstream heads에 대한 실증이 아니다.

**다음 검증.** 먼저 한 표면의 날짜·마모·reset이 다른 session에서 독립 pulling test로 마찰 방향을 확정하고, calibration 물체와 deployment 물체를 나눈다. 각 block에서 처음 방향을 무작위화한 뒤 unilateral, alternating, 같은 방향별 접촉 횟수를 유지한 grouped order를 비교한다. 하나의 공유 모델을 보지 않은 물체와 surface session에 그대로 배포하고, 독립 sign Brier와 실제 stick Brier, motion/reset을 포함한 end-to-end p90을 따로 잰다. 새 방법의 채택 기준은 Brier 0.003과 같은 사전 기준 또는 접촉 시간을 반영한 비용으로 정해야 한다. Simulator에서 확인된 정보 판독만으로 실제 로봇의 비용 절감을 주장하지 않는다.

## 기존 연구는 접촉 방향을 바꾸면 무엇을 알 수 있다고 보는가?

표에서 **표면 방향**은 어느 쪽으로 미끄러질 때 friction이 더 큰가를 뜻하고, **probe 방향**은 측정자가 선택한 운동 방향을 뜻한다. 운동 방향을 알고 있으면 응답에서 표면 방향을 추정할 수 있다. 이후의 관측 선택에 이 추정이 필요한지는 결정 과제의 제약과 loss로 판단한다. 아래 여섯 원문은 이 연결의 각 부분을 다룬다. [Tramsen et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC5805965/) · [Huang et al.](https://openreview.net/pdf?id=zBG7WogAvm)

| 원문과 연구 질문 | 방법·표본 | 저자가 확인한 결과와 결론 | v34에서 적용할 수 있는 범위 |
|---|---|---|---|
| Tramsen et al. (2018). 어느 쪽 pulling에서 마찰력이 큰지 표면 형상만으로 정할 수 있는가? | 96개 sample–substrate 조합. Sawtooth를 따라 두 방향으로 당기고 네 normal loads에서 측정한 힘으로 friction coefficient를 추정했다. Stiffness, aspect ratio, roughness를 바꿨다. | 특정 조합에서 같은 비대칭 구조의 along·against 차이가 반전됐다. 재료와 표면 형상을 함께 측정해야 부호를 정할 수 있다. | 같은 조합의 양방향 직접 측정이 근거다. 서로 다른 물체에 한 번씩 가한 impact의 outgoing velocity에서 부호를 읽는 v34는 geometry·물체 간 편차도 추정해야 한다. [원문 Figs. 2–5](https://pmc.ncbi.nlm.nih.gov/articles/PMC5805965/) |
| Fazeli, Tedrake & Rodriguez (2017). Motion과 contact forces로 어떤 rigid-body parameters를 식별할 수 있는가? | Planar rigid contact의 complementarity 방정식을 유도하고 block과 ellipse의 drop simulation 및 유리판 사이 실제 drop을 평가했다. Simulation에는 4개 signal-to-noise ratios를 적용했다. | 알려진 외력이 없는 경우 관성량·접촉력 일부는 mass에 대한 비로 식별된다. 적절한 known perturbation 아래에서는 개별 값을 분리할 수 있다. 측정 noise가 커지자 회귀 오차도 증가했다. | 알려진 입력과 관측 방정식이 identifiability를 정한다는 근거다. 저자가 식별한 물리량과 v34의 signed friction law가 다르므로 양방향 두 번이면 항상 충분하다는 정리는 적용할 수 없다. [원문 §§4–5, Table 1](https://groups.csail.mit.edu/robotics-center/public_papers/Fazeli17.pdf) |
| Dutta et al. (IEEE T-RO, 2025). 어느 push·pull이 물성 추정을 개선하는가? | 실제 로봇에서 homogeneous·heterogeneous·articulated 각 40개, 총 120개 object configurations 및 생활 물체를 평가했다. Visuo-tactile filtering과 여러 단계의 information gain으로 action을 선택했다. | Active action selection은 여러 조건에서 필요한 학습 interaction을 줄였다. 특히 articulated object의 pushing에서 저자는 uniform action selection 대비 약 25% data efficiency 향상을 보고했다. | Push와 grasp 후 pull은 서로 다른 접촉 방식이다. 같은 속력의 부호만 바꾼 v34와 측정 조건이 다르고, 직접 지표도 parameter error·trajectory 수와 다른 물체의 Brier loss로 다르다. [원문 §IV, Figs. 7·10·15](https://arxiv.org/html/2411.09020v1) |
| Elazar et al. (TACL, 2021). Representation으로 속성을 해독할 수 있으면 원래 예측기가 그 속성을 쓰는가? | BERT에 대해 여섯 sequence-tagging tasks, English UD·OntoNotes에서 학습용 100,000 tokens를 표집했다. 선형 정보를 지우는 intervention을 같은 차원 수의 random removal과 비교했다. | 여섯 속성 모두 보조 분류기로 읽을 수 있었다. 속성을 지웠을 때 fine-grained POS에서는 word prediction accuracy가 크게 떨어졌지만 phrase boundary 두 과제에서는 오히려 소폭 올랐다. | Independent direction classifier와 frozen scorer의 역할을 구분하는 방법론적 근거다. Linear removal은 관련된 다른 정보도 지울 수 있어, 물리 모델에 옮길 때는 실제 결정의 변화까지 따로 대조해야 한다. [원문 §3–4, Table 1](https://aclanthology.org/2021.tacl-1.10.pdf) |
| Huang, Guo, Acerbi & Kaski (NeurIPS, 2024). 마지막 결정의 utility를 목표로 관측을 설계하면 유리한가? | Decision Utility Gain과 비근시적 학습을 사용했다. 한 실험에서는 4개 Gaussian processes로 가능한 4개 행동의 결과를 생성하고, 100개 test points에서 최적 행동을 골랐다. | 이 실험에서 학습한 정책은 uncertainty sampling과 decision-information baselines보다 높은 평균 의사결정 정확도를 보였다. 따라서 downstream 목적에 맞춰 관측을 설계하는 접근에는 직접적인 선례가 있다. | 합성 과제에는 target point와 utility가 주어진다. 같은 비용으로 여러 환경에 접촉을 나눠 주고, 새 물체에 대해 다시 sensor를 고르는 구조에서는 그 구조에 맞춘 평가가 필요하다. [원문 §6.2, Fig. 3](https://openreview.net/pdf?id=zBG7WogAvm) |
| Go, Qian & Yoon (2026, arXiv preprint). Parameter information을 최대화하는 측정점과 downstream cost를 최소화하는 측정점이 일치하는가? | Source localization·epidemic control·pharmacokinetic control의 세 수치 과제. 첫 과제에서 7×7 sensor grid를 조사하고 각 위치에 outer·inner samples를 각각 10,000개 썼다. | 2D source 전체의 EIG는 중심에 집중됐다. 목표인 수직 위치 오차는 더 넓은 수평 띠에서도 작았다. 설계를 달리해도 결정에 필요한 차원에서 비슷한 결과를 얻을 수 있었다. | Single-observation 및 지정된 objective에 대한 수치 결과다. V34에서 방향 정확도가 향상된 뒤 전체 Brier가 비슷하게 나와도 양립할 수 있는 사례를 제시한다. [원문 §4, Fig. 2](https://arxiv.org/html/2605.26093v1) |

### 연구 흐름과 증거의 강도

**측정에서 식별까지.** Tramsen et al.은 통제된 양방향 실물 측정으로 부호가 재료에 따라 달라짐을 보였다. Fazeli et al.은 known input과 motion이 어느 물성을 구별할 수 있는지 식으로 제한했다. Dutta et al.은 로봇이 구별력 있는 interaction을 선택하는 단계로 이 질문을 확장했다. 앞의 두 연구는 독립적으로 물리적 측정과 식별 가능성을 뒷받침한다. 두 번의 저비용 probe로 새 surface의 부호를 얼마나 정확히 알 수 있는지는 이 세 연구가 사용한 힘·모션 센서, 물체 수, 접촉 방식에 따라 달라진다. [Tramsen et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC5805965/) · [Fazeli et al.](https://groups.csail.mit.edu/robotics-center/public_papers/Fazeli17.pdf) · [Dutta et al.](https://arxiv.org/html/2411.09020v1)

**식별에서 결정까지.** Elazar et al.이 측정한 것은 정보를 읽을 수 있는 정도와 이를 교란한 뒤 원래 모델 성능의 변화다. Huang et al.은 처음부터 downstream utility를 높이도록 학습했고, Go et al.은 어떤 parameter가 loss를 바꾸는지 objective 자체를 바꿔 확인했다. 세 연구가 공통으로 지지하는 진단은 별도 모델에서의 높은 decodability, 원래 scorer의 행동 변화, 최종 loss 변화를 각각 측정해야 한다는 것이다. Huang et al.에서 설계에 따라 성능이 크게 달라진 결과와 Go et al.에서 여러 설계가 비슷했던 결과는 서로 다른 action set·utility·목표 차원으로 설명할 수 있다. 동일 조건 간 직접 대조가 없어 어느 현상이 일반적인지는 정할 수 없다. [Elazar et al.](https://aclanthology.org/2021.tacl-1.10.pdf) · [Huang et al.](https://openreview.net/pdf?id=zBG7WogAvm) · [Go et al.](https://arxiv.org/html/2605.26093v1)

**v34 해석에 쓰는 물리적 가설.** 같은 물체의 안정된 sliding에서 관측 가능한 유효 계수가 이상적으로 r₊ = μ(1 + Da), r₋ = μ(1 − Da)라고 하자. 여기서 D는 숨은 방향 ±1이고 a는 양의 asymmetry다. 그러면 (r₊ − r₋)/(r₊ + r₋) = Da다. 한 방향만 측정하면 μ와 Da가 한 값에 섞인다. 이는 v34를 이해하기 위한 단순화한 계산이다. 실제 paid prefix는 매번 다른 물체에서 얻은 noisy outgoing velocity다. 물체마다 geometry·restitution·μ가 다르므로 위 식의 상쇄는 개별 두 점에 그대로 성립하지 않는다. 알려진 prior가 좁고 물체 수가 많으면 한 방향에서도 통계적으로 부호를 맞힐 수 있다. [v34 calibration generator](target34.py) · [Fazeli et al.](https://groups.csail.mit.edu/robotics-center/public_papers/Fazeli17.pdf) · [Tramsen et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC5805965/)

### 결과에 근거해 검토할 연구 질문 10개

| 연구 질문 | 아직 답이 필요한 이유 | 비교할 변수와 방법 |
|---|---|---|
| 1. 첫 동작의 부호가 같을 때도 reversal이 정보를 추가하는가? | 접촉 결과는 입력과 현장의 contact 조건에 함께 의존한다. [Fazeli et al.](https://groups.csail.mit.edu/robotics-center/public_papers/Fazeli17.pdf) | 시작 부호를 균형화한 뒤, 같은 환경·시간·센서 오차에서 두 schedule의 direction Brier 차이를 짝지어 비교한다. |
| 2. 같은 물체를 두 방향으로 밀면 서로 다른 물체에 한 방향씩 쓰는 것보다 적은 접촉으로 충분한가? | 실물 양방향 시험은 sample–substrate 조합을 통제한다. [Tramsen et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC5805965/) | 물체 재사용과 운동 방향을 독립적으로 바꾸고, reset 오차 및 물체 간 분산을 측정한다. |
| 3. Friction asymmetry가 어느 정도일 때 sign classifier가 실제로 구별하는가? | 실물 계수 차이의 크기와 부호는 stiffness·roughness에 따라 달라진다. [Tramsen et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC5805965/) | 부호가 정의된 양의 amplitude와 amplitude=0 대조군에서 test accuracy·Brier·confidence를 함께 측정한다. |
| 4. Force와 outgoing velocity 중 어느 측정이 geometry와 friction의 혼동을 줄이는가? | Motion과 known forces에서 구별되는 물리량이 다르다. [Fazeli et al.](https://groups.csail.mit.edu/robotics-center/public_papers/Fazeli17.pdf) | 같은 impact에서 두 sensor를 동기화하고 velocity-only, force-only, 둘 모두의 direction error를 새 물체에서 비교한다. |
| 5. 반대 방향 데이터에서 개선된 sign readout이 surface 교체 뒤에도 유지되는가? | 재료·접촉 방식이 달라지면 응답과 측정 noise가 바뀐다. [Tramsen et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC5805965/) · [Dutta et al.](https://arxiv.org/html/2411.09020v1) | 독립 surface session을 test로 떼어 시작 부호별 balanced accuracy와 calibration을 측정한다. |
| 6. Scorer가 예상한 gain에 올바른 방향이 있는가, 그 정보를 쓰면 실제 순위가 개선되는가? | 속성의 decodability와 원래 예측기 행동의 변화는 실험에서 달랐다. [Elazar et al.](https://aclanthology.org/2021.tacl-1.10.pdf) | 같은 데이터·모델 용량의 scorer로 predicted sign, shuffled sign, 실제 sign의 배분·sensor 선택·Brier를 비교한다. |
| 7. 어느 방향에서 시작했는지를 모르는 encoder도 reversal의 정보를 보존하는가? | 물리 방정식의 known input과 실험자가 사후에 읽는 정보는 지식 조건이 다르다. [Fazeli et al.](https://groups.csail.mit.edu/robotics-center/public_papers/Fazeli17.pdf) · [Elazar et al.](https://aclanthology.org/2021.tacl-1.10.pdf) | 응답 순서는 유지하고 direction tag만 지운 control에 같은 readout을 학습한다. Test worlds와 probe 횟수를 고정한다. |
| 8. 방향 정보의 가치가 두 sensor의 expected risk가 비슷한 물체에 집중되는가? | 목표 parameter 성분과 나머지 성분의 정보는 downstream loss에 다르게 작용했다. [Go et al.](https://arxiv.org/html/2605.26093v1) | Factual-only action gap으로 구간을 사전 정의하고, sign 정보 제공 전후의 sensor 전환율·전환당 Brier 차이를 독립 측정한다. |
| 9. 어떤 query에서 posterior uncertainty 감소와 예측 loss 감소가 함께 일어나는가? | 로봇 물성 탐색과 Decision Utility Gain 연구는 서로 다른 목표를 최적화했다. [Dutta et al.](https://arxiv.org/html/2411.09020v1) · [Huang et al.](https://openreview.net/pdf?id=zBG7WogAvm) | Query의 incident direction과 stick threshold 근접도를 미리 바꾸고, 동일 예산에서 sign accuracy와 held-out Brier를 동시에 측정한다. |
| 10. 실물에서는 방향 전환 시간을 지불한 뒤에도 교대가 이득인가? | 서로 다른 action은 횟수가 같아도 로봇의 준비 동작과 조작 방식이 다르다. [Dutta et al.](https://arxiv.org/html/2411.09020v1) | Reset·reposition·센서 대기를 포함한 wall time과 GPU p90을 별도로 측정하고, 같은 초 단위 예산에서 stopping policy를 비교한다. |

### 심사자가 확인할 주장

이 문헌에서 직접 지지되는 것은 방향에 따라 실측 friction이 달라지는 사례, 알려진 perturbation이 식별 가능한 물리량을 바꾸는 조건, task utility에 맞춘 관측 설계의 선례다. v34에서 방어할 수 있는 차별점은 같은 probe budget으로 얻은 **표면 방향의 held-out readout**, **실제로 배분이 바뀐 정도**, **다른 물체 집단에서의 held-out Brier**를 한 실험에서 연결하는 데 있다. Readout의 실패는 해당 readout과 표본 수에 대한 결과로 한정한다. Readout이 성공해도 true sign을 입력한 비교에서 같은 예산의 결정이 개선되어야 scorer가 놓친 정보에 실용적 가치가 있었다고 판단할 수 있다. 그 비교 역시 지정한 allocator·decoder·action set에 조건부다. 원문 여섯 편은 직접 관련성으로 선정한 문헌으로, 분야 전체의 최초성을 평가하기에는 범위가 제한된다. [Dutta et al.](https://arxiv.org/html/2411.09020v1) · [Elazar et al.](https://aclanthology.org/2021.tacl-1.10.pdf) · [Huang et al.](https://openreview.net/pdf?id=zBG7WogAvm) · [Go et al.](https://arxiv.org/html/2605.26093v1)

## 실행·재현 자료

서버 위치: `datacenter3:/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/bidirectional_probe_v34`. 기본 GPU 평가는 allocation 2744468에서 완료했다. 새 head에서 128개 schedule·orientation·block의 순차 결정을 독립 재생하고, timing에 들어간 32개 기록도 대조했다. 실제 선택 확률에서 재계산한 평균 Brier의 최대 오차는 0.00e+00였다. [재현 순서](REPRODUCE_KO.md) · [Signed action 물리와 likelihood 검증](action_audit34.json) · [GPU 평가 코드](run34.py) · [기본 GPU 완료](results_policy/completion.json) · [추가 GPU timing 완료](results_head_time/completion.json) · [결정 재생 검산](independent_audit34.json).
