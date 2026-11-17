# v66 · 어떤 contour 정보를 버릴 수 있는가?

**새 물체에서 삭제한 기하 좌표를 복원하면 Brier가 0.043678 낮아졌다.** 그러나 실제 형상을 바꿀 때 stick/slip이 얼마나 자주 뒤집히는지와, 그 방향을 약화한 예측기가 유리한지는 서로 다른 질문이었다. 이 결론은 새 contour family 두 개에서 예측을 봉인한 뒤 확인했다.

## 첨부 서베이에서 채택한 판단

가장 설득력 있는 지적은 세 가지 질문을 분리하라는 것이었다. **한 물체의 형상을 바꿔도 실제 stick/slip이 같은가? 축을 지운 두 관측으로도 서로 다른 물체의 stick 확률을 구별할 수 있는가? 현재 decoder의 likelihood에서 그 축의 가중치를 줄이면 Brier가 낮아지는가?** 세 질문의 답은 다를 수 있다. 원고의 초록·Related Work·결론을 이 구분에 맞춰 수정했다. 열거된 22편을 원 출처와 대조하고 가장 가까운 4편은 가정과 실험 조건을 읽었다. [문헌별 검토와 채택 범위](LITERATURE_KO.md)

| 첨부 제안 | 기존 증거 또는 새 조치 | 논문에서의 의미 |
|---|---|---|
| Task-dependent invariance와 representation loss를 정의 | Achille & Soatto의 sufficiency, Johansson 등의 non-invertibility, Xiao 등의 정보 보존을 인용했다. 실제 연산은 hypothesis별 whitened residual의 특정 축에 작용한다. | 형상 차이가 크다는 사실만으로 제거할 nuisance로 판정할 수 없다. [Achille & Soatto](https://jmlr.org/papers/v19/17-646.html) · [Johansson et al.](https://proceedings.mlr.press/v89/johansson19a.html) · [Xiao et al.](https://arxiv.org/abs/2008.05659) |
| 물리 구조에서 유도한 압축과 비교 | Fazeli 등은 평면 impact에서 contact inertia·velocity와 momentum 기반 표현을 비교했다. 본 연구는 같은 비용에 얻은 일곱 contour 측정의 변동 방향을 이용한다. | 새 baseline은 배포 시 얻을 수 있는 profile에서 도출해야 한다. 정확한 inertia나 pressure를 주면 privileged diagnostic이다. [Fazeli et al.](https://proceedings.mlr.press/v78/fazeli17a.html) |
| Axis alignment × physical coupling을 교차 | v47에서 이미 같은 pre-contact 입력을 두고 독립적으로 교차했다. 고정렬에서 projection−full은 inert일 때 −0.009763, strong일 때 +0.027425였다. | 새 실험으로 중복 계산하지 않고 본문으로 올렸다. 같은 route와 evidence에서 target axis를 바꿔 v46 excess loss가 90.4% 줄었다는 수치는 그 개입에 한정한다. [v47](../factorial_projection_v47/REPORT_KO.md) |
| Relative-risk router, 그림·법칙을 바꾼 test | v47의 observation-only router는 0.151723, full은 0.151709였다. v66은 source-selected suppression, 동일 구조 readout과 새로운 contour holdout에 집중했다. | 일반적인 routing algorithm이나 sim-to-real 개선의 독창성을 주장할 근거는 부족하다. [기존 router](../factorial_projection_v47/ROUTER_KO.md) |

## 1. 이미 조사한 두 target family에서의 재분석

V45 fold·ripple에서 학습 및 선택하고 v46 terrace·dimple 12개 catalog에서 재분석했다. 각 방법에 동일하게 구매된 측정, 같은 noise, 같은 pressure solver outcome을 썼다. 비교 단위는 개별 pixel·충돌·noise replicate가 아닌 catalog다. 아래에서 Brier는 낮을수록 좋고, 각 실험별 primary 비교 2개에 동시 95% paired interval을 적용했다. [Soft likelihood 사전 계획](soft_plan.md) · [Readout 사전 계획](EXP_PLAN_KO.md)

| 실험과 비교 기준 | Source에서 고정한 선택 | Target mean Brier | 비교 결과 |
|---|---|---:|---|
| 같은 location route에서 축 가중치만 바꾸기 | full / α=0.75 / hard | 0.155170 / 0.161644 / 0.162776 | 부분 억제−full **+0.006473** [0.005985, 0.006962]. 부분 억제−hard **−0.001133** [−0.001216, −0.001049] |
| 모든 paid profile에 같은 구조의 head 재학습 | full / dropped / restored | **0.139452 / 0.166583 / 0.139483** | restored−dropped **−0.027100** [−0.028674, −0.025526]. full−dropped **−0.027131** [−0.028877, −0.025384] |
| 같은 모든 paid profile에서 기존 likelihood 유지 | full / hard | 0.155170 / 0.172776 | dropped도 재학습하면 0.166583까지 회복했다. 삭제 때문에 남은 손실과 기존 decoder의 개선 여지가 함께 관찰됐다. |

첫 실험에서 α는 signed residual에 곱할 억제량이다. α=0.75이면 residual의 25%, likelihood energy의 6.25%가 남는다. 이 선형 변환은 가역적이지만 기존 likelihood가 그 방향을 덜 신뢰하도록 바뀐다. 두 번째 실험에서는 일곱 원래 좌표, 여섯 직교 성분과 0, 여섯 직교 성분과 signed 좌표를 각각 입력했다. Source의 8개 catalog로 학습하고 4개로 epoch을 선택했다. 같은 64-unit SiLU readout, 공통 projected physics log odds, 3개 seed, 동일 probe fallback이다. 양 실험의 hard 값이 다른 이유는 첫 비교가 location route에서 선택한 경우에만 억제하고 둘째는 모든 paid profile을 변환하기 때문이다. [억제량 전체 곡선](SOFT_KO.md) · [가중치·분모·catalog별 검정](EXP_RESULTS_KO.md)

**개선은 calibration만으로 설명되지 않았다.** 고정된 폭 0.1의 확률 bin 10개로 같은 예측을 분해했다. Dropped→restored에서 reliability penalty는 0.004521→0.000995로 감소했고 resolution은 0.086388→0.109859로 증가했다. 같은 base rate에서 Brier 차이는 −0.003526−0.023472−0.000102 = −0.027100이다. 마지막 수는 같은 bin 안의 서로 다른 예측을 평균할 때 생기는 잔차다. 이 descriptive 분해에서 더 큰 차이는 서로 다른 outcome 빈도의 사례를 구별하는 데서 나왔다. Bin 정의나 target 분포가 바뀌면 항별 크기도 달라질 수 있다. [계산 계획](calibration_plan.md) · [서버 계산과 catalog 값](calibration66.json)

## 2. 정보가 실제로 사라지는 조건을 확인한 진단

V47의 구매된 query profile을 바탕으로, 같은 midpoint에서 frozen axis에 정확히 평행한 ±1.20 mm 변형 두 개를 새로 구성했다. 독립적인 pressure solver가 이미 산출한 matched outcomes를 사용했다. 같은 쌍 안에서 factual evidence, paid action, 대체 probe 신호, 위치와 normal variance는 같고, full signed coordinate는 2.4 mm 차이가 난다. 축을 지우면 두 입력은 수치오차 5.33×10⁻¹⁵ mm 이내로 겹친다. 두 label이 다르면 두 경우에 똑같은 확률 p를 내는 예측기의 평균 Brier는 정확히 (p−½)²+¼이고 최솟값은 ¼이다. 두 label이 같으면 pair identity까지 주어진 최적 예측기는 loss 0에 도달할 수 있다. [설계](EXP_TWINS_PLAN_KO.md) · [검산 결과](exp_twins.json)

각 coupling에서 264,960개 matched condition 중 inert에서는 0개, strong에서는 89,856개(33.91%)의 stick/slip이 반전됐다. 동일 projected input을 공유하는 균등 twin에서 strong 조건의 Brier 하한은 **0.084783** [0.083008, 0.086557]이었다. 이 값은 설계한 twin 분포에서의 absolute risk 하한이다. 일반 배포에서 full 모델의 최적 risk를 모르므로 앞 절의 0.027100을 Bayes information loss라고 부를 수 없다. 또한 같은 물체 내 변형이 항상 label을 보존해도, 축 삭제가 서로 다른 물체를 겹치게 하면 정보가 사라질 수 있다. [수학적 반례와 적용 범위](LITERATURE_KO.md) · [Side information 검산 코드](exp_twin_side.py)

## 3. 더 적은 수의 물리적 geometry feature를 주면 어떻게 되는가?

일곱 profile 측정의 간격은 2 mm다. 같은 일곱 값에 quadratic curve를 least squares로 적합해 중심의 높이, 기울기, 2차 미분만을 추가 readout 채널로 줬다. 높이 미분은 noisy profile에서 추정한 quantity이고, 하중이나 정확한 curvature·inertia를 측정한 값은 아니다. 이 비교는 동일 관측에서 얻는 저차 표현의 실험이다. Fazeli 등의 contact wrench 모델을 재현했다는 의미는 아니다. 세 방법이 공유하는 hard-projected physics forecast는 여전히 6차원 직교 성분의 정보를 사용한다. [사전 계획](moment_plan.md) · [Fazeli et al.](https://proceedings.mlr.press/v78/fazeli17a.html)

Terrace·dimple에서 별도로 동일 CPU training 조건을 맞춰 재학습한 결과, 세 계수를 넣은 head는 **0.139355**, full은 **0.139455**, dropped는 **0.166432**였다. 세 계수−full은 **−0.000100** [−0.000593, +0.000392], 세 계수−dropped는 **−0.027077** [−0.029125, −0.025028]였다. 이 두 family에서는 7개 좌표를 모두 추가해야만 성능을 회복하는 것은 아니다. Source axis energy 중 quadratic subspace에 속하는 비율은 두 높이에서 66.85%와 82.54%였다. 저차 표현에도 제거된 축과 겹치는 정보가 남는다. 해당 비율만으로 slope 하나가 이 차이를 일으켰다고 식별할 수 없다. [축의 겹침](moment_axis.json)

## 4. 결과를 보기 전 고정한 새 contour family의 확인 실험

기존 광학 형상과 다른 두 식으로 새 물체를 만들었다. Wendland는 변형이 유한한 구간에 국한되는 모양이고 dispersive는 중심 양쪽에 반대 방향의 shoulder가 있고 꼬리가 길다. 각 family에 세 변형을 두고 원점의 높이와 tangent를 맞췄다. 과거 48개 variant와 비교해도 두 새 family의 여섯 곡선은 높이·기울기의 단순 이동으로 겹치지 않았다. 새 12개 catalog에서 뽑은 6,144개 중 고유 prototype은 5,945개였다. 보관된 15개 실험의 deployment·calibration manifest와 대조해 정확히 같은 6차원 prototype은 없었다. 같은 두 surface와 law, 기존 camera와 finite-pressure contact solver를 사용했다. 바뀐 독립 축은 contour family와 물체 표본이다. [형상과 가설 사전 고정](locked/LOCK_KO.md) · [기존 자료와 중복 검산](locked/independence.json)

기존 v45·46 결과를 바탕으로 방법을 정한 뒤 새 label을 만들기 전에 frozen likelihood, location routes, source-trained readout 및 quadratic baseline의 모든 새 예측을 hash로 봉인했다. 세 계수 baseline의 가설은 다른 두 가설보다 늦게, 기존 terrace/dimple의 결과를 본 후에 추가했으며 새 target label에는 접근하지 않았다. 각 실험의 2개 주요 대조 안에서만 simultaneous 95% interval을 보장한다. 비열등 허용치 +0.002와 실용 개선폭 0.002는 실측 로봇 비용에서 환산한 값이 아니며 연구자가 사전 결정했다. [Readout 사전 계획](EXP_LOCKED_PLAN_KO.md) · [세 계수 사전 계획](moment_lock_plan.md)

| 새 두 family, 전체 policy 기준 | 기준 대비 Δ Brier | Simultaneous 95% CI | 사전 판단 |
|---|---:|---|---|
| Frozen location route에서 α=0.75 − full | **+0.009712** | [+0.009246, +0.010178] | source에서 고른 억제량이 평균적으로 손해라는 예측 재현 |
| 같은 route에서 α=0.75 − hard | **−0.002167** | [−0.002279, −0.002055] | 완전 삭제보다 나을 것이라는 예측 재현 |
| 공통 readout, restored − dropped | **−0.043678** | [−0.045457, −0.041899] | 좌표 복원의 개선이 0.002를 넘음 |
| 공통 readout, full − dropped | **−0.046525** | [−0.048234, −0.044816] | 원 좌표 보존에서도 개선이 0.002를 넘음 |
| CPU-matched baseline, quadratic − full | **+0.003586** | [+0.003261, +0.003910] | 비열등 기준 +0.002를 충족하지 못했고 interval 전체가 이를 초과 |
| CPU-matched baseline, quadratic − dropped | **−0.042418** | [−0.044018, −0.040818] | 저차 표현이 삭제보다 낫다는 방향 재현 |

전체 readout Brier는 full **0.135604**, restored **0.138451**, dropped **0.182129**였다. Restored와 full은 같은 정보를 담지만, 새 family에서 별도 학습한 두 head의 차이는 **+0.002847**이었다. 정보량이 같아도 유한한 head가 그 정보를 학습하고 새 분포에서 활용하는 정도는 달라진다. CPU 학습을 맞춘 별도 quadratic 비교의 절대 Brier는 quadratic 0.139439, full 0.135853, dropped 0.181857이다. 두 학습 환경의 숫자를 섞어 효과를 계산하지 않았다. [새 readout 원자료](exp_locked_results.json) · [새 quadratic 원자료](moment_locked_results.json)

**평균만 보면 놓치는 반례가 이번 연구의 중심 결과다.** Wendland에서는 hollow↔bulge만 바꿔도 stick/slip이 뒤집힌 경우가 **33.32%**였지만, 같은 관측에서 α=0.75의 Brier는 full보다 **0.000491 낮았다**. Dispersive에서는 label 반전이 **7.82%**로 더 드물었는데 같은 방식의 억제가 Brier를 **0.019915 높였다**. 두 family의 관찰로 모든 모양에 대한 상관을 추정할 수는 없다. 다만 물리적 label 반전 빈도 하나로 구체적인 likelihood 연산의 이득을 판정하는 해석에는 반례가 된다. 새 contact law는 이 비교에 포함하지 않았다. [Family별 결과](locked/LOCKED_RESULTS_KO.md) · [물리 label 검산](locked/locked_score_audit.json)

## 5. 논문 기여와 아직 남은 검정

- **본문 기여.** 같은 paid evidence에서 (i) 형상을 물리적으로 바꾸는 것, (ii) 그 형상을 읽는 likelihood의 주축을 약화하는 것, (iii) 정보를 남기고 readout만 다시 학습하는 것을 각각 조작할 수 있다. 새 두 family에서 서로 다른 손익이 나왔고, 세 geometry 계수 baseline까지 같은 예산으로 비교했다. 일반적인 ‘invariance는 위험하다’보다 재현 가능하고 검증 범위가 명확한 주장이다.
- **외부 타당성.** 새 형상 확인 실험은 같은 finite-pressure solver와 기존 surface·contact law를 사용했다. 원고에 v56–v65의 calibration·sensor 결과를 추가하면서 실제 DiLiGenT 정지 영상의 2/17 tracking 오등록, simulation의 0.1863% 구매 전환, 실물 pilot 설계 및 미측정 field Brier를 구분했다. [광학·실물 설계](../paired_light_pilot_v65/REPORT_KO.md)
- **다음 우선순위.** 첫째, 동일한 contour에 contact law만 독립적으로 바꾸고 source axis까지 여러 번 새로 fit해 효과가 어디까지 전이되는지 본다. 둘째, v65 실물 pilot에서 독립 marker·strain·slip과 동일 impact 조명 쌍의 판독 수율을 확보한다. 실제 deployment나 clock·reset 비용의 이득은 그 뒤 판단할 수 있다. [근거를 붙인 후속 질문 10개](FOLLOWUP_KO.md)

재현 단위는 두 split 각각 12개 catalog다. Source 학습 3회는 head seed 변동을 다루며 source axis를 세 번 독립 학습한 것이 아니다. 전체 NPZ·weights·seal은 datacenter3의 기존 프로젝트 아래 `invariance_audit_v66/`에 저장했다. 채점용 manifest에서 개행을 처리하는 I/O 오류를 수정한 후 이미 봉인한 예측과 같은 통계식으로 채점했다. [정책·모델 사전 고정](EXP_LOCKED_PLAN_KO.md) · [Source weights](exp_fit.json) · [Label release](locked/FORECASTS_SEALED)
