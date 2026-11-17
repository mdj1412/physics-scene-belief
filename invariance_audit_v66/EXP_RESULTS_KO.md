# V66 · 같은 readout을 다시 학습해도 삭제한 geometry 방향이 필요한가?

**Terrace·dimple에서는 그 방향을 보존한 readout이 유리했다.** 동일한 source에 같은 architecture를 각각 학습했을 때 네 번째 관측까지 포함한 정책의 held-out Brier는 full 0.139452, hard-projected 0.166583, 제거 성분을 별도 좌표로 복원한 경우 0.139483이었다. 한편 hard-projected 입력 자체에서도 재학습으로 frozen physics decoder보다 loss가 0.006193 낮아졌다. 이 두 관찰은 decoder가 남긴 개선 여지와, 같은 재학습을 거쳐도 남는 삭제 비용을 구분한다. [분석 계획](EXP_PLAN_KO.md) · [12개 catalog 결과와 개별 seed](exp_results.json)

## 정보 조건과 평가 단위

V45의 fold/ripple 12개 catalog 중 8개로 fit하고 나머지 4개로 epoch을 선택했다. 평가에는 v46의 별도 support 600–611과 새로운 두 contour family를 사용했다. Surface와 law의 종류는 source와 target이 공유하므로 결과는 contour family transfer에 해당한다. V46의 기존 physics score는 연구진이 이미 본 결과다. 신규 readout의 가중치를 source에서 동결하고 target의 12개 prediction archive를 hash한 뒤 별도 script로 label을 열었다. 기존 benchmark를 다시 분석한 것으로서, 피험 조건까지 사전에 숨겨 둔 신규 confirmatory study와는 근거 수준이 다르다. [Source 학습과 prediction](exp_readout.py) · [Target scoring](exp_score.py)

세 readout은 factual 11개 채널, 알려진 query 높이, 공통 hard-projected physics log odds를 사용한다. 공통 physics log odds는 candidate profile에서 제거 방향을 이미 버린 likelihood로 계산된다. 원래 입력의 삭제 성분을 이 경로로 되살릴 수 없다. 세 readout이 달리 받는 것은 구매한 seven-channel query의 좌표뿐이다. Full은 원래 좌표, dropped는 직교 성분 6개와 상수 0, restored는 직교 성분 6개와 signed parallel 성분 1개를 받는다. 모든 fit에서 같은 64-unit SiLU residual readout과 AdamW를 사용했다. 별도 normalization은 source에만 맞췄다. 복원 좌표로 full profile을 재구성할 수 있는지 수치로 확인했다. 단순한 좌표 복원이 아니라 target forecasting 성능을 비교하기 위해 세 모델을 독립적으로 재학습했다. [Source axis](../image_pressure_v45/source_axis45.json) · [입력과 readout 정의](exp_readout.py)

세 source-fit seed를 사용했다. Fit에는 query row 353,280개, source validation에는 353,280개, target에는 1,059,840개가 있다. 여기에는 조건과 noise 반복이 들어 있으며 독립 표본 단위는 **target catalog 12개**다. Target에서 6,144개의 object draw가 있고 catalog 내 고유 prototype은 5,976개다. 각 128개 물체 중 115개는 query, 13개는 probe를 구매했다. Probe arm은 세 모델에서 같은 archived forecast를 그대로 쓴다. Readout은 law·family·variant·radius·offset·true label을 입력받지 않는다. 두 finite radius와 세 offset에 걸친 균등 평균을 사용했고, 개별 binary label의 squared loss를 명시적으로 평균해 동일한 값이 나오는지 교차 검산했다. [가중치 및 source 성능](exp_fit.json) · [v46 독립 표본](../normal_variance_v46/REPORT_KO.md)

| Forecast | 전체 정책 Brier | Purchased query만 |
|---|---:|---:|
| Frozen physics, full | 0.155170 | 0.148904 |
| Frozen physics, dropped | 0.172776 | 0.168500 |
| Source-retrained, full | **0.139452** | **0.131409** |
| Source-retrained, dropped | 0.166583 | 0.161607 |
| Source-retrained, restored | 0.139483 | 0.131443 |

Primary인 **restored − dropped**는 −0.027100 [−0.028674, −0.025526], **full − dropped**는 −0.027131 [−0.028877, −0.025384]였다. 구간은 두 paired contrast에 Bonferroni를 적용한 catalog-level simultaneous 95% t interval이다. Seed별 risk를 평균했으며 seed를 추가 독립 표본으로 세지 않았다. 보조적으로 restored − full은 +0.000031 [−0.000270, +0.000333]이었다. 등가성 margin은 설정하지 않았다. Family별 restored − dropped는 terrace −0.042485 [−0.044574, −0.040397], dimple −0.011714 [−0.013253, −0.010176]로 효과 크기가 달랐다. [원시 catalog 점수](exp_results.json)

**판정.** 이 두 family에서 삭제한 성분은 주어진 source-trained readout에 유용했고, hard projection의 손실을 기존 hand-specified likelihood만으로 설명하기 어렵다. 반대로 같은 dropped representation에서도 재학습이 도움이 됐으므로, frozen decoder의 Brier 차이를 그대로 정보 손실량으로 환산하는 해석 역시 성립하지 않는다. Source 분포·가중치·모델 클래스와 optimization을 조건으로 얻은 추정이다. 새 solver나 실제 접촉에서의 Bayes risk 차이는 아직 미지수다. [Frozen 재현과 readout contrast](exp_results.json)

## 완전 정렬된 matched twins로 정보 소실을 직접 보일 수 있는 범위

V47에는 squared alignment 0.1과 0.9를 교차한 실험이 이미 있다. 여기서는 같은 물체, factual 측정, sensor noise와 구매 여부를 보존하고 **새 optical 조건인 exact alignment 1**을 구성했다. 동일 query midpoint에서 frozen axis를 따라 ±1.20 mm 이동시킨다. Pressure evaluator에 optical axis가 입력되지 않으므로 v47의 기존 hollow/bulge solver label을 그대로 사용할 수 있다. 양쪽의 측정값은 삭제 방향에서만 다르다. 96개 catalog·family·height 구성(각 512개 object draw)을 서버에서 재생했다. Factual 11개 채널, perturbation 전 query, alternative probe, 기존 acquisition, 영상에서 추정한 x와 normal variance가 두 variant에서 원소 단위로 같았다. 따라서 공통 factual와 동일한 projected profile로 재계산하는 frozen projected log odds 역시 같다. [Side information 검산](exp_twin_side.json) Projected 좌표 한 개를 두 member가 공유하도록 정의해 floating-point residue로 sign을 읽지 못하게 했다. 별도로 양쪽 profile에 projection을 직접 계산한 최대 차이는 이 shared coordinate에 대해 5.33×10⁻¹⁵ mm였고, signed 좌표의 최소 pair 간격은 2.4 mm였다. [사전 정의](EXP_TWINS_PLAN_KO.md) · [수치 검증](exp_twins.json)

두 member의 label이 다르면 동일한 예측 p를 낼 수밖에 없는 model의 pair 평균 loss는 ½[p²+(1−p)²] ≥ ¼이다. 두 label이 같을 때의 최솟값은 0이다. 따라서 동일 가중치를 준 pair의 ¼×flip rate는 **이 제한된 모집단에서 projected forecast가 달성 가능한 absolute Brier의 하한**이다. 각 pair의 identity까지 알려주는 privileged lookup에서는 그 하한을 정확히 달성한다. 같은 privileged context 안에서 signed 좌표를 추가하면 두 member를 구별해 empirical lookup loss가 0이 된다. 실제 예측기는 이 context와 정답표를 받지 않는다. 서로 다른 pair의 full observation이 언제나 식별 가능하다는 가정도 필요하지 않다.

| Mechanical coupling | 다른 label 수 / 구매한 query의 matched condition 수 | Label flip rate | Projected Brier floor [catalog 95% CI] |
|---|---:|---:|---:|
| Inert | 0 / 264,960 | 0% | 0 |
| Strong | 89,856 / 264,960 | 33.913% | **0.084783 [0.083008, 0.086557]** |

264,960은 같은 object를 네 family·두 높이·두 radius·세 offset에 재사용한 조건 수다. Noise 반복은 이 수에 포함하지 않았다. Inert의 floor가 0이어도 unprivileged full/dropped 시스템이 정확하다는 보장은 없다. 이 synthetic collision은 특정 개입에서 task-relevant 정보를 없앨 수 있음을 직접 보인다. v46에서 관찰한 −0.027100의 전부를 설명하거나 두 운영 predictor 사이의 Bayes gap을 추정하지 않는다. [Panel별 결과](exp_twins.json) · [물리 label solver](../factorial_projection_v47/mechanics47.py)

학습은 datacenter3 Slurm 2744471의 RTX PRO 6000 Blackwell 한 장에서 수행했다. 시작 시 GPU process가 없었고 memory는 0 MiB였다. Twin 계산은 CUDA를 숨긴 별도 Slurm CPU step에서 실행했다. 서버에서 재현하려면 exp_readout.py의 fit, predict를 차례로 수행하고 exp_score.py와 exp_twins.py를 각각 실행한다. 같은 source fit을 재사용할 수 있도록 아홉 checkpoint와 원 target prediction archive의 SHA-256을 저장했다. [실행 기록](exp_run.log) · [Forecast checksum](exp_forecasts.json) · [계측·side information 검산](exp_twin_side.json)
