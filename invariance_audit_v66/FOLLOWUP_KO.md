# 재분석 이후 남는 연구 질문

아래 질문은 결과가 확보된 benchmark 밖으로 주장을 확장할 때 필요한 검정을 정리한 것이다. 목표·실행 조건·분석 단위를 명시하고, 관측 가능한 값을 쓰는 방법과 privileged 검정을 구분한다.

| 질문 | 아직 답하지 못한 이유와 중요한 변수 | 다음 검정과 판단 기준 |
|---|---|---|
| 1. Source axis를 다시 추정해도 coordinate 보존의 이득이 남는가? | 세 seed는 head를 바꿨지만 fold/ripple에서 얻은 axis 자체는 공유했다. Axis와 family 다양성의 역할을 나눌 수 없다. | Source family 묶음과 paired profile bank를 독립 재수집하고 axis·head 모두 refit한다. Target catalog와 source bank를 각각 cluster로 둔 contrast를 보고한다. [Source fitting](EXP_PLAN_KO.md) |
| 2. 다른 contact law에서 세 geometry 계수로도 충분한가? | 저차 계수의 손익은 현재 생성기의 surface·law에 조건부다. | Kernel과 물성을 함께 바꾸지 말고 law만 독립적으로 hold out한다. 사전 지정한 Brier noninferiority와 family별 방향을 확인한다. [Physics와 compression](https://proceedings.mlr.press/v78/fazeli17a.html) |
| 3. 상대속도·유효 inertia를 추정해 주면 더 나은 압축이 되는가? | 현재 local slope와 2차 미분은 impact state 전체를 표현하지 않는다. | 동일 paid input으로 해당 물리량을 추정하는 head, 정확한 값을 주는 privileged head, full head를 비교한다. [Planar impact](https://proceedings.mlr.press/v78/fazeli17a.html) |
| 4. Target에서 axis만 보정할 때와 likelihood만 재보정할 때 어느 편이 효율적인가? | Source-selected α가 실패하는 데는 geometry와 sensor model이 모두 영향을 줄 수 있다. | Target label 없이 paired calibration 0·2·8개를 배정하고 축 또는 noise covariance만 각각 바꾼다. 미사용 물체의 Brier와 calibration 비용을 함께 평가한다. [Soft control](SOFT_KO.md) |
| 5. 범위가 작은 deformation에서도 두 종류의 물체가 같은 projected input으로 겹치는가? | Exact twins는 관측 충돌을 설계했다. 자연 발생 빈도는 여기서 추정할 수 없다. | Instrument resolution에 맞춘 tolerance로 near-collision을 찾고 아직 쓰지 않은 반복 impact로 conditional uncertainty를 추정한다. 임계값 민감도를 같이 낸다. [Twin diagnostic](exp_twins.json) |
| 6. 형상 자체가 바뀐 경우와 조명이 형상을 다르게 보이게 한 경우를 구별할 수 있는가? | 두 개입은 카메라에서는 비슷해도 실제 접촉은 다를 수 있다. | 같은 impact에서 조명만 바꾸고, 별도 matched specimen에서만 실제 형상을 바꾼다. 독립 strain·ID·slip을 함께 기록한다. [실물 설계](../paired_light_pilot_v65/RIG_KO.md) |
| 7. 예측확률이 비슷해도 action cost가 달라지는가? | Mean Brier와 실제 sensing·reset 비용 사이 변환이 정해져 있지 않다. | false stick·false slip의 물리 비용과 intervention threshold를 정하고 case-weighted decision loss 및 calibration을 비교한다. [현장 endpoint](../paired_light_pilot_v65/METROLOGY_RECORDS_KO.md) |
| 8. 추가 좌표가 resolution을 높인다는 관찰이 bin 선택에 견디는가? | Fixed 10-bin 분해는 직접 검산 가능하지만 분할에 의존한다. | 새 holdout에서 고정 5·10·20 bins와 calibration/test 분리한 재보정으로 비교하고 within-bin 잔차를 공개한다. [분해 결과](calibration66.json) |
| 9. 강한 physics-conditioned head에서도 query-local probe를 추가로 살 가치가 있는가? | 이번 연구는 기존 sensing budget과 구매 결정을 동결했다. | 동일 terminal head에서 stop, fixed probe, source-trained acquisition을 비교한다. 실제 reset 비용 포함 net utility를 사전 정의한다. [기존 procurement 검증](../paired_light_pilot_v65/POWER_KO.md) |
| 10. 실물 환경에서 finite-pressure 결과의 손익 방향이 재현되는가? | 같은 generator를 쓰는 solver의 일치는 외부 물리 검증을 대신하지 못한다. | 첫 pilot은 independent ID·same-face strain·sync 수율을 검증한다. 이후 독립 물체·표면·날짜에서 사전 봉인한 모델의 slip 확률과 독립 outcome을 비교한다. [실물 근거와 범위](https://arxiv.org/abs/2110.00541) · [현장 계획](../paired_light_pilot_v65/REPORT_KO.md) |
