# v65 · 장비 없이 확인한 paired replay와 기록 QA

## v64의 0.186%가 센 것

V64에서 같은 여덟 번의 simulated calibration impact에 두 lamp를 가정하고 각각의 frozen router가 고정 3R+I 또는 v54 gate를 추천한다. 후속 표면에서 **같은 세 reference**를 양쪽에 넣고, 목표 네 번째 관측이 `reference (R)`인지 `impact (I)`인지 평가한다. 상위 router가 달라도 v54가 I를 고르면 네 번째 구매는 같아진다. V64의 0.186%는 이런 **한 calibration bank와 한 shared deployment prefix의 조합**에 대한 marginal disagreement다. 실제 장비에서 접촉을 두 번 실행해 확인한 변화율이 아니다. [v64 설계](../reflective_field_v64/SIMULATION_PLAN_KO.md) · [v64 채점](../reflective_field_v64/sim_score64.py)

서버의 Slurm CPU에서 v64 원자료로 재생했다. 저장한 24 catalog × 2 표면의 raw fixture 48 banks에 대해 동일 index에 배정된 여덟 impact와 여덟 reference를 양측에 맞췄다. 같은 위치의 reference pixels와 switch·photodiode waveform이 두 lamp에서 byte 단위로 같았다. 신호별로 v62 reader를 다시 돌린 3,072개 encoded reset-pair 값, 768개 router 추천이 원래 보존한 값과 정확히 일치했다. 저장한 raw fixture는 catalog당 한 episode다. 전체 12,288개 banks에서는 원 archived code를 사용했다. 실물 시리얼이나 발광 시간이 v64에 없으므로 두 lamp의 event 대응은 generator의 동일 sample index 및 shared evidence 재현으로 검증했다. [서버 replay 코드](dry_replay65.py)

24개 catalog의 122,880개 prefix instance에서 4번째 값·label·성능 정보를 열지 않고 1~3번째 weak/reference로 `select54`를 다시 평가했다. 검증용 strain·simulator truth·marker identity는 frozen decision 함수의 입력에 없다. 저장된 v54 gate와 모든 값이 같았다. 마지막에는 동일 surface/material의 각 calibration bank를 가능한 모든 common prefix와 교차하고 좌우 추천을 명시적으로 만들었다. 아래 분모는 독립 실물 샘플 수가 아니다.

| 가정한 specular lobe | 상위 router가 달랐던 경우 / mirrored bank 수 | 네 번째 추천 R/I가 달랐던 경우 / bank–prefix 조합 수 |
|---|---:|---:|
| 8° | 125 / 49,152 (0.2543%) | 207,663 / 125,829,120 (0.1650%) |
| 26° | 158 / 49,152 (0.3215%) | 261,173 / 125,829,120 (0.2076%) |
| 합계 | 283 / 98,304 (0.287882%) | 468,836 / 251,658,240 (0.186299%) |

관측 수는 실제 생성한 calibration bank 12,288개, 각 8개의 서로 다른 물체·impact slot 등 총 98,304개다. 각 bank를 2개 lobe와 P/C×환경의 4개 counterfactual에서 재사용했고, material에 맞는 2,560개 prefix를 같은 bank에 교차했다. 우연히 전체 상위 비교의 분모와 물리적 impact slot 수가 같은 수치다. 세부 index와 실제 XOR trace 5개를 보존했다. [데이터 집계](dry_replay65.json)

## 현장 제출물과 실패 검산

실물 session 하나는 독립 surface ID, 8개 calibration 물체, 16개 접촉 ID, 두 lamp의 원본과 동기 상태, 공통 후속 물체·세 reference, 사전 봉인한 두 최종 추천, 실제 한 번의 actuator receipt로 연결한다. 등록된 session에 검증용 strain·identity 판독이 없더라도 두 권고가 모두 관측 가능하면 terminal primary는 유지한다. 한쪽의 raw가 실제로 소실되거나 시간 대응이 허용 예산을 넘으면 paired outcome을 미상으로 남긴다. Frozen reader가 유효 raw를 읽고 보류하는 동작은 정상적인 policy 입력이다. [현장용 기록 사양](METROLOGY_RECORDS_KO.md) · [기계 판독 예제](contract_demo65/session65.json)

Datacenter3 Slurm CPU에서 명시적으로 `schema_test`로 표시한 8-slot manifest에 **14개 변형**을 적용했다. Trigger 불일치·중복, 물체 중복, 다른 후속 prefix, 봉인 시간 역전, receipt에 다른 행동, raw hash 변조, validation을 policy camera로 제출, identity/strain 장치 재사용, 9번째 slot 추가 등 10개를 거부했다. 짝이 맞지 않는 clock과 한쪽 영상 유실은 두 시나리오 모두 결과를 미상으로 보존했다. 같은 두 추천을 둔 채 독립 strain 한 기록을 추가해도 flip이 바뀌지 않았고, receipt를 유실해도 두 봉인 추천은 남았다. 계산된 수치로 기구의 독립성·marker 정상 작동·strain 정확도를 추정하지 않았다. [검사기](field_qc65.py) · [변형 테스트](test_field_qc65.py) · [실제 실행 결과](qc65.json)

실행 예시는 `python field_qc65.py contract_demo65/session65.json`이다. 재생 및 검산은 `srun`의 compute step에서 CUDA를 숨겨 `dry_replay65.py`와 `test_field_qc65.py`를 실행한다. [재현 checksum](replay65.sha256)
