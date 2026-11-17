# v65 · 실물 event ID, 검증 기록과 blind replay handoff

## 현장 파일을 연결하는 방법

기계적으로 독립인 **표면 시편 ID**를 `surface_instance`에, 같은 시편의 표면 교체·날짜·재장착 정보를 `surface_day`에 기록한다. 한 pilot의 기본 키는 `(surface_instance, session)`이다. 표면 1개를 여러 날 쓰면 동일 시편 아래의 반복으로 남긴다. Pilot당 여덟 물체 각각에 2.0 N reference와 impact를 한 번씩 배정한다. 여덟 번의 impact는 두 lamp가 공유한다. 각 reference/impact의 하드웨어 trigger에는 예산 등록 시 중복되지 않는 `trigger_id`를 발행한다. 각 session의 네 reset pair에서 둘째 물체 직전 어느 전자센서를 reset했는지도 보존한다. 두 조명을 여러 날 각각 다시 치는 방식에는 동일 event ID를 부여할 수 없다.

입력 파일은 `session65.json`에 raw 경로·역할·SHA-256으로 연결한다. Native raw 영상, ADC trace, calibration 및 pulse timing은 각각 보존한다. 여러 stream이 같은 파일에 들어가는 경우에는 원본 archive와 hash를 보존하고, 검증 가능한 event별 stream export를 만든다. `mid_us`에 기록하는 여덟 시점은 긴 원본 중 frozen reader에 입력하는 여덟 frame의 **같은 clock 기준 exposure midpoint**다. 하드웨어 sample rate가 달라질 때는 사전 고정한 adapter와 원본의 frame/sample index를 함께 보관한다. 예제의 20 μs, 5 μs는 schema-test 값이다. 현장 허용 오차는 rig의 위치 오차 예산과 측정된 jitter·drift에 맞춰 연구자가 고정한다. [실행 가능한 QA 예제](contract_demo65/session65.json) · [검사기](field_qc65.py)

| 파일에 남길 것 | 필수 의미 | 누락 처리 |
|---|---|---|
| 두 lamp의 event별 raw | 실제 발광 장치 ID, 동일 기계 trigger ID, policy camera ID, 공통 clock의 midpoint와 clock 오차 상한. 라우팅에 쓰는 물체 순서와 reset 기록도 포함한다. | 각 lamp의 촬영/저장 실패를 `raw: null`과 사유로 남긴다. 진짜 판독 보류는 영상·파형이 남아 있어 기존 router가 fallback할 수 있다. |
| 독립 identity | `(trigger_id, camera-frame, material-marker ID)`, 고정된 identity 장치 원본과 해시, 투영 보정 파일, frame별 reference ID와 tracker ID, `correct/swap/unresolved/occluded`, 복수 판독자 간 불일치, blind adjudication. 연속 잘못된 구간의 시작·끝과 eligible frame을 보존한다. | 가림·기준 marker 간섭은 unresolved로 명시한다. Tracker의 내부 confidence를 identity 정답으로 복사하지 않는다. |
| 독립 local strain | 대상 위치·marker와 gauge 사이 거리, local gauge 좌표계와 감도/방향, 인장·압축 부호 convention, channel/장치 ID, 보정식·원시 ADC 해시, 기준/접촉 창·sample index, 단위·불확실성·zero deadband와 saturation 상태를 기록한다. 다축이면 판단에 사용한 strain component를 명시한다. | 결측·deadband·계측 범위 밖을 별도 범주로 둔다. 정상 판독 수와 identity가 함께 검증된 수, 둘 다 누락된 수를 구분한다. |
| 공통 후속 prefix와 두 sealed recommendation | 여덟 calibration 물체에서 독립된 후속 물체의 첫 3회 reference raw, frozen policy/adapter hash와 각 lamp의 입력 hash, upper 선택과 최종 `R/I` 추천, common clock seal 시각. 두 정책 모두 같은 prefix hash를 가리킨다. | 양측 추천을 확인할 수 없으면 primary paired outcome은 미상이다. 측정 가능한 한쪽과 이유는 유지하고 `unknown = no change`로 대체하지 않는다. |
| 배정·execution receipt·outcome | 사전 lamp assignment와 배정확률, actuator receipt·시각·실제 R/I, query holdout의 stick/slip과 독립 측정 경로. | 두 추천을 각각 실행한 것으로 기록하지 않는다. Receipt가 누락됐어도 seal된 두 추천의 불일치는 평가할 수 있다. Brier는 미래 outcome과 실제 취득 관측을 확인한 경로에서만 평가한다. |

## 역할과 집계

Tracker와 router의 입력은 자기 lamp의 raw 영상, 허용된 전자센서, reset 위치와 공통 prefix다. Bench technician은 immutable raw→SHA manifest를 생성한다. 독립 판독자는 lamp policy·실행 결과를 가린 검증 뷰에서 identity 및 strain sidecar를 기록한다. 정책 출력과 audit 결과를 각자 봉인한 뒤 `trigger_id`로 결합한다. 같은 파일이 임의로 다시 이름 붙여졌는지, camera housing·beam splitter·trigger·marker의 오차가 독립 장치들에 공통으로 남는지는 실물 bench에서 검증해야 한다. 소프트웨어의 role check만으로 물리적 독립성은 증명되지 않는다.

한 session의 primary는 `I[rec_A != rec_B]`이며 두 권고를 실제로 모두 집행했다는 뜻은 아니다. 한 개의 공통 후속 물체와 하나의 실제 actuator receipt만 허용한다. 판단이 가능한 session에는 optional identity/strain 누락 때문에 primary를 버리지 않는다. 순서·hash·짝이 손상된 bank는 QC 예외를 남기고, 전체 모집 session 수에서 지우지 않는다. 누락 session에 대한 완전자료 분석과 0/1 극단 경계, 그리고 누락 원인/표면별 비율을 함께 제시한다.

검사기는 파일 provenance와 핵심 lineage를 검사한다. Native 파일 decoder, coordinate registration, 전기적 isolation, 정답 독립성, 모든 시편에 걸친 중복 ID 및 시험 장비 clock drift는 별도 현장 검정 대상이다. Calibration 좌표·독립 평가 기록의 실제 파일 스펙은 선택한 장치에 맞춰 sidecar로 공급한다. 시범용 manifest와 원자료 자리표시는 전부 `origin=schema_test`이며 field estimate에 넣을 수 없다.
