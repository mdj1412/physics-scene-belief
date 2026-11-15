# v38 · 네 정책이 공유한 입력과 보정 선택

세 번의 conventional probe와 +1.10의 네 번째 probe를 모두 치른 뒤, 아직 접촉하지 않은 물체의 query가 stick할 확률을 반환한다. 두 단계의 차이를 채점하기 위해 동일한 네 번의 기록에서 네 번째 outgoing velocity를 가리기도 한다. 비교의 첫 세 방법과 continuous baseline은 v36–v37에서 사용한 source artifact를 그대로 읽는다. v38의 형상명, contact law, orientation, regime, prototype ID, deployment outcome은 policy head에 전달하지 않는다. Common physics decoder는 기존 조건과 같이 population coupling α를 알고 있다. [GPU runner](run38.py) · [사전 분석 계획](PLAN_KO.md)

## Calibration evidence를 사용할 수 있는 경로

환경의 1,024개 factual calibration row와 세 번의 유료 접촉으로 common acquisition을 계산한다. 128개 deployment 물체 중 13개에 probe, 나머지 115개에 query view를 배정해 물체마다 실제로 취득한 한 종류의 observation만 decoding한다. Acquisition에 허용한 유료 접촉을 세 번으로 고정했으므로, 네 번째 observation을 안 보고 점수화한 방법도 이 observation을 간접적으로 전달받지 않는다. v37에서는 네 번째 observation이 배정도 바꿀 수 있었다. 각 방법이 같은 네 번째 속도를 쓴다는 점은 v37 고속 조건과 공유하지만, 두 버전의 전체 수치 차이에는 holdout population과 이 acquisition control의 변화가 함께 들어 있다. [Prefix-only acquisition과 선택된 관측의 decoding](run38.py)

Calibration head는 첫 세 접촉에서 distribution over source model, 방향 cue z를 계산한다. 네 번째를 허용하면 source likelihood로 distribution을 한 번 더 갱신한다. 모든 방법·단계에서 z와 신호 민감도 β=0.8289760825845313은 동일하다. Common physics probability p, 선택한 weight w에서 최종 확률은 sigmoid(logit(p)+βzw)다. p가 0 또는 1일 때는 그대로 둔다. 보정 전후 frame은 다음과 같다. [v36 source posterior](../query_matched_probe_v36/gate36.py) · [v37 speed adaptation](../probe_speed_v37/gate37.py)

| 방법 | Deployment prediction에서 weight w를 정하는 방식 |
|---|---|
| Fixed | 항상 0.325를 적용한다. Fourth response는 사용하지 않는다. |
| Binary | 현재 posterior에서 source gain table의 full correction 기대 이득을 읽고, 0보다 크면 1, 나머지는 0으로 둔다. |
| Three-action | 동일한 table로 0.325z와 z의 expected gain을 비교하고 0의 gain을 0으로 둔다. 동률 우선순위는 0.325, 0, 1이다. |
| Continuous | 같은 posterior에서 near-field directional strength의 평균을 구해 연속 weight로 사용한다. 이 방법은 최종 gain table의 최대점을 직접 찾는 별도의 continuous optimizer와 다르다. |

## GPU p90 구간

한 번의 측정은 8개 환경의 총 1,024개 물체를 처리한다. CUDA를 동기화하고 timer를 시작한 뒤, 각 환경의 prefix likelihood, 128개 물체에 대한 acquisition, 선택된 센서의 physics probability, 해당 정책의 보정 선택과 최종 확률 계산을 수행한다. Host에서 결과를 얻고 CUDA를 다시 동기화한 후 종료한다. 실제 장비의 probe·reset, 기록의 생성과 one-time particle bank 생성은 제외한다. 네 방법마다 매번 실제 acquisition과 decoding을 다시 실행한다. Warm bank가 존재하는 상태에서 순서를 무작위화한 1회 warmup과 3회 실측을 8개 paired timing block에서 실시한다. 한 방법당 24회이며 같은 호출에서 관측 noise 첫 realization의 예측을 raw archive의 해당 예측과 비교한다. 고정 정책은 네 번째 관측의 비용을 지불하더라도 해당 보정 선택을 계산하지 않아도 된다. [실측 경로](run38.py) · [추정 방법](stats38.py)

## Archive와 독립 검산

- 각 cell과 block 안의 shared_plus/minus 파일에는 calibration factual rows, 첫 세 outgoing velocity, paid directions, query/probe의 선택, 선택한 deployment sensor readings, common physics probability와 z가 들어 있다. Orientation 안에서 세 regime의 이 항목이 같음을 생성 중 확인한다.
- Six regime–orientation 파일에는 네 번째 관측, held-out label, weight, 환경별 empirical Brier, timing과 방법 실행 순서를 기록한다. Noise 32회는 같은 물체를 반복 측정한 것이다.
- audit38.py는 모든 archive의 SHA-256을 확인한다. Frozen gate를 독립 호출하고 source gain table을 NumPy interpolation으로 다시 평가해 네 정책의 weight와 held-out Brier를 재구성한다. Final choice별 Brier 기여도는 해당 행동을 고른 환경의 비중이 곱해진 값이다. 이를 더하면 3-action과 고정 32.5%의 차이가 된다. Prefix3→prefix4의 행동 조합에 따라 Brier 변화도 같은 방식으로 분해한다. [정책 검산](audit38.py)
