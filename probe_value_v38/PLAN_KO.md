# v38 사전 분석 계획

2026-11-15 KST. 이 계획은 v38의 target Brier와 latency 측정 전 작성했다. 설계 결정과 별도로, 코드는 기존 v37에서 동결한 source artifact를 사용한다.

## 가설과 primary comparison

Fixed 4 중 첫 세 probe는 높이 0에서 부호를 교대한다. 마지막 접촉의 높이는 0.023, 부호는 양, 속도는 +1.10이다. Query는 0.025, +0.425다. Held-out 두 geometry와 두 contact law를 교차한다. Strong, weak, local null과 orientation 두 방향을 동일 비율로 둔다. 이 여섯 가지 가능성은 evaluation용 true-law 모델에만 명시한다. 속도, 보정 계수 β, source gain table, tie rule, 평가 표본을 결과에 따라 변경하지 않는다.

사전 지정한 세 행동은 0%, 32.5%, 100%다. 동률일 때 v37의 frozen rule에 따라 32.5%, 0%, 100% 순서로 고른다. 같은 source artifact의 binary gate와 fixed 32.5%를 공동 주 비교 대상으로 둔다. Continuous baseline은 source posterior mean near-field strength를 log-odds 보정에 곱하는 v37의 frozen soft correction이다. 모든 방법에서 signed cue z는 처음 세 번으로부터 같은 방식으로 계산한다. Continuous baseline은 target에서 새로 적합하지 않는다.

## 비교에 허용하는 정보

네 번째 probe를 보기 전과 본 뒤에 쓸 deployment 관측을 같은 것으로 유지한다. 128개 대상 가운데 probe를 받을 13개의 선정은 prefix 3와 공통 factual evidence에만 의존한다. 따라서 p3와 p4는 한 충돌에 대한 중첩된 정보 집합의 예측이다. 이는 네 번째 측정이 acquisition까지 바꾸던 v37과 다른 통제 조건이다. 평가 대상 물체마다 해당 물체의 선택된 sensor 값만 쓸 수 있고, 다른 물체의 deployment response는 공유하지 않는다. 같은 셀과 환경에서 8 environments × 128 deployment draws를 평가하고 noise 32회를 동일 draws에 반복한다. 4 cells × 8 block을 사용하며, 각 block 내부에 2개 부호와 3개 regime를 짝짓는다. 독립 단위는 서로 다른 seed block이다. 유한 catalog에서 물체를 복원추출하므로 서로 다른 추출 횟수와 서로 다른 prototype 개수를 구분한다.

두 catalog는 calibration과 deployment 용도로 따로 만든 2,048개씩의 latent support이며, evaluator는 catalog의 확률분포와 고정 design factor(cell·alpha)를 안다. 어떤 항목을 실제로 뽑았는지, 실제 orientation·regime·label은 conditioning input이 아니다. Ideal gain의 기본 대상은 이 명시적 finite-population model에서 per-object Bayes predictor가 네 번째 관측으로부터 얻는 평균 Brier 감소 E[(p4−p3)²]다. 이 값은 unknown-law deployment 전체에서의 보편적인 상한이 아니다. Regime를 고정한 표만으로 Bayes 항등식을 검증하지 않고, 명시한 prior로 합친 값을 확인한다.

## 지표와 판정

- 작은 값이 좋은 held-out Brier를 같은 world·noise에서 paired 비교한다. Primary는 3-action minus fixed와 3-action minus binary 두 값에 대한 동시 95% interval이다. Upper endpoint가 0 아래이면 해당 baseline 우위, −0.003 아래이면 기존 실용 기준 통과로 판정한다. 두 기준을 분리한다.
- 각 정책의 네 번째 probe 회수량은 그 정책의 prefix-3 Brier에서 prefix-4 Brier를 뺀 값이다. 다른 정책과의 절대 Brier 차이와 구별한다. Fixed 보정은 의도적으로 네 번째 관측을 쓰지 않으므로 같은 acquisition에서 회수량은 0이다. 분모는 위 ideal gain이다. 비율은 평균들의 비로 계산하고, 분모가 작을 때 절대량과 block bootstrap interval을 우선한다. 음수·100% 초과 비율을 자르지 않는다. 이상적인 prefix-3 예측과 각 정책의 prefix-3 예측이 다르다.
- Bootstrap은 4개 고정 cell마다 독립 block을 재표집해 pairing을 유지한다. Cell·law·regime별 결과는 기술적 보조 분석이다. Practical bound 0.003 미만으로 추정된 ideal gain이 기존 모델의 총 개선 여지 전체를 제한한다는 해석은 피한다.
- GPU p90은 한 장치에서 8 environments × 128 objects 한 호출의 동기화된 host-to-host inference wall time이다. 공통 acquisition, 실제 관측의 decoding, 해당 정책의 보정 선택, 확률 반환을 포함한다. 코드 경로를 먼저 warm up하고 8 paired timing clusters × 3 randomized repeats에 네 정책을 모두 실행한다. 각 방법의 p90, 3-action과 두 baseline 간 p90 비율과 불확실성을 보고한다. 비용 참고 기준은 +2%다. 실물 동작, reset, one-time setup과 reference oracle은 배포 timing에서 제외한다.
- Label을 보는 oracle 진단과 source-only policy 추론을 분리한다. Shared data 및 source artifact, posterior 계산, selected-sensor restriction을 확인하고, raw data로 Brier와 정책 고정·선택을 재검산한다. 추정 오차 또는 코드 수정이 있으면 공개한다.
