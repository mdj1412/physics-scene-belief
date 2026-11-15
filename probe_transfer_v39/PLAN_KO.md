# v39 사전 검증 계획

2026-11-15 KST. v39의 target catalog를 실행하기 전에 결정한 설계와 판정 기준이다. 한 물체에서 측정한 뒤 같은 환경의 다른 물체가 다음 충돌에서 stick할 확률을 예측한다. 최종 관심은 제한된 표본에서 최선의 수치를 다시 고르는 것이 아니라, 기존 source 예측과 v38의 조건별 성패가 새 형상·catalog에 얼마나 전달되는가이다.

## 정책과 표본

- 비교 방법은 v38의 `continuous38.source_optimum`과 기존 fixed 32.5% shrinkage다. β, z, source gain table, grid interpolation, tie break를 고정한다. 둘 다 같은 signed cue, 같은 prefix-3 deployment acquisition, 같은 base physics probability, 같은 선택된 deployment sensor를 받는다. Calibration은 fixed 4, 네 번째 접촉은 23 mm에서 +1.10, 예측할 충돌은 25 mm에서 +0.425다.
- 신규 형상 shelf와 ridge를 v38의 weakening, strengthening contact law에 각각 교차한다. 동일 law를 유지해 형상과 catalog가 바뀌는 질문을 시험한다. 네 고정 cell의 비교는 사전 정의된 두 형상 및 두 법칙에 한정된다.
- 각 cell에 독립 catalog panel 12세트를 만들고 각 세트에서 두 seed block을 뽑아 시작 방향을 균형화한다. 각 block에서 8 환경 × 128 별도 deployment 추출 × 32 noise repeat, 매 환경 13개 deployment probe를 쓴다. Orientation 두 방향 및 세 near-field regime은 paired다. 환경·물체·noise draw를 독립 catalog처럼 세지 않는다. 각 catalog panel 안의 calibration과 deployment support는 분리한다. 표면 방향과 regime 비중은 1/2와 1/3, 네 cell의 가중치는 같다.

## 사전에 낸 예측과 검증

- Δ = Brier(continuous) − Brier(fixed). 음수는 continuous의 이득이다. 네 cell 각각에 대해 v39의 물리 결과를 열기 전에 sign·수치·강도와 근거를 별도 [예측표](PREDICTIONS_KO.md)에 기록한다. 동결한 예측은 실험자가 지정한 형상·law 조건에 대한 것이다. 조건 이름을 모르는 배포에서 자동 분류할 수 있다고 가정하지 않는다.
- 각 cell에서 12개 catalog 평균의 평균과 표준편차를 사용한다. 두 block·모든 paired 환경은 catalog 안에서 먼저 평균한다. 네 가지 전체를 동시에 판정하기 위해 cell별 양측 Bonferroni–t 95% familywise interval(df=11)을 사전 지정한다. 예측 부호 쪽에 구간 전체가 있으면 확인, 반대편이면 반증, 0에 걸치면 미확정이다. Point sign 적중 수와 통계적으로 확인된 수는 각각 보고한다. 실패를 단순한 유의성 부재와 혼동하지 않는다.
- 전체 평균 Δ의 불확실성은 네 고정 cell, cell 안의 독립 catalog로부터 Welch–Satterthwaite t 95% interval을 계산한다. Upper endpoint <0이면 통계적 우위, <−0.003이면 기존 실용 기준 충족이다. 가장 유리한 cell이나 regime를 나중에 골라 전체 우위를 주장하지 않는다.
- 형상별 수치 예측의 평균절대오차(MAE), ordering 및 v38 전체 평균 Δ를 모든 cell에 쓰는 단순 forecast와의 비교를 보조 지표로 기록한다. 네 개의 고정 조건에서 조합 24가지만 가능한 ranking이므로 broad statistical generalization을 하지 않는다. 각 catalog에서 예측 부호가 맞은 횟수와 new-catalog predictive interval(정규 근사)을 제시한다.
- 예상 이득이 음수인 사전 cell에서는 continuous, 나머지는 fixed를 쓰는 가정적 선택과 양쪽을 항상 쓰는 두 기준을 paired Brier로 비교한다. 이 규칙에는 cell 정보가 필요하다. 별도의 label-blind 규칙이 있으면 v38 자료만으로 계수·판정을 동결한다. 여기서 네 번째 관측으로 계산한 z·weight를 쓰는 예측과 probe를 사기 전의 예측을 구별한다.
- 각 형상·law별 signed 효과와 pooled 효과가 일치하지 않을 수 있다. 보조적 진단에는 regime·orientation별 Δ와 Brier 분해를 쓴다. 어떤 보정 방향이 옳았는지 정답으로 판단한 값은 실현된 성과를 분석하는 용도로만 쓴다.

## Compute 비용과 검산

한 장의 실제 유휴 GPU에서 8×128 물체의 시간당 처리 단위가 동일한 full-path GPU p90을 구한다. Input부터 공통 acquisition, selected-sensor decoding, frozen continuous 보정량 최적화 또는 fixed 선택, 반환까지 동기화해 잰다. Bank 구축, 센서 측정·reset은 별개다. 4 cells × 짝수 catalog 6개, block 0에서 같은 workload의 두 방법 순서를 섞어 warmup 후 각각 3회 측정한다. Paired timing cluster를 cell 안에서 재표집해 p90 ratio의 95% interval을 구한다. Upper endpoint ≤1.02이면 +2% 허용 기준을 통과한 것으로 판정한다.

서버에 계획·예측·코드의 SHA-256과 UTC 시각을 기록한 후 GPU를 실행한다. 이후 원자료로 두 Brier를 독립적으로 복원하고 catalog split, 수치 범위, 선택된 관측 수, 실행 순서, worker 종료를 점검한다. 한글 보고서에는 실물 데이터 부재, 새 형상 수, source와 target의 차이, 사후 진단의 범위를 명시한다.
