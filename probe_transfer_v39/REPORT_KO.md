# v39 · 새 형상에서 언제 continuous gain optimizer를 신뢰할 수 있는가

**같은 네 번의 접촉을 사용했을 때, v38에서 동결한 optimizer는 새 `ridge` 물체에서 두 contact law 모두 고정 32.5% shrinkage를 이겼다. `shelf`에서 고정값이 이길 조건은 사전에 식별하지 못했다.** 두 형상을 합치면 held-out Brier는 0.113638에서 0.113252로 줄었다. 감소량 0.000386은 사전 실용 기준 0.003의 약 13%다. 같은 1,024개 물체의 GPU p90은 1,418.4 ms에서 1,420.8 ms로 약 2.4 ms 증가했다.

### 무엇을 언제 예측했는가

한 환경에서 calibration용 물체를 골라 총 네 번 접촉한 뒤, 같은 환경에 놓인 **별도 물체**가 주어진 충돌에서 미끄러지지 않고 붙는 사건(`stick`)의 확률을 예측한다. 고정 정책은 첫 세 접촉에서 얻은 signed cue를 항상 32.5%만 반영한다. Optimizer는 네 번째 outgoing velocity로 source posterior를 갱신하고, v38의 source gain table에서 예상 Brier가 가장 낮은 보정 비율을 0~100% 사이에서 고른다. 두 방법은 같은 23 mm·+1.10의 마지막 접촉, 같은 25 mm·+0.425의 query, 같은 K256 decoder 및 deployment observation을 받았다. 따라서 두 loss의 차이는 같은 관측을 어떻게 보정에 사용하는지에 대한 추정치다. 네 번째 접촉 자체의 인과적 가치는 이 비교에서 측정하지 않았다. [보정식과 정보 범위](POLICY_KO.md)

v39에서 바꾼 것은 **catalog와 contour**다. 두 기존 contact law를 유지한 채, 마지막 probe와 query 높이에서 local slope가 과거 물체와 가까운 새 shelf·ridge를 설계했다. v38에서 가장 가까운 형상의 조건별 Brier 차이를 v39의 사전 예측으로 삼았다. 정책·예측·형상식·판정 기준의 hash를 **target 생성 전**에 서버에 기록했다. 형상과 law는 이 예측을 작성한 실험자가 아는 조건이며, optimizer에는 주지 않았다. [예측표](PREDICTIONS_KO.md) · [형상 방정식](PHYSICS_KO.md) · [사전 계획](PLAN_KO.md)

각 shape × law에 독립 catalog 12개, catalog당 두 block을 사용했다. Calibration과 deployment의 prototype support를 분리했다. Deployment draw 98,304개가 92,544개 서로 다른 prototype을 사용했다. 각 draw에서 두 표면 방향, 세 가지 query 부근 방향성 강도, 32회 sensor noise를 대응시켜 두 정책을 비교했다. 이 반복을 독립 표본 수에 더하지 않았다. 독립 CPU 구현에서 저장된 query label 불일치는 0건이었고, 별도로 재계산한 Brier의 최대 절대 오차는 5.56×10⁻¹⁷ 미만이었다. [물리 검산](PHYSICS_AUDIT_KO.md) · [Brier 검산](audit39.json)

![사전 예측과 실측 효과. 마름모가 예측, 원이 실제 평균, 굵은 선이 동시 구간, 회색 점 하나가 독립 catalog 하나다.](v39.png)

## 사전 방향과 실측 결과

Δ는 같은 held-out 물체와 noise에서 계산한 **optimizer Brier − fixed 32.5% Brier**다. 음수는 optimizer에 유리하다. Historical Δ는 v39 표본을 만들기 전에 기록한 수치다. 실측 구간은 cell별 독립 catalog 12개를 기준으로 네 대조에 Bonferroni–t 보정을 적용했다.

| 조건 | 사전 Δ | 고정 32.5% Brier | Optimizer Brier | 실측 Δ [동시 95%] | 판정 | Optimizer 우세 catalog |
|---|---:|---:|---:|---:|---|---:|
| shelf / weakening | −0.000003 | 0.113427 | 0.113460 | +0.000033 [−0.000587, +0.000654] | 방향 미확정 | 7/12 |
| shelf / strengthening | +0.000638 | 0.114292 | 0.114110 | −0.000181 [−0.000551, +0.000189] | 방향 미확정 | 8/12 |
| ridge / weakening | −0.000769 | 0.110862 | 0.110384 | −0.000478 [−0.000951, −0.000005] | 사전 방향 확인 | 9/12 |
| ridge / strengthening | −0.001354 | 0.115972 | 0.115052 | −0.000920 [−0.001405, −0.000436] | 사전 방향 확인 | 12/12 |

각 cell에서 12개 catalog 모두를 포함했다. 표의 마지막 열은 한 catalog 안의 두 block을 평균했을 때의 부호다. Catalog 하나를 새로 뽑았을 때의 변동은 cell 평균의 구간보다 크다. [사전 예측 원문](PREDICTIONS_KO.md) · [계획과 SHA-256](freeze39.sha256) · [모든 catalog 추정치](stats39.json)

| 조건 | 새 catalog 두 block 평균의 95% prediction interval¹ | 보정 이동의 제곱 평균 | 기존 잔차와의 정렬 항 |
|---|---:|---:|---:|
| shelf / weakening | [−0.001617, +0.001684] | 0.000989 | +0.000956 |
| shelf / strengthening | [−0.001165, +0.000803] | 0.001135 | +0.001317 |
| ridge / weakening | [−0.001738, +0.000782] | 0.000676 | +0.001154 |
| ridge / strengthening | [−0.002211, +0.000370] | 0.000904 | +0.001824 |

¹ 동일한 생성 분포에서 catalog 하나를 다시 뽑아 두 block으로 측정하는 경우의 정규 근사다. 물리 법칙이 바뀌는 실물 표면에 대한 구간은 아니다. 정렬 항은 2 E[(q_optimizer−q_fixed)(Y−q_fixed)]다. Δ는 이동의 제곱 평균에서 정렬 항을 뺀 값이다. 큰 확률 수정도 기존 잔차와 방향이 맞지 않으면 손해다. 이 항등식은 사후 분해용이며 정책에는 Y가 입력되지 않는다.

네 수치 예측의 MAE는 0.000395이다. 모든 조건에 v38 전체 평균 −0.000372를 넣은 단순 예측은 0.000313, 차이가 없다고 한 예측은 0.000403다. Spearman 순위 상관은 0.80였다. 형상·law를 아는 실험자가 조건별 평균을 예측하는 문제에 대한 네 점의 기술 통계다.

## 전체 성능 및 손실의 단위

전체 fixed Brier는 **0.113638**, optimizer는 **0.113252**였다. 네 cell을 균등 가중한 Δ는 **−0.000386 [−0.000554, −0.000218]**다. 각 cell 안에서 catalog를 최상위 독립 단위로 사용하고 Welch–Satterthwaite 95% interval을 계산했다. 실용적 우위의 사전 기준은 Δ의 상한이 −0.003보다 작은 경우다.

실제 표면의 방향성이 query 부근에서도 유지되는 strong에서 Δ는 −0.000320, 약해지는 weak에서는 −0.000496, 사라지는 local null에서는 −0.000344였다. 각 regime는 같은 latent draw를 공유한다.

Ridge와 shelf의 평균 Δ 차이(ridge−shelf)는 −0.000625 [−0.000961, −0.000289]였다. Law에 따른 형상 대조의 변화, 즉 (ridge strengthening−weakening)−(shelf strengthening−weakening)은 −0.000228 [−0.000900, +0.000444]였다. 두 값은 보조 pointwise 95% 추정치다.

## GPU p90 및 시간 측정 범위

| 실제 경로 | 1,024개 물체 GPU p90 | 95% paired-cluster bootstrap 구간 |
|---|---:|---:|
| Fixed 32.5% | 1,418.4 ms | [1,416.4, 1,419.0] ms |
| Frozen continuous optimizer | 1,420.8 ms | [1,418.9, 1,423.1] ms |

Optimizer/fixed p90 비율은 **1.00170 [1.00050, 1.00334]**였다. 사전에 정한 +2% 허용 상한은 **통과했다**. NVIDIA RTX PRO 6000 Blackwell Server Edition 한 장에서 24개 timing cluster × 3회씩, 두 방법을 무작위 순서로 실행했다. Common acquisition, selected-sensor decoding, source gain 최적화 또는 고정 보정 선택, host 반환을 포함한다. 실제 접촉·reset과 one-time bank 생성은 제외했다. 각 timing 호출의 예측값도 저장된 추론 결과와 대조했다. [Timing·통계](stats39.json) · [측정 코드](run39.py)

## 무엇을 예측할 수 있었는가

**조건 평균.** 가장 큰 이득을 예측한 ridge/strengthening은 실제로 가장 크게 개선됐고 12개 catalog에서 모두 이겼다. Ridge/weakening은 9/12에서 이겼으며 동시 구간의 상한이 −0.000005로 경계에 가깝다. Shelf/strengthening에서는 고정값 승리를 예측했지만 optimizer가 8/12에서 이겼고, 평균 차이의 구간은 양쪽을 포함한다. Shelf/weakening은 사전부터 사실상 동률이었고 실제 차이도 +0.000033에 그쳤다. 사전 부호 적중은 **2/4**, 비교적 확신한 세 조건에선 **2/3**이다. 동시 구간 기준으로는 **확인 2, 반증 0, 미확정 2**다. 따라서 fixed가 유리한 조건을 외부 자료에서 확인했다는 주장은 지지되지 않는다. [조건별 추정치](stats39.json)

**새 catalog 하나의 성패.** 사전 예측의 방향은 48개 catalog 평균 중 32개와 맞았다. 네 조건 모두 새 catalog 하나에 대한 95% prediction interval은 양쪽 부호를 포함했다. 같은 catalog에서 시작 방향과 추출 물체를 바꾼 두 block의 부호가 일치한 경우는 27/48(56.3%, bootstrap 95% 43.8~68.8%)였고, 같은 cell 평균을 제거한 block 간 Pearson r은 0.044였다. 두 block에서는 시작 방향, 추출 물체, sensor noise와 QMC bank seed가 동시에 바뀌므로 각각의 기여도는 추정할 수 없다. 개별 catalog를 골라 optimizer를 켜려면 평균의 재현성보다 더 강한 증거가 필요하다. [반복 검증](repeatability39.json)

**Source가 높게 평가한 보정과 target에서 통하는 보정.** v38 gain table은 공통 fixed 가중치도 후보로 포함하므로 최적화된 값의 source 예상 이득은 음수가 될 수 없다. 이 예상 이득은 네 cell에서 0.000988~0.001439였는데 실제 이득은 −0.000033~+0.000920였다. 특히 shelf/strengthening은 source 이득이 네 조건 중 가장 컸지만 실제 개선은 0.000181이었다. Ridge/strengthening에서는 예측 확률을 움직이는 비용 0.000904보다 기존 잔차에 맞춰 움직인 정렬 항 0.001824가 컸다. 이 분해에는 target label이 들어가므로 deployment에서 얻는 신호와 구별해야 한다. [Source 재계산](audit39.json) · [Target 분해](stats39.json)

### 어떤 보조 정책이 실제 선택을 개선했는가

| 보정 정책 | Held-out Brier | 증거의 범위 |
|---|---:|---|
| 항상 fixed 32.5% | 0.113638 | 주 비교 |
| 항상 frozen continuous optimizer | **0.113252** | 주 비교 |
| 사전에 승리를 예측한 shape × law에만 optimizer | 0.113297 | 사전 cell 이름을 사용, optimizer와의 차이 +0.000045 [−0.000023, +0.000114] |
| 네 번째 관측 후 signed cue와 보정량으로 routing | 0.113339 | v38에서만 fitting, optimizer와 차이 +0.000087 [−0.000048, +0.000222] |
| Fixed에 v38에서 맞춘 global logit intercept를 더함 | 0.113838 | 주 실험 시작 후 target score 열람 전에 동결한 보조 대조 |

관측 기반 router는 52.8%에서 optimizer를 선택했다. 소스에서 얻은 계수의 부호 때문에 평가한 전체 18,874,368개 확률에서 이 규칙은 **두 방법 중 높은 stick 확률을 고르는 규칙과 수치상 동일**했다. Optimizer 선택 집단에서 실제 Δ는 −0.000567, fixed 선택 집단에서도 −0.000185였다. Pooled Spearman ρ는 0.108이었다. 두 집단에서 각 shape × law가 25%씩 차지하도록 맞추면 집단 간 Δ 차이가 −0.000382에서 −0.000141 [−0.000518, +0.000247]로 줄었다. 두 ridge cell에서는 예상한 방향으로 집단을 구분했지만 두 shelf cell에서는 관계가 뒤집혔다. 표면을 넘어 일관되게 작동하는 환경별 gain detector라는 해석은 어렵다. [Router 검증](ROUTING_KO.md)

V38에서 fixed는 stick 확률을 평균 0.01746 낮게 예측했다. 이 자료에서 얻은 단일 logit intercept +0.15982를 v39에 그대로 더했더니 평균 예측오차가 −0.00662에서 +0.01006으로 바뀌고 Brier가 +0.000200 [ +0.000065, +0.000335 ] 악화됐다. 따라서 이번 target에서는 source의 평균 편향을 일괄 보정하는 방법으로 optimizer의 이득을 재현하지 못했다. 다만 조건별 편향을 보정하는 설명은 여전히 가능하다. 별도 router와 intercept의 GPU p90은 측정하지 않았다. [Intercept 분석](INTERCEPT_KO.md) · [보조 계획](INTERCEPT_PLAN_KO.md)

### 같은 물체의 contour만 바꾸면 어떻게 되는가

주 실험이 시작된 뒤 target score를 읽기 전에 설계한 보조 대조다. 48개 catalog에서 사전에 정한 한 block씩을 취해 latent 물체·마찰·기본 curvature·sensor noise를 동일하게 유지하고, contour의 추가 변위만 과거 식과 새 식으로 바꿨다. 두 경우 모두 관측·sensor 선택·충돌 정답을 다시 계산했다. 이 표는 각 catalog의 한 block을 쓰므로 두 block을 평균한 주 추정치와 값이 다르다.

| 조건 | 과거 contour에서 Δ | 새 contour에서 Δ | 새 contour − 과거 contour [pointwise 95%] | Deployment sensor 선택 변경 |
|---|---:|---:|---:|---:|
| shelf / weakening | +0.000457 | +0.000320 | −0.000137 [−0.000324, +0.000051] | 4.9% |
| shelf / strengthening | −0.000440 | −0.000577 | −0.000137 [−0.000357, +0.000083] | 4.9% |
| ridge / weakening | −0.000775 | −0.000521 | +0.000254 [+0.000081, +0.000426] | 8.9% |
| ridge / strengthening | −0.001035 | −0.000614 | +0.000421 [+0.000237, +0.000604] | 9.2% |

**Ridge에서 이긴 원인을 새 곡선 자체로 돌리기는 어렵다.** 같은 물체에서 bead 식을 ridge 식으로 바꾸자 두 law에서 optimizer의 이득은 오히려 0.000254·0.000421 작아졌다. Shelf/strengthening도 같은 v39 물체에 과거 shoulder 식을 적용했을 때부터 optimizer 쪽이었다. 이 표본에서는 새 shelf 곡선 하나만으로 v38 예측과의 부호 차이를 설명할 수 없다. Geometry 교체로 label이 바뀐 비율은 shelf에서 약 1.1%, ridge에서 약 0.35~0.42%였지만 deployment sensor 선택은 각각 약 4.9%, 8.9~9.2% 바뀌었다. 이는 물리와 관측을 함께 바꾼 총효과다. 어느 경로가 Brier 변화를 매개했는지는 별도 실험이 필요하다. [대조 계획](GEOMETRY_CONTROL_PLAN_KO.md) · [48개 paired 결과](geometry39.json) · [관측 공간의 형상 차이](PHYSICS_AUDIT_KO.md)

## 논문에 남길 주장과 다음 판정

| 주장 | 현재의 근거와 판단 |
|---|---|
| 동결한 보정이 지정된 ridge family에서 독립 catalog 평균의 Brier를 줄인다. | 두 law의 사전 방향과 일치했다. 특히 strengthening은 12개 catalog 모두에서 재현됐다. Synthetic contact prediction의 제한된 결과로 본문에 실을 수 있다. |
| 두 local slope가 어느 형상에서든 승패와 이득 크기를 예측한다. | 네 조건의 부호 적중은 2/4이고, 조건별 수치 예측은 단일 source 평균보다 MAE가 컸다. 단순한 slope 대응만으로 일반 법칙을 주장할 근거는 부족하다. |
| 네 번째 관측의 사용 여부를 개별 환경마다 정할 수 있다. | 사전 router가 항상 optimizer를 적용하는 정책을 개선하지 못했다. 네 번째 접촉을 모든 정책이 이미 지불했으므로 접촉 전의 stopping policy도 별도 검증해야 한다. |
| 같은 latency에서 실질적으로 우수한 방법이다. | 같은 장비와 workload에서 +2% 허용 범위는 충족했다. 사전에 정한 Brier 개선 0.003에는 미달했다. Robot cycle time과 실제 조작 성공률은 미측정이다. |

현재 논문 기여로 가장 적합한 부분은 **source expected gain이 잘못된 조건 순위를 줄 수 있음을 보이고, 그 실패를 target label 공개 전 forecast와 독립 catalog에서 계량한 실험**이다. Ridge의 positive result, shelf의 불확실성, 단순 global intercept의 실패를 함께 제시하면 어느 정도의 주장을 재현할 수 있는지 독자가 판단할 수 있다. 물리적 원리의 발견 또는 범용 optimizer의 단독 방법론 논문을 뒷받침하기에는 형상 family 수와 효과 크기가 부족하다.

다음 실험의 우선순위는 **같은 두 접촉 높이의 tangent와 나머지 contour를 독립적으로 바꾸는 held-out shape-family 시험**이다. 같은 tangent에 여러 contour, 같은 contour 범위에 여러 tangent를 배정하고, 같은 latent 물체·surface·noise에서 쌍을 이룬다. 새 contact law 전체를 hold out하고 모든 forecast를 다시 동결한다. 비교 기준에는 v38 전체 평균, fixed 32.5%, 항상 optimizer, source에서만 맞춘 shape-blind calibration을 둔다. 기존 MAE baseline을 이기고, 이득 조건과 손해 조건을 각각 새 family의 동시 구간으로 판별하는지를 1차 판정으로 제안한다. 개별 환경 routing은 항상 optimizer 대비 개선이 독립 family에서 확인된 뒤 논문의 주 기여로 올리는 편이 타당하다.

## 문헌에서 본 v39의 위치

이 검토는 예측 가중치의 일반화, 실험 정보의 이용, 접촉의 관측성이라는 세 질문을 대상으로 했다. 논문별 평가 단위가 다르므로 각 연구가 직접 측정한 결과만 비교했다.

| 원문과 질문 | 방법과 표본 | 결과 및 결론 | v39에 적용할 때 남는 한계 |
|---|---|---|---|
| Yao et al., *Bayesian Hierarchical Stacking*. 두 모델의 우열이 입력에 따라 달라질 때 어떻게 결합하는가? | 여러 predictive distribution의 input-dependent weights를 partial pooling으로 추정하고 이론과 응용 자료에서 평가했다. | 예측 성능이 이질적인 영역에서 가중치가 변하도록 모델링할 수 있음을 보였다. | 원문은 predictive density의 결합을 다룬다. v39는 방향 단서를 log-odds로 얼마나 보정할지 고른다. 연속 가중치를 고른다는 사실만으로 독창성을 주장하기 어렵다. [원문](https://arxiv.org/abs/2101.08954) |
| Koh et al., *WILDS* (ICML 2021). 같은 방법이 다른 수집 환경에서도 작동하는가? | 병원, camera trap, 지역·시기 등 자연적으로 발생하는 shift를 반영한 10개 dataset에서 평가했다. | 통상적인 학습이 distribution shift에서 일반적으로 성능 저하를 보인다는 결과를 보고했다. | v39에서 독립 catalog는 synthetic population의 반복이다. 두 형상·두 law 실험에서 얻은 interval로 실제 재료 분포를 대표할 수 없다. [원문](https://proceedings.mlr.press/v139/koh21a.html) |
| Rossa et al., *ACTION-BED* (NeurIPS 2026, 저자 공개본). 어떤 실험이 이후 손실을 낮추는가? | Joint model에서 생성한 sample과 expected future loss로 design policy와 action policy를 학습하고 source-location(두 source, 30단계, 2,048개 평가 rollout)·dynamics 예제 등에서 평가했다. | Parameter uncertainty 감소 외에도 downstream loss에 직접 맞춘 실험 설계를 구성할 수 있음을 보였다. | v39는 네 번의 동작을 고정한다. 검증 대상은 source에서 학습된 보정의 효과가 어떤 target에서 유지되는가이다. [원문](https://arxiv.org/pdf/2606.23662) |
| Saha et al., *New Evidence, Same Choice* (2026 공개 원고). 물리 증거가 달라지면 다음 선택도 달라지는가? | 6개 VLM, 144개 parameter set의 sliding·bouncing·spring 문제에서 추가 측정을 고르는 선택과 최종 답을 따로 평가했다. | 정답 행동이 바뀐 pair에서도 direct response는 자주 같은 행동을 반복했다. | 선택 변경이 필요한 사례를 만든 선례다. v39에서는 행동 변경률 대신 새 물체의 실제 충돌을 Brier로 채점한다. [원문](https://arxiv.org/html/2609.11022v1) |
| Fazeli et al. 물리식과 실제 충돌이 어느 범위에서 일치하는가? | 실물 planar drop 2,000회 중 품질 검사를 통과한 1,718회에서 6개 rigid-body model에 parameter를 맞추고 post-impact impulse를 비교했다. | 서로 다른 parameter가 비슷한 impact를 설명할 수 있고, 모델이 표현할 수 있는 impulse에도 제약이 있었다. | 두 높이에서 local slope가 가까워도 미지 물체의 correction error까지 가까워지는 물리적 보장은 없다. [원문](https://arxiv.org/pdf/1710.04979) |
| Zhong et al., *ActivePusher* (ICRA 2026). 예측을 보정하고 불확실한 push를 고르면 실제 manipulation이 향상되는가? | Residual physics와 uncertainty로 학습용 push 및 실행할 동작을 고르고 simulation·실제 UR10에서 평가했다. | 더 적은 상호작용으로 모델을 학습하고 여러 단계의 planning success를 개선한 결과를 보고했다. | 네 번의 측정 후 별도 물체의 사건확률을 맞힌 v39와 실물 성공률의 평가 단위가 다르다. [저자 공개 자료와 원문](https://elpislab.org/projects/activepusher/) |
| Gneiting & Raftery. 불확실한 binary outcome의 확률을 어떻게 비교하는가? | Proper scoring rules의 이론을 정리하고 forecast 사례를 분석했다. | True distribution에 최적인 expected score의 조건을 정리했다. | Brier는 다음 충돌에 대한 확률의 품질을 측정한다. 장비 손상이나 slip 대응 행동에 따르는 실제 비용은 별도 효용을 정의해야 한다. [원문](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf) |
| Steyerberg et al. 다른 집단으로 예측기를 옮길 때 복잡한 재학습이 필요한가? | TIMI-II의 3,339명에서 얻은 모델을 GUSTO-I 40,830명 자료에서 평가하며, 크기가 다른 fitting sample로 intercept 또는 slope 보정과 더 큰 수정을 비교했다. | 검증 표본을 200명부터 10,000명까지 바꿔 model updating의 표본 의존성을 평가했다. | 의학적 결과를 충돌에 적용할 근거는 아니다. 보정 이득이 사실상 전체 발생확률의 편향인지 검토하는 single-intercept 대조의 방법론적 선례다. [원문](https://onlinelibrary.wiley.com/doi/10.1002/sim.1844) |

**연구 관계와 해석.** Contact geometry와 friction은 주어진 속도에서 어떤 운동을 관측할지 결정한다. 관측은 source model의 표면 posterior와 signed cue를 바꾸고, 보정량은 두 endpoint 사이에서 예측 확률을 이동시킨다. 그 확률 이동이 실제 잔차의 방향에 맞아야 Brier가 줄어든다. 앞의 물리적 관측성과 끝의 scoring 원리는 근거가 강하지만 **두 local slope만으로 새 환경에서 그 정렬까지 예측할 수 있는가**는 미확정 상태였다. 네 개의 사전 예측은 이 연결에 대한 좁은 검증이다. [Impact model](https://arxiv.org/pdf/1710.04979) · [Input-dependent weighting](https://arxiv.org/abs/2101.08954) · [Proper scores](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf)

**충돌하는 것처럼 보이는 결과.** Partial pooling 연구에서 input-dependent weighting이 도움이 될 수 있다는 결과가 있어도, 같은 source weight가 모든 target shape에서 유리할 필요는 없다. 동일 환경에서 예측기가 상호 보완적인지, 빠져 있던 새 환경에 기존 weight mapping이 옮겨가는지는 다른 가정이다. 실제 로봇의 push planning에서 얻은 개선도 네 번의 접촉과 terminal Brier에 그대로 대응하지 않는다. 이러한 차이를 감안해 네 cell의 성패를 따로 기록하고, 사후 설명과 사전 예측을 구분했다. [Hierarchical stacking](https://arxiv.org/abs/2101.08954) · [WILDS](https://proceedings.mlr.press/v139/koh21a.html) · [ActivePusher](https://elpislab.org/projects/activepusher/)

### 문헌이 남기는 검증 가능한 질문 10개

| 질문과 근거 | 왜 필요한가 | 변수와 후속 검증 |
|---|---|---|
| 1. 두 높이의 slope가 비슷하지만 나머지 contour가 다르면 부호가 유지되는가? [Fazeli et al.](https://arxiv.org/pdf/1710.04979) | 관측이 같은 물리 상태를 유일하게 정하지 않을 수 있다. | Slope-matched shape 여러 개를 생성하고 전체 형상을 단위로 hold out한다. |
| 2. Slope를 연속적으로 바꾸면 손익분기점이 재현되는가? [Fazeli et al.](https://arxiv.org/pdf/1710.04979) | 두 형상만으로 transition curve를 정할 수 없다. | 한 축을 sweep하고 별도 곡선 family에서 crossing을 예측한다. |
| 3. 새 contact law에서도 기하에 따른 부호가 유지되는가? [WILDS](https://proceedings.mlr.press/v139/koh21a.html) | Catalog shift와 mechanism shift는 다르다. | 기존 두 law를 학습에만 두고 rate dependence가 다른 law 전체를 제외해 평가한다. |
| 4. 실물 sensor noise를 대입하면 효과가 줄어드는가? [ActivePusher](https://elpislab.org/projects/activepusher/) | 실제 행동의 불확실성은 동작에 따라 달라질 수 있다. | 속도별 noise·clipping을 측정해 동결한 예측의 Brier를 다시 잰다. |
| 5. 표면 이름 없이 예상 이득이 큰 환경을 고를 수 있는가? [Yao et al.](https://arxiv.org/abs/2101.08954) | Cell 이름을 쓰는 규칙은 미지 표면의 실시간 정책과 다르다. | 허용된 접촉 cue만으로 선택하고 law-blind holdout으로 검증한다. |
| 6. 세 번째 관측에서 이미 네 번째 보정의 성패를 예측할 수 있는가? [Saha et al.](https://arxiv.org/html/2609.11022v1) | 접촉한 뒤의 판단과 접촉할지의 판단은 다른 문제다. | Prefix-3 scorer와 prefix-4 scorer를 별도 모델로 고정하고 비용까지 계산한다. |
| 7. Source gain의 최적값을 축소하면 손해 cell의 악화를 줄이는가? [Hierarchical stacking](https://arxiv.org/abs/2101.08954) | 한 source에서 맞춘 가중치는 prediction error를 옮길 수 있다. | 보정량 shrinkage를 source 안에서 결정하고 독립 catalog의 worst-cell Brier를 측정한다. |
| 8. 어느 sensor가 이미 충분한 정보를 제공하는가? [ACTION-BED](https://arxiv.org/pdf/2606.23662) | 추가 관측의 가치는 이미 얻은 factual evidence에 의존한다. | Deployment query와 probe를 무작위 배정해 correction과의 interaction을 측정한다. |
| 9. 네 cell의 현실적 빈도를 적용하면 정책 순위가 달라지는가? [WILDS](https://proceedings.mlr.press/v139/koh21a.html) | 균등 평균은 배포 시 발생률과 다를 수 있다. | 독립 material cohort에서 prevalence를 구해 같은 frozen 결과를 재가중한다. |
| 10. 작은 Brier 개선이 로봇 행동을 바꾸는가? [Proper scoring rules](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf) · [ActivePusher](https://elpislab.org/projects/activepusher/) | 확률 지표와 실패 비용의 단위가 다르다. | Slip 실패 비용과 동작 threshold를 사전에 정하고 regret·cycle time을 실측한다. |

## 프로토콜과 자료를 재검산하는 방법

서버 경로는 `datacenter3:/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/probe_transfer_v39`다. 예측표와 정의·코드 8개의 SHA-256을 남기기 직전 서버 결과 디렉터리가 없음을 검사했다. SHA-256 목록 자체의 digest는 `be0a1fbbcec10f8117ef2488be64229b74addda7ec855f3ec8694b2d47b91b52`이고 동결 시각은 **2026-11-15 08:56:40 UTC (17:56:40 KST)**다. Worker는 해당 목록과 기존 v38 dependency hash까지 검사한 후 실행됐다. [계획](PLAN_KO.md) · [동결 기록](freeze39.sha256) · [시각](frozen_utc.txt) · [예측 계수](forecast39.json)

주 실험에는 NVIDIA RTX PRO 6000 Blackwell Server Edition 한 장을 사용했다. Slurm allocation 2744468의 node55에서 process와 GPU memory·utilization을 확인한 후 추론을 실행했다. 이미 다른 작업이 돌아가던 allocation은 그대로 유지했다. 독립 통계·물리 검산은 node56의 allocation 2744471에서 CUDA를 끈 CPU로 수행했다. 각 block은 별도 NPZ와 hash를 남긴다. [실행 metadata](results/run.json) · [96개 block 완료 기록](results/completion.json) · [주 worker 종료](monitor/results/worker_exit.json) · [보조 geometry worker 종료](monitor_geometry/results/worker_exit.json)

같은 부모 프로젝트의 빈 할당 GPU에서 실행하는 명령은 아래와 같다. 원본 자료는 보존하고, 새로 실행할 때는 생성물이 없는 별도 작업 사본을 사용한다. 기존 완료 block의 경우 hash가 일치할 때만 재사용한다.

```bash
sha256sum -c freeze39.sha256
srun --overlap --jobid="$GPU_ALLOCATION" -n1 bash run39.sh
```

서버의 기존 torch 환경을 이용해 CPU에서 주 표와 그림을 다시 만들 수 있다.

```bash
export CUDA_VISIBLE_DEVICES='' BELIEF_DEVICE=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
py=../continuous_v2/.venv/bin/python
"$py" stats39.py
"$py" tables39.py
"$py" figure39.py
```

[추론과 타이밍](run39.py) · [Target 생성 방정식](PHYSICS_KO.md) · [정책과 정보 범위](POLICY_KO.md) · [독립 Brier 검산 코드](audit39.py) · [물리 및 표본 검산 코드](audit_physics39.py) · [보조 geometry 대조](geometry_control39.py)
