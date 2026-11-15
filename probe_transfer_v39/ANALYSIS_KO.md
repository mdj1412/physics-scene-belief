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
