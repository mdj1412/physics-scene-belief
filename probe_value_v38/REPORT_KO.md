# v38 · 마지막 접촉이 주는 정보와 충돌 예측에서 회수한 이득

네 번 중 마지막 접촉을 **높이 23 mm, 속도 +1.10**에 고정했다. 새로운 두 contact law와 두 물체 형상에서, 이 한 번의 측정이 추가로 줄일 수 있는 이상적인 Brier loss는 **0.000913**이었다(95% interval **0.000818~0.001007**). 형상과 법칙의 이름도 가린 보조 계산에서는 **0.001040**이었다. 두 추정치 모두 이 연구에서 미리 정한 실용 기준 0.003보다 작았다. 이 기준은 비교를 위해 설정한 Brier 단위이며 실제 로봇의 비용에서 환산한 값은 아니다. [사전 계획](PLAN_KO.md) · [Bayes reference](oracle_stats38.json) · [형상·법칙을 가린 결과](marginal_stats38.json)

**사전 지정한 0%·32.5%·100% 정책은 binary gate를 재현성 있게 앞섰지만, 고정 32.5%보다 낮은 loss를 재현하지 못했다.** 같은 1,024개 물체를 처리하는 GPU p90은 각각 1,438.0 ms, 1,441.1 ms, 1,437.4 ms였다. 보조 실험에서는 같은 frozen source gain table에서 보정량을 연속적으로 선택한 방법이 가장 낮은 Brier를 얻었다. 그 방법의 고정값 대비 차이도 −0.000372에 머물렀다. [주 결과](policy_stats38.json) · [연속 보정 보조 결과](continuous38.json)

![두 주 대조, 네 번째 관측의 이득, 네 형상·법칙 조합의 차이](v38.png)

그림 A는 두 주 비교의 동시 95% interval, B는 각각의 pointwise 95% interval이다. B의 Bayes reference는 평균 제곱 확률 변화이며 나머지 점은 각 정책에서 실제 label로 채점한 prefix-3 대비 Brier 감소다. C는 네 고정 cell의 평균이다.

## 1. 네 정책에 정확히 무엇을 같게 주었나

같은 환경의 calibration 물체 네 개에 접촉한 뒤, **별도로 추출한 물체의 다음 충돌에서 stick할 확률**을 예측했다. 예측할 충돌은 높이 25 mm, 속도 +0.425다. 먼저 세 번의 속도 부호를 교대해 표면의 방향 단서 z를 얻었다. 네 번째 물체의 factual geometry는 두 예측 모두 알고, 네 번째 접촉에서 얻은 noisy outgoing velocity만 한쪽에 추가했다. 이후 어느 물체에 deployment probe나 query view를 제공할지는 처음 세 관측만으로 정해 모든 방법에 동일하게 적용했다. 그래야 네 번째 접촉이 주는 정보와 이후 달라진 관측의 효과를 분리할 수 있다. Fixed 정책도 접촉 예산은 네 번이며, 마지막 velocity를 보정에 쓰지 않는다. 속도는 simulator의 정규화된 수치다. [정책의 입력 및 계수](POLICY_KO.md)

| 정책 | 처음 세 관측과 마지막 관측을 사용하는 방법 |
|---|---|
| Fixed 32.5% | 동일한 z의 32.5%를 항상 적용한다. |
| Binary gate | Source에서 예측한 full correction의 Brier 이득이 양수일 때만 100%를 적용한다. |
| **3-action policy** | 같은 source table에서 0%, 32.5%, 100%의 이득을 읽고 가장 큰 것을 고른다. 동률이면 32.5%를 우선한다. v37의 사후 발견을 v38 주 가설로 동결했다. |
| Continuous posterior mean | Source posterior가 추정한 query 부근 방향 강도의 평균을 보정량으로 사용한다. 표면마다 [0, 1]의 실숫값을 낼 수 있다. |
| Continuous gain optimizer · 보조 | 같은 frozen table에서 보정량 w∈[0,1]의 expected Brier gain을 직접 최대화한다. 선형 보간의 양 끝과 내부 knot를 검사한다. 보조 계획은 target 점수를 열기 전에 작성했다. [보조 설계](CONTINUOUS_PLAN_KO.md) |

Target에는 rational shoulder와 Gaussian bead 형상을 사용했다. 두 surface law 모두 충돌에서 stick을 결정하는 static coefficient와 slip 뒤의 kinetic coefficient가 다르다. 한 법칙에서는 속도에 따라 kinetic friction이 약해지고, 다른 법칙에서는 강해진다. 기존 source에 없는 형상과 함수형이다. 완전히 자유로운 미지의 표면을 포괄하지는 않는다. Orientation 두 방향, query 근처의 방향 강도 strong·weak·local null은 source와 개념을 공유한다. [방정식과 센서 분포](PHYSICS_KO.md)

네 개 고정 cell마다 독립 seed block 8개, block당 8개 환경·128개 deployment draws·32개 observation-noise repeats를 썼다. 여섯 표면 조건은 같은 draw에 대응시켰다. Calibration과 deployment는 완전히 분리된 2,048개씩의 prototype catalog에서 복원추출했다. 총 32,768회 deployment draw 중 서로 다른 prototype은 25,816개다. Bootstrap은 cell과 첫 접촉 방향을 보존하며 seed block을 재표집했다. 따라서 interval은 이 네 cell과 동결한 catalog에 대한 불확실성이다. [물리·표본 검산](physical_summary.json)

## 2. Held-out Brier와 네 번째 probe의 회수량

아래 절대 loss는 모두 **새 물체의 실제 held-out stick label**로 계산했다. 감소량은 각 정책 자신의 세 관측 Brier에서 네 관측 Brier를 뺀 값이다. 마지막 열은 이 감소량을 공통 ideal gain으로 나눈 비율이다. 비율의 interval은 분자와 분모에 같은 block을 재표집해 구했다.

| 정책 | 3관측 Brier | 4관측 Brier ↓ | 실제 Brier 감소 [95%] | Ideal 대비 비율 [95%] |
|---|---:|---:|---:|---:|
| Fixed 32.5% | 0.115953 | 0.115953 | 0.000000 | 0% |
| Binary gate | 0.117476 | 0.116202 | +0.001274 [+0.000878, +0.001656] | 139.6% [95.2%, 186.3%] |
| **3-action policy** | **0.115948** | **0.115754** | **+0.000194 [−0.000116, +0.000515]** | **21.3% [−12.6%, 57.5%]** |
| Continuous posterior mean | 0.115966 | 0.116029 | −0.000062 [−0.000194, +0.000063] | −6.8% [−22.2%, 6.6%] |
| Continuous gain optimizer · 보조 | 0.115957 | 0.115581 | +0.000376 [+0.000102, +0.000655] | 41.2% [11.0%, 73.6%] |

**Binary의 139.6%는 Bayes predictor보다 정확하다는 뜻으로 해석할 수 없다.** Binary는 관측 세 개일 때 Brier가 0.117476으로 네 방법 중 가장 높았다. 네 번째 정보가 자체 보정 오류까지 줄이는 데 쓰였다. 각 정보 집합의 최적 예측을 p₃, p₄, 정책 예측을 q₃, q₄, 평균 squared risk를 R이라고 두면 다음 관계가 성립한다.

> R(q₃) − R(q₄) = E[(p₄ − p₃)²] + E[(q₃ − p₃)²] − E[(q₄ − p₄)²].

따라서 이 비율은 정책이 자신의 출발점에서 얼마나 개선됐는지를 비교하는 지표다. 같은 관측에서 label의 조건부 평균 p₄로 다시 계산한 Rao–Blackwellized 비율은 binary 148.2%, 3-action 19.0%, posterior mean −1.6%, 보조 continuous optimizer 47.3%였다. 실제 label로 얻은 각 점추정치와 차이가 있지만 동일한 경향이다. [Bayes 수치와 모든 interval](oracle_stats38.json) · [보조 정책의 회수율](continuous_recovery38.json)

### 고정값과 binary를 이겼는가

| 네 관측 후의 대조 | Brier 차이 · 음수가 앞 정책에 유리 | 판정 |
|---|---:|---|
| **3-action − fixed 32.5%** | **−0.000199 [−0.000557, +0.000158]** | 두 공동 주 비교의 동시 95% 구간이 0을 포함했다. 고정값 우위도 입증되지 않았다. |
| **3-action − binary** | **−0.000449 [−0.000638, −0.000260]** | 동시 95% 구간 전체가 0 미만이다. 0.003의 실용 기준에는 미치지 못했다. |
| 3-action − continuous posterior mean | −0.000275 [−0.000665, +0.000101] | 보조 pointwise 비교. 우열 불확실. |
| Continuous gain optimizer − fixed 32.5% | −0.000372 [−0.000645, −0.000105] | 보조 pointwise 비교. 같은 source table 안에서 연속 강도를 허용한 결과다. |
| Continuous gain optimizer − 3-action | −0.000173 [−0.000260, −0.000088] | 보조 pointwise 비교. 효과는 작으며 새 catalog에서 독립 재현이 남았다. |

v37의 +1.10에서 사후 3-action − fixed는 −0.00080이었다. v38은 방법을 사전 지정했지만, 강한 고정값을 이기는 효과가 같은 크기로 재현되지는 않았다. Contact law·shape·catalog와 deployment acquisition 조건이 함께 달라졌으므로 그 감소를 어느 한 요소 탓으로 단정할 수 없다. 같은 소스의 두 연속 보정이 서로 달랐던 점도 주목할 만하다. Near-field strength의 posterior mean은 물성의 평균을 추정하고 gain optimizer는 예측 손실을 최소화할 보정 강도를 고른다. [원자료 대조](policy_stats38.json) · [연속 보정 대조](continuous38.json)

## 3. 이상적인 Brier gain은 어떻게 검증했나

조건을 아는 evaluator가 실제 물체의 index나 정답을 읽으면 네 번째 접촉의 가치를 측정할 수 없다. 여기서는 각 환경의 여섯 hidden surface state와 독립 catalog의 2,048개 prototype을 Gaussian sensor likelihood로 모두 합산해, 해당 물체가 stick할 확률을 관측 전후로 각각 계산했다. 이 **finite-model posterior는 정확히 합산**하고 그 확률이 실제로 얼마나 바뀌는지의 **평균은 held-out sample로 추정**했다. 다른 deployment 물체의 유료 측정값은 사용하지 않았다. [수식과 독립 구현](ORACLE_KO.md)

| 평가 방법 | 추정 결과 | 답하는 질문 |
|---|---:|---|
| Cell·α를 아는 ideal, 전체 32 noise repeats | E[(p₄−p₃)²] = **0.000913 [0.000818, 0.001007]** | 같은 정보에서 최적으로 예측하면 이 한 측정이 평균적으로 얼마만큼 가치 있는가? |
| 같은 ideal의 실제 label Brier | **0.101321 → 0.100344**, 감소 **0.000977 [0.000680, 0.001265]** | 실제 충돌 정답을 쓰는 별도 추정도 같은 크기를 보이는가? 두 추정치 차이의 95% 구간은 [−0.000278, +0.000379]다. |
| 형상·법칙을 가린 ideal, 32 blocks에서 첫 noise repeat | **0.001040 [0.000904, 0.001186]** | 평가기도 어느 cell인지 모르면 얼마인가? 같은 noise에서 cell을 알려 준 값은 0.000914다. 차이 +0.000126 [0.000066, 0.000194]. |

마지막 계산은 cell 4개 × orientation 2개 × regime 3개를 잠재상태로 두고, calibration factual pool 1,024개와 deployment factual rows 128개, 해당 물체의 선택된 sensor를 정확히 한 번씩 반영했다. 네 개 후보 law·형상과 α, catalog 분포 자체는 여전히 알려져 있다. 초기 불확실성이 더 클 때 네 번째 접촉이 해소할 정보도 늘 수 있으므로, cell을 가린 뒤 추가 이득이 커진 것은 모순이 아니다. 첫 noise repeat만 쓰는 별도 분석이어서 두 조건의 비교도 같은 repeat로 맞췄다. [전체 cell-blind 검산](marginal_stats38.json)

세 관측 기준과 네 관측 기준의 true-law 예측은 source-only policy보다 모두 낮은 절대 loss를 보였다. 네 관측의 ideal 0.100344와 3-action 0.115754 사이에는 약 **0.0154**의 차이가 있다. 이 차이는 evaluator가 정확한 생성 법칙과 catalog를 아는 이점 등을 포함한다. 추가 접촉 하나로 회수할 수 있는 양으로 해석할 수 없지만, 같은 수의 관측을 더 잘 해석하는 방법을 연구할 이유는 충분하다. 알려진 catalog에서도 선택된 센서에 대한 effective prototype 수는 query view 약 20.6, deployment probe 약 28.0이었다. [Oracle 진단](oracle_stats38.json)

**Calibration 검산에는 남는 불확실성이 있다.** 주 표본에서 p₃의 평균 예측은 실제 발생률보다 0.00286 낮았다. 사전 block-bootstrap 구간은 0을 제외했고, 작은 표본을 고려한 stratified t 구간은 [−0.00582, +0.00010]이었다. 평균 p₄−p₃도 +0.00166으로 pointwise t 구간이 0을 제외했다. 독립 구현으로 새로운 32,768개 환경을 만들고 deployment sensor를 무작위 배정해 확인했을 때는, 예측·발생률 차이와 gain 항등식을 포함한 네 잔차의 동시 95% 구간이 모두 0을 포함했다. 이 별도 검산에서의 ideal gain은 0.001006이었다. 잔차가 없다고 단정하거나 원래 acquisition 조건의 결과를 이 값으로 대체하지 않았다. [원 표본 검산](calibration_audit38.json) · [새 환경 검산](stress_oracle38.json)

## 4. 어느 환경에서 정책 선택이 유용했나

| 형상 / 법칙 | Ideal gain | 3-action − fixed Brier | 3-action 자체의 네 번째 관측 gain |
|---|---:|---:|---:|
| Shoulder / weakening | 0.000641 | +0.000146 | −0.000111 |
| Shoulder / strengthening | 0.001471 | +0.000921 | −0.000908 |
| Bead / weakening | 0.000494 | −0.000463 | +0.000459 |
| Bead / strengthening | 0.001045 | −0.001401 | +0.001337 |

**가장 정보가 많은 cell에서 frozen 3-action의 고정값 대비 성능은 오히려 가장 나빴다.** Shoulder/strengthening은 네 cell 중 ideal gain이 가장 컸지만, 정책은 네 번째 관측을 사용한 뒤 손해를 보았다. 같은 strengthening에서 bead는 이득을 얻었다. 이 네 cell의 기술적 대조는 단순히 “정보가 없는 문제”라고 해석하기 어렵게 한다. 동시에 형상 이름만 알면 보정에 성공한다는 일반화도 보장하지 않는다. 형상 효과와 catalog realization을 분리하려면 새 catalog seed와 새 형상이 필요하다. [Cell별 원자료](policy_stats38.json) · [Cell별 ideal](oracle_stats38.json)

3-action은 네 번째 접촉 뒤 **69.5% [66.4%, 72.7%]**의 환경에서 선택을 바꿨다. 최종 선택은 off 41.5%, fixed 25.1%, full 33.4%였다. 전체 평균에서 full을 선택한 사례가 고정 대비 −0.000333을, off가 +0.000133을 기여해 일부 상쇄됐다. Local null에서는 off의 개선 −0.000710과 full의 손해 +0.000689가 거의 상쇄됐다. 각 값은 그 행동을 고른 집단 비중까지 반영한 기술량이다. 선택된 집단끼리의 차이를 무작위 배정된 행동의 causal effect로 볼 수는 없다. [행동 전환과 손실 분해](ROUTING_KO.md)

## 5. 보정 선택을 포함한 GPU p90

| 네 관측 후 방법 | 1,024개 물체 한 batch의 GPU p90 |
|---|---:|
| Fixed 32.5% | 1,437.4 ms |
| Binary gate | 1,441.1 ms |
| 3-action policy | 1,438.0 ms |
| Continuous posterior mean | 1,438.6 ms |
| Continuous gain optimizer · 별도 paired session | 1,433.5 ms · 같은 session의 3-action 1,431.5 ms |

NVIDIA RTX PRO 6000 Blackwell Server Edition 한 장에서 common acquisition부터 선택된 sensor decoding, policy별 보정 선택, host 반환까지 측정했다. 8 timing clusters × 3 randomized repeats, 각 cluster의 warmup을 제외했다. 주 session에서 3-action/fixed p90 ratio는 **1.00043 [0.99764, 1.00321]**, 3-action/binary는 **0.99788 [0.99669, 0.99907]**이었다(두 비교 동시 95%). 사전 기준인 +2% 이내였다. 보조 session의 optimizer/3-action ratio는 **1.00137 [0.99537, 1.00322]**였다. 속도 차이의 실용적 중요성은 작다. 서로 다른 session의 raw p90 순위는 해석하지 않았다. GPU kernel 시간만 재는 지표와 달리 host의 보정 선택도 포함하며, 실제 장치 접촉 및 reset·초기 bank 생성·oracle 비용은 제외했다. [Primary timing](policy_stats38.json) · [같은 session 내 보조 timing](continuous38.json)

## 6. 논문 기여와 판단

**방어할 수 있는 기여는 추가 접촉의 정보가 적었던 경우와, 정보가 있었는데 source policy가 잘못 사용한 경우를 한 실험에서 구분했다는 점이다.** 특히 shoulder/strengthening과 bead/strengthening 대조는 “더 많이 미끄러뜨리면 된다” 또는 “gate를 부드럽게 만들면 된다”보다 구체적인 후속 가설을 준다. 일반적인 decision-aware acquisition과 conditional weighting에는 선례가 명확하다. 이 실험은 simulator 안에서의 controlled diagnostic으로 기여할 수 있으며, 실물 마찰의 보편 법칙이나 3-action의 일반적 우위를 뒷받침하지 않는다.

**다음 한 실험을 고른다면** 새 catalog와 독립 shape family를 먼저 고정하고, source에서만 학습한 continuous gain optimizer와 고정 shrinkage가 어느 쪽에서 이길지를 **target label을 열기 전에** 예측하겠다. 최종 Brier와 calibration, 셀을 가린 Bayes gain을 함께 채점한다. 성능 차이가 계속 0.003보다 작다면, 네 번째 접촉 횟수를 늘리기보다 같은 관측의 decoder misspecification 또는 force·velocity sensor 정보 차이를 다루는 쪽이 더 설득력 있다. 실제 장비의 접촉 비용과 행동 성공률은 별도 데이터가 필요하다. [선행연구 비교와 후속 질문](LITERATURE_KO.md)

## 선행연구와 v38의 기여 범위

v37의 차별성 서베이 중 v38의 해석을 직접 바꾸는 원문을 다시 대조했다. 물성의 식별, 정보를 얻는 실험의 선택, 얻은 정보로 내리는 예측을 각각 무엇으로 채점했는지가 선정 기준이다.

| 연구 질문 | 방법과 평가 대상 | 저자들이 확인한 결과 | v38과의 관련성과 한계 |
|---|---|---|---|
| 물리적 질문에 답하기 위해 어떤 추가 실험이 필요한가? | Saha et al.은 sliding·bouncing·spring의 144개 parameter set에서 6개 VLM을 평가했다. 관측이나 질문만 바꾼 pair에 알려진 정답 실험이 있다. | Direct response는 정답 행동이 다른 image pair의 95.1–100%에서 같은 행동을 반복했다. 측정의 선택과 주어진 측정의 사용을 별도로 평가했다. | 획득과 이용의 분리에는 선례가 있다. 이 benchmark의 답·행동 정확도와 다른 물체의 noisy impact에 대한 Brier gain은 평가 대상이 다르다. [Saha et al., 2026 공개 원고](https://arxiv.org/html/2609.11022v1) |
| 측정의 목적을 미래 행동의 손실로 직접 정의할 수 있는가? | Rossa et al.은 design policy와 action policy의 expected future loss를 함께 학습했다. Source location과 stochastic pendulum 등에서 비교했다. | Joint samples에서 downstream loss를 최적화하는 방법을 유도하고 task별 실험에서 성능을 평가했다. | v38은 한 속도를 사전 고정했다. 새 optimal design algorithm의 검증보다 주어진 probe의 정보량과 frozen policy가 그중 회수한 양을 측정하는 데 해당한다. [ACTION-BED, 2026 공개 원고](https://arxiv.org/pdf/2606.23662) |
| 실제 충돌에서 rigid contact model이 설명하는 범위는 어디까지인가? | Fazeli et al.은 실물 impact data에서 여섯 rigid-body model의 parameter를 추정하고 impulse 예측 범위를 평가했다. | Sticking regime에서 여러 parameter가 같은 결과를 내며, 모델 family의 표현 범위에도 한계가 있었다. | 알려진 finite catalog와 정확한 noise distribution을 전제로 한 v38 Bayes reference는 그 catalog에서만 정확하다. 실물 friction의 정보 상한을 뜻하지 않는다. [Fazeli et al.](https://arxiv.org/pdf/1710.04979) |
| 실제 물체에서 sliding 관측을 물성 추정에 쓸 수 있는가? | Arslan Waltersson과 Karayiannidis는 gripper에서 force와 sliding velocity를 측정해 static·Coulomb friction과 contact radius를 추정했다. 실제 네 물체를 사용했다. | Sliding 중 online estimation을 시연하고 빠른 slip–stick에서 추정이 어려운 경우를 다뤘다. | Force·velocity time series의 높은 관측 밀도와 한 번당 noisy outgoing velocity만 주는 우리 조건의 차이가 성능 차이를 설명할 수 있다. [ISER 2025](https://arxiv.org/pdf/2507.06824) |
| 입력별로 예측 모델을 섞는 게 고정 가중치보다 나은가? | Yao et al.은 data-dependent partially pooled weights를 이론과 응용 예제에서 조사했다. | 모델별 예측 성능의 상대 순위가 입력에 따라 다를 때, 고정 결합을 유연한 결합으로 확장할 수 있음을 보였다. | Predictive distribution의 pooling과 v38의 log-odds 보정은 수식이 다르다. 두 후보 사이에서 가중치를 고른다는 아이디어 자체에는 선례가 있다. [Bayesian Hierarchical Stacking](https://arxiv.org/abs/2101.08954) |
| 다음 사건의 확률 예측을 무엇으로 채점해야 하는가? | Gneiting과 Raftery는 categorical forecast를 포함해 proper scoring rule을 이론적으로 분석하고 예측 사례를 다뤘다. | True distribution을 예보할 때 expected score가 최적이 되는 조건을 정리했다. | Binary squared loss에서는 conditional mean이 최적이다. Brier 차이와 실제 로봇 행동의 이익은 서로 다른 단위다. [JASA 원문](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf) |

**연구 관계와 상반돼 보이는 결과.** 실물 gripper의 friction 식별과 v38의 held-out Brier는 측정 장비, 표본, 목표가 모두 다르다. 한 표면에서 여러 force·velocity를 얻어 물성을 설명하기 쉬워도, 같은 표면의 다른 물체 하나가 stick할지는 이미 확보한 geometry와 deployment observation으로 상당 부분 정해질 수 있다. 이 경우 물성의 posterior가 크게 변해도 추가 Brier gain은 작다. 반대로 source decoder의 체계적 오차를 상쇄한 정책은 자기 자신의 약한 prefix-3 기준에서 크게 개선될 수도 있다. v38의 설계는 이들 설명을 비교하지만 실물 인과효과를 식별하지는 않는다. [물성 식별](https://arxiv.org/pdf/2507.06824) · [Decision-aware design](https://arxiv.org/pdf/2606.23662) · [Proper scores](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf)

**지식 지도.** Contact law·geometry → 관측 가능한 velocity → 가능한 sign·표면 강도의 posterior → 다른 물체의 stick probability → held-out Brier 순으로 연결된다. 물리 모델의 비식별성과 Brier의 최적 예측기는 근거가 확립된 부분이다. 선택 가능한 보정의 폭과 target shift가 손실을 얼마나 바꾸는지는 경험적인 질문이다. 실물에서 한 번의 추가 접촉이 가치 있는지에는 reset 비용과 그 확률로 바꿀 행동이 더 필요하다. [Impact model](https://arxiv.org/pdf/1710.04979) · [실험 선택과 결과 사용](https://arxiv.org/html/2609.11022v1)

### 문헌과 실험에서 이어지는 10개 연구 질문

| 질문 | 왜 필요한가 | 검증할 변수와 방법 |
|---|---|---|
| 1. 새 catalog에서도 순위가 재현되는가? | Finite catalog 내 정확성과 새 population으로의 일반화는 별개다. | Catalog seed를 최상위 독립 단위로 재추출하고 within/between-catalog variance를 분리한다. |
| 2. Continuous population에서도 같은 ideal gain인가? | 알려진 유한 support가 물체 식별을 쉽게 할 수 있다. | Support 크기를 늘려 ESS, posterior predictive gain의 수렴을 측정한다. |
| 3. Regime 비율을 바꾸면 fixed가 어느 정도 유리한가? | Source에서 정한 shrinkage는 구성 비율에 의존한다. | Train prior를 고정하고 연속적인 deployment prevalence에서 Brier를 비교한다. |
| 4. 두 힘의 측정을 더하면 한 속도에서도 충분한가? | 실제 friction 추정은 force와 velocity를 함께 활용한다. | 같은 접촉 수로 normal/tangential force를 별도 sensor channel로 추가한다. |
| 5. 속도에 따라 noise가 바뀌어도 posterior가 유용한가? | 실물의 빠른 slip–stick에서 측정 난도가 달라진다. | Speed-specific noise·clipping 아래에서 calibrated likelihood와 고정 likelihood를 비교한다. |
| 6. 한 표면에서 다른 위치로 갈 때 방향이 반전하면 어떻게 되는가? | 현재 near-field strength 세 값에는 sign reversal이 없다. | 연속 위치·반전 길이를 바꾸고 지역별 gain을 예측한다. |
| 7. 두 개의 deployment sensor 중 어느 쪽이 네 번째 접촉과 중복되는가? | 기존 정보 집합에 따라 추가 관측 가치는 달라진다. | 같은 world의 두 관측을 실험자가 무작위 배정해 interaction을 측정한다. |
| 8. 새로운 source에서 학습한 continuous correction이 더 나은가? | Posterior mean과 expected Brier 최적화는 서로 다른 기준이다. | 환경 단위 cross-fitting 후 동결한 연속 weight와 finite action을 대조한다. |
| 9. 추가 정보를 cohort 내에서 공유하면 얻는 양은 얼마인가? | v38은 각 deployment 물체의 실제 측정만 해당 물체에 사용한다. | Context를 공유하는 순서와 비용을 사전 지정하고 per-object와 sequential pooling을 비교한다. |
| 10. 실제 행동을 바꾸는 데 필요한 최소 Brier gain은 얼마인가? | Proper score만으로 장비 운용 효용을 정할 수 없다. | Slip 실패 비용·접촉 시간을 먼저 정해 predicted probability, action switch, expected cost를 함께 본다. |

1–3과 8은 model combination 및 expected future loss의 적용 범위에서, 4–7은 실제 센서와 contact model의 제한에서, 9–10은 정보를 얻고 사용하는 단위의 차이에서 유도한 후속 가설이다. 표의 항목들은 완료된 검증과 구분한다. [Model combination](https://arxiv.org/abs/2101.08954) · [Decision-aware design](https://arxiv.org/pdf/2606.23662) · [실물 contact 동작](https://arxiv.org/pdf/2507.06824) · [Physical experiment selection](https://arxiv.org/html/2609.11022v1)

## 서버 실행과 재현 자료

서버 정본은 datacenter3의 `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/probe_value_v38`에 있다. 부모 디렉터리의 기존 physics-scene-belief 환경과 v36/v37 source를 함께 사용했다. 36개 source·실행 파일의 hash를 [run manifest](results/run.json)에 기록했다. 실험 전 GPU process와 device 사용량을 확인했고, NVIDIA RTX PRO 6000 Blackwell Server Edition 한 장(node55, allocation 2744468)에서 primary와 보조 timing을 차례로 실행했다. 다른 GPU에 학습·추론을 배치하지 않았다. 두 GPU worker 모두 외부 process 없이 정상 종료했다. CPU reference와 검산은 별도 계산 노드 node56의 allocation 2744471에서 CUDA를 비활성화해 진행했다. [Primary 종료](monitor/results/worker_exit.json) · [보조 timing 종료](monitor_continuous/results/worker_exit.json)

동일 project의 빈 할당 GPU에서 주 실험을 시작하는 명령은 아래와 같다. 기존 완료 block은 hash가 일치할 때만 사용한다. 새로운 독립 재현에서는 생성물 없는 작업 사본을 사용한다.

```bash
srun --overlap --jobid="$GPU_ALLOCATION" -n1 bash run38.sh run38.py
```

완료된 결과 디렉터리에는 cell별 seed block 8개, block마다 공통 관측 archive 두 개와 실제 regime·orientation별 archive 여섯 개가 있다. 공통 archive의 deployment sensor 배열은 관측이 선택된 물체의 값만 저장한다. 각 block의 meta에는 SHA-256, calibration·deployment seed, 데이터 생성에 사용한 첫 방향이 들어 있다. [GPU runner](run38.py) · [완료 기록](results/completion.json) · [policy source](POLICY_KO.md)

다음은 할당된 서버 CPU에서 실행하는 계산이다. 상대 경로는 v38 작업 디렉터리 기준이다. `marginal38.py`의 인자 없는 실행은 첫 block만 평가한다. 전체 cell-blind 대조는 각 cell의 여덟 block, 첫 noise repeat를 사용한다. 부가적인 GPU time comparison은 주 worker가 종료된 뒤 별도 monitor directory에서 같은 GPU guard로 continuous38.py를 실행했다. GPU는 보조 session에서도 각 방법의 end-to-end 출력을 raw prediction과 대조했다.

```bash
export CUDA_VISIBLE_DEVICES='' BELIEF_DEVICE=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
py=../continuous_v2/.venv/bin/python
"$py" oracle38.py
"$py" stats38.py
"$py" oracle_stats38.py
mkdir -p marginal_blocks
for b in 000 001 002 003 004 005 006 007; do
  "$py" marginal38.py --block "block_$b" --output "marginal_blocks/block_$b.json"
done
"$py" marginal_stats38.py
"$py" continuous_stats38.py
```

독립 CPU 구현으로 32개 block의 192개 regime·orientation archive를 검산했다. 512개 calibration·deployment seed가 달랐고, archive의 평균 Brier를 재계산한 최대 오차는 5.55×10⁻¹⁷이었다. 기존 physics decoder의 일부 raw probability가 1을 최대 1.8×10⁻¹⁵ 초과했다. 비교에는 원래의 부동소수점 출력을 공통으로 사용했다. 별도의 물리 검산은 3,932,160개 contact state·조건을 확인했다. 실제 label과 stick/slip 판정은 모든 block에서 일치했고, 기록된 sensor를 재생성한 최대 오차는 7.22×10⁻¹⁶이었다. [정책 검산](audit38.json) · [물리 검산](physical_summary.json) · [숨은 cell 추론](marginal38.py)
