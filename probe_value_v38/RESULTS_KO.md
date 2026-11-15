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
