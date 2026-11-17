# v56 · 소수의 calibration impact로 새 표면의 정책을 바꿀 수 있는가?

**v55 교차점 근처에서는 calibration impact 8회를 사용해도 두 고정 정책보다 평균 held-out Brier나 session loss p90이 낮다는 근거를 확보하지 못했다.** 새 표면에서 두 정책의 평균 손익이 바뀌는 위치는 44.1%로 v55의 43.5%와 가까웠다. 병목은 교차점의 이동보다, 한 표면에서 얻은 8회 판독만으로 어느 쪽에 있는지 판단하는 데 있었다. 동일한 횟수로 무작위 전환하는 정책보다는 pilot에 따라 전환하는 편이 유리했다. 교차점에서 먼 두 환경에서는 두 고정 정책 대비 평균 개선도 관찰됐다.

## 정책과 표본

새 표면에서 **reference와 impact의 변형 방향을 모두 읽는 pilot**을 먼저 구입한다. K=8이면 서로 다른 sacrificial session에서 2.0 N reference 8회와 impact 8회를 수행한다. 한 쌍은 같은 물체와 위치에 대응하며, reset 후에도 물체의 성질이 유지된다고 가정한다. Inactive 또는 미판독도 비용에 포함하고 재시도하지 않는다. 이 한 번의 표면 결정을 이후 deployment session에 공유한다. 다음 두 정책 중 하나만 실행한다.

- **Frozen v54 gate.** 새 deployment session의 첫 세 weak/reference 쌍을 읽고 네 번째 측정을 reference(R) 또는 impact 방향(I)으로 고른다.
- **Fixed 3R+I.** 첫 세 쌍 뒤에는 항상 I를 산다. 두 정책 모두 교체 전후 관측을 동일한 v53 continuous joint decoder에 넣어 impact 뒤 잔여 접선 속도 ≤0.020 m/s인 stick event의 확률을 예측한다.

Switch는 pilot의 두 부호가 모두 읽힌 N쌍 중 불일치 X건을 센다. e는 한 session 안에서 **활성 물체의** reference와 impact 방향이 어긋날 확률이고, w는 표면의 미래 session 중 e=0.5인 비율이다. 나머지 session에서 e=0과 e=0.25가 반반이라는 v55 설계에서는 P(mismatch | active)=q=0.125+0.375w다. 따라서 w의 추정치는 (X/N−0.125)/0.375를 [0,1]로 제한한 값이다. 이 값이 고정된 0.434807보다 크면 fixed 3R+I, 작거나 같으면 gate를 쓴다. X/N의 전환점은 0.288053이다. N=0이면 gate를 유지한다. 예를 들어 네 개를 판독해 하나만 어긋나면 gate, 두 개가 어긋나면 fixed 3R+I로 간다. Selector에는 w, e, material 이름, deployment label이 입력되지 않는다. [구현](switch56.py) · [사전 분석 계획](SCORE_PLAN_KO.md)

24개 fresh catalog에서 braid와 seam material을 하나씩 새로 생성해 총 **48개 synthetic surface**를 얻었다. 각 material에서 기존 두 rate law를 교차했다. 새 deployment 물체 12,288개와 교체 전후 calibration용 물체를 포함한 36,864개의 physical parameter vector에 대해 과거 v50·v55에서 사용한 49,152개와의 일치가 없음을 확인했다. 새 holdout은 같은 surface·contour·law family에서 나온 새 물체와 재질이다. Pilot용 sacrificial session은 별도 gauge model에서 생성했다. 이 pilot 수를 PyBullet로 실측한 물체 수에 더하지 않았다. [생성 코드](generate56.py) · [표본 감사](world_audit56.json)

각 material에서 w=10%, 35%, 40%, 43.48%, 47%, 52%, 80%를 시험했다. Pilot과 deployment는 같은 w에서 독립적으로 뽑는다. 주 비교는 **w=35%와 52% 동등 가중, K=8**이다. 각 material·w당 512개 pilot group, session마다 128개 deployment draw를 사용했다. 평균 Brier는 방향 전이의 randomness를 적분하고, session p90은 실제 binary outcome으로 채점한 session별 평균의 분포에서 산출했다. 같은 표면을 공유하는 두 law나 조건별 반복을 독립 표본으로 세지 않았다. 24개 catalog 단위의 paired bootstrap 5,000회로 두 mean contrast에 대한 simultaneous 95% interval을 구했고, 두 p90 대조는 별도의 secondary family로 평가했다. 세 정책 모두 동일 pilot을 지불하며, 고정 정책은 그 결과를 무시한다. [채점 코드](score56.py)

## 교차점 근처의 결과

| 정책 | 마지막 I의 실제 구매 비율 | Mean held-out Brier ↓ | Session loss p90 ↓ |
|---|---:|---:|---:|
| K=8 pilot switch | 65.42% | **0.158239094** | 0.188808307 |
| 항상 v54 gate | 35.25% | 0.158239423 | 0.188826740 |
| 항상 fixed 3R+I | 100% | 0.158240573 | **0.188794509** |

| Switch에서 대조를 뺀 차이 (Brier ×10⁻⁶) | Mean [동시 95% CI] | Session p90 [동시 95% CI] | Mean 우세 catalog |
|---|---:|---:|---:|
| Switch − v54 gate | −0.33 [−2.30, +1.60] | −18.43 [−38.10, +2.41] | 13/24 |
| Switch − fixed 3R+I | −1.48 [−3.72, +0.82] | +13.80 [−5.57, +40.28] | 14/24 |

두 고정 정책에 대한 사전 지정 우위 기준을 통과하지 못했다. 두 mean interval은 v55에서 개발상 참고한 10×10⁻⁶ 크기의 개선도 지지하지 않는다. Ultra-resolution에서는 switch−gate의 mean 추정치가 +0.40×10⁻⁶으로 부호가 바뀌었다. 독립 deployment draw 두 개에서도 switch−gate의 추정치는 각각 −0.83, +1.29×10⁻⁶이었다. 새로운 표면마다 무조건 switch를 사용하자는 근거는 약하다. [통계 결과](score56.json)

대신 **같은 비율로 fixed 3R+I로 전환하되 어느 표면에서 전환할지를 무작위로 정하면** pilot switch가 평균 0.86×10⁻⁶ 낮았다(탐색적 95% CI 0.77–0.96×10⁻⁶ 개선). 여기서 일치시킨 비율은 *outer policy의 전환율*이다. 이 대조는 첫 세 쌍으로 네 번째 센서를 고르는 v54의 inner gate 효과와 구분된다. True w가 주어지고 v55 threshold만 쓰는 privileged policy의 mean은 0.158232124였다. 더 좋은 고정 대조인 gate에서 이 값까지 줄일 수 있는 7.30×10⁻⁶ 중 pilot switch가 회수한 양은 약 **4.5%**(점추정)였다. Material별 실제 expected Brier까지 알고 고르는 hindsight 진단은 0.158231610이었다. 어느 쪽도 배포 정책으로 사용할 수 없다. [Oracle 및 구매율 대조](score56.json)

## 8회로 식별할 수 있는 범위

![Pilot 횟수별 전환율과 교차점 근처의 Brier 차이](pilot_switch56.png)

그림 왼쪽에서 음영은 주 비교의 두 구성비 사이이고 점선은 v55 교차점이다. 오른쪽은 각 K에서 두 고정 대조와의 paired mean 차이다. 오차막대는 해당 K 안의 두 대조에 대한 동시 95% interval이다. K 전체에서 정책을 탐색해 얻은 familywise 보장은 아니다.

| 실제 high-mismatch session 비율 w | Active pair의 이론적 불일치율 q | K=8에서 fixed 3R+I로 전환한 비율 | 해당 w에서 평균이 더 낮은 고정 정책 |
|---:|---:|---:|---|
| 10% | 16.25% | 23.73% | v54 gate |
| 35% | 25.63% | 40.86% | v54 gate |
| 40% | 27.50% | 44.34% | v54 gate |
| 43.48% | 28.81% | 46.76% | v54 gate |
| 47% | 30.13% | 48.79% | Fixed 3R+I |
| 52% | 32.00% | 52.34% | Fixed 3R+I |
| 80% | 42.50% | 68.69% | Fixed 3R+I |

K=8에서 평균 **4.01쌍**을 읽을 수 있었고 0.41%는 한 쌍도 읽지 못했다. w=35%와 52%의 관측을 각각 전부 합치면 불일치율은 25.69%와 32.20%로 생성값과 가까웠다. 반면 한 표면에서 실제로 확보 가능한 8회만 쓰면 w 추정치의 RMSE는 각각 0.390, 0.401이었다. 작은 K에서 clipped estimate가 0과 1에 자주 닿았다. 양쪽에 알맞은 고정 정책을 고른 실험상 balanced accuracy는 **55.74%**였다. [Pilot 반복 통계](pilot_stats56.json)

조건 자체의 난이도도 계산했다. w=35%·52%가 각각 절반이고 활성률 50%, 부호 판독 정확도 100%인 기준 model에서 (유효 수, 불일치 수)의 exact binomial likelihood를 모두 열거했다. 두 모집단 판별에 최적인 규칙도 **K=8에서 55.65%**, K=16에서 57.91%다. 모든 시도가 활성이어도 K=8에서는 58.08%다. 55.74%와 55.65%의 차이는 finite simulation과 계산 모형 간의 차이로, 상한을 초과하는 알고리즘의 증거가 아니다. 더 나아가 두 모집단의 실제 평균 Brier까지 아는 analyst가 K=8의 각 count에서 최선의 행동을 택해도 이 모형상 더 좋은 constant 대비 예상 개선은 **0.46×10⁻⁶**이었다. 환경별 손실의 비대칭을 반영한 값이므로 단순 판별 정확도와 다르다. [사전 진단 계획](IDENTIFICATION_PLAN_KO.md) · [Exact 계산](identification56.json)

새 데이터에서 집계한 두 고정 정책의 교차점은 **44.12%**였다. 개별 material별 교차점의 중간 50% 범위는 40.11–48.50%였다. 평균 교차점이 잘 전이돼도 개별 표면의 이득이 똑같이 바뀌지는 않는다. 교차점에서 먼 w=10%와 80%를 동등 가중한 **탐색적** 비교에서는 switch가 gate보다 15.30×10⁻⁶ [13.23, 17.42], fixed I보다 13.68×10⁻⁶ [11.24, 16.42] 낮은 평균 Brier를 보였다. 이 결과는 등록한 near-crossing 주 비교를 대체하지 않는다.

## Mismatch 빈도 하나가 최적 정책을 결정하지 못하는 사례

Low-mismatch 두 구성요소가 늘 반반이라는 가정을 풀었다. Pilot에서 평균 불일치가 정확히 29%로 같아지도록 두 환경의 조성을 **결과를 열기 전에** 고정했다. 아래의 차이는 새 catalog에서 계산한 gate−fixed 3R+I다.

| 환경 | e=0 / 0.25 / 0.5인 session의 비율 | 유효 pilot의 평균 불일치 | Mean Brier 차이 (×10⁻⁶) | 더 좋은 정책 |
|---|---|---:|---:|---|
| A | 42% / 0% / 58% | 29% | +41.75 [+38.16, +45.67] | Fixed 3R+I |
| B | 0% / 84% / 16% | 29% | −84.18 [−92.75, −75.97] | v54 gate |

각 session에서 유효한 부호를 한 번씩만 얻으면 A와 B의 mismatch sequence는 모두 Bernoulli(0.29)의 독립 반복이다. **K를 늘려 29%를 정확히 알아도 이 관측만으로 두 최적 행동을 구별할 수 없다.** 이 결론은 동일한 activation·판독 모델과 표의 두 mixture에 대한 반례다. 같은 session에서 서로 다른 active specimen 두 개가 e를 공유하고 두 판독이 조건부 독립이면, 두 번 모두 어긋날 확률은 E[e²]다. A는 14.50%, B는 9.25%다. 따라서 같은 총 impact 횟수에서도 어느 session 안에서 묶어 읽는지가 다음 실험의 검증 가능한 변수다. 이 수치는 이론 예측이며 실제 paired-within-session 계측 성능은 아직 검증하지 않았다. 표의 interval은 두 반례에 대한 catalog bootstrap 동시 95% 범위다. [고정한 조건](IDENTIFICATION_PLAN_KO.md) · [계산 코드](info56.py) · [결과](identification56.json)

현 instrument는 활성 impact의 방향을 정확히 판독한다고 가정했다. Pilot 부호에 독립적인 5%, 10% symmetric error를 추가하고 기존 switch를 그대로 쓰면, fixed 3R+I 전환율은 각각 50.22%, 53.79%로 올라간다. Gate 대비 평균 개선 점추정은 원래 0.33×10⁻⁶에서 각각 0.18, 0.04×10⁻⁶으로 감소했다. 참 불일치율이 0.5 미만인 환경에서 sign error는 관측 불일치율을 높여 불필요한 전환을 유발할 수 있다. 이 추가 분석은 primary의 sensor 가정을 점검한 simulation sensitivity다. [Noise 실험](pilot_noise56.json)

## 물리 비용과 비교 범위

Pilot 하나는 2.0 N reference와 actual impact 방향을 한 쌍으로 읽는다. 네 번째 deployment 측정과 달리 pilot에서는 weak preload를 사지 않는다. K=8이면 표면당 **16 contacts·16 resets**를 추가로 지불한다. 모든 정책의 deployment calibration은 session당 **8 contacts·8 resets**다. 표면당 배포를 한 번만 하면 세 정책 모두 24 contacts·24 resets이고, 두 번이면 총 32 contacts·32 resets다. 교체 전의 공통 8 contacts·8 resets와 개별 배포 물체의 공통 측정은 아래 증분 비교에서 제외했다.

Weak·reference·impact 측정과 각 reset까지의 시간을 각각 0.44초, 1.04초, T초로 놓았다. 같은 표면의 pilot 하나를 H개 deployment session이 공유하면 session당 시간은 다음과 같다.

- 실제 사용량에 대응하는 모델 시간은 **K(1.04+T)/H + 5.92 + r(T−1.04)**초다. r은 마지막 deployment 측정으로 I를 사는 비율이다.
- 동일 wall-clock budget은 **K(1.04+T)/H + 4.88 + max(1.04,T)**초다. 더 짧은 측정을 끝낸 arm은 공통 slot이 끝날 때까지 대기한다고 가정한다.

| K=8 시나리오 | Pilot switch | 항상 gate | 항상 fixed I | 동일 시간 예산 |
|---|---:|---:|---:|---:|
| T=1.04초, H=1 | 22.56초 | 22.56초 | 22.56초 | 22.56초 |
| T=2.08초, H=1 | 31.56초 | 31.25초 | 31.92초 | 31.92초 |
| T=1.04초, H=2 | 14.24초 | 14.24초 | 14.24초 | 14.24초 |
| T=2.08초, H=2 | 19.08초 | 18.77초 | 19.44초 | 19.44초 |

여기서 sensing과 reset 시간은 hardware 실측값이 아니다. 고정 정책이 현실에서 pilot을 생략하면 같은 Brier를 유지하면서 T=1.04초일 때 surface당 16.64초를 아낀다. 동일 예산에서 정보에 접근하는 이득과, 그 정보를 얻으려고 pilot을 살 실용적 가치는 분리해야 한다. [비용 원장](sensing56.json)

## 서버 GPU 측정

| 정책 | 512 objects당 GPU p90 | CPU 선택 p90, 4 sessions | Switch−대조 GPU p90 차이 [동시 95% CI] |
|---|---:|---:|---:|
| K=8 pilot switch | 433.370 ms | 0.1355 ms | — |
| 항상 v54 gate | 433.598 ms | 0.1259 ms | −0.228 ms [−0.965, +0.468] |
| 항상 fixed 3R+I | 433.193 ms | 0.0022 ms | +0.177 ms [−1.992, +0.788] |

사용 중인 allocation을 그대로 두고, 빈 node55에서 fresh forecast, 빈 node56의 NVIDIA RTX PRO 6000 Blackwell에서 timing을 실행했다. 두 material pilot과 네 deployment session의 실제 선택부터 최종 GPU prediction까지 CUDA sync를 포함해 쟀다. Warmup 후 정책당 144 batch, 24 catalog의 matched crossover이며 16,000회 paired bootstrap을 사용했다. 실제 기계 reset과 sensor capture를 GPU 시간에 합산하지 않았다. 두 latency 차이의 interval 모두 0을 포함한다. [GPU 상세](GPU_KO.md) · [Timing 결과](timing56.json)

## 논문에서 채택할 주장

**현재의 8회 pilot switch를 교차점 부근의 성능 개선 방법으로 채택할 근거는 없다.** 반면 *왜* 개선이 어려운지는 독립 holdout에서 구체적으로 설명할 수 있다. 첫째, v55의 교차점은 새 모집단 평균에서 다시 나타났고, 참 구성비를 주면 더 낮은 Brier에 도달한다. 둘째, 표면 하나에서 실제 얻는 부호 수로는 인접한 두 구성비를 구별하기 어렵다. 셋째, 모집단 조성에 대한 제약을 풀면 동일한 관측 분포가 서로 다른 최적 정책과 양립한다. 따라서 논문의 기여로는 새로운 switching 알고리즘의 우위보다 **어느 관측 단위가 향후 sensor 선택을 식별해 주는지 시험하는 실험과 반례**가 적합하다.

다음 검증에서는 총 8회를 유지한 채 **8개 session × 1회**와 **4개 session × 서로 다른 물체 2회**를 교차 비교할 수 있다. 두 물체가 같은 latent e를 실제로 공유하는지 먼저 확인하고, 같은 raw readout과 decoder에서 두 설계의 mean Brier·session p90을 평가한다. 같은 session 내 반복에서 wear·carryover가 크면 E[e²] 진단의 해석도 달라진다. 이를 구분할 independent direction readout과 실제 reset 시간 측정이 실물 타당성의 다음 조건이다.

서버 작업 경로는 `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/mismatch_switch_v56`다. 새 데이터와 sensor 생성은 [생성 부록](GENERATION_KO.md), 두 replay와 세부 inferential 결과는 [정량 부록](RESULTS_KO.md), GPU와 물리 비용의 회계는 [비용 부록](GPU_KO.md)에 정리했다.

## 문헌 검토와 심사

### 1. 논문별 질문, 방법, 표본 및 결론

| 연구 | 질문·방법 | 표본과 평가 단위 | 결과와 결론 | v56에 적용할 때의 한계 |
|---|---|---|---|---|
| Vandermeulen & Scott, *On the Identifiability of Mixture Models from Grouped Samples* | 같은 latent component에서 나온 관측들을 묶으면 mixture가 유일하게 정해지는가? 분포를 표현하는 연산자로 충분조건과 반례를 증명한다. | 임의의 m개 probability measure. 수집한 로봇 trial은 없다. | Group당 2m−1개 관측이면 일반적인 식별이 가능하며 더 작은 group에 대한 반례가 존재한다. | Known e 값을 세 개로 제한한 v56의 최소 필요 횟수를 이 일반 정리에서 그대로 읽을 수 없다. 여기서는 하나의 Bernoulli mismatch가 전달하는 moment를 직접 계산한다. [원문](https://arxiv.org/pdf/1502.06644) |
| Stoye, *Minimax Regret Treatment Choice with Finite Samples* (2009) | 유한한 표본을 보고 두 행동 중 고를 때 worst-case regret이 가장 작은 규칙은 무엇인가? 작은 표본의 risk를 직접 계산한다. | 두 treatment와 다양한 표본 배정으로 정의한 이론적 실험. 실제 치료 환자 cohort를 평가한 연구는 아니다. | Empirical success rule이 자주 근접하지만 작은 표본과 일부 설계에서는 minimax 규칙과 regret이 달라진다. | 이 논문은 v56의 Brier나 sensor noise를 연구하지 않았다. 여기에서 frozen threshold가 minimax라는 결론은 나오지 않는다. [원문](https://cemmap.ac.uk/wp-content/legacy/resources/files/manski09_7.pdf) |
| Heath, Manolopoulou & Baio, *Estimating the Expected Value of Sample Information across Different Sample Sizes* | 표본 크기에 따른 EVSI를 효율적으로 추정할 수 있는가? posterior variance와 nonlinear regression을 결합해 moment matching을 확장했다. | Chronic pain intervention의 cost-effectiveness model에서 복수 trial design을 비교했다. 실제 환자 치료 효과를 새로 측정하지 않았다. | 예제에서 여러 표본 크기를 효율적으로 비교했다. Simulation 자체가 비싼 model에서는 추가 계산 부담이 있다. | Brier 1단위와 장비 시간 1초의 교환 가치가 주어지지 않으면 여기서 최적 K를 경제적으로 결정할 수 없다. [원문](https://doi.org/10.1177/0272989X19837983) |
| Acosta, Yang & Posa, *Validating Robotics Simulators on Real-World Impacts* (2022) | Contact parameter를 맞춘 simulator가 실제 impact를 얼마나 재현하는가? 동일 관측과 Drake·MuJoCo·Bullet의 궤적을 대조했다. | Cube toss 550개 trajectory, Cassie jump 22회. 로봇에는 Drake와 MuJoCo를 비교했다. | 비탄성 충돌과 전체 움직임을 잘 재현한 사례가 있지만 elasticity 및 세부 dynamics에서 오차가 남았다. | 동일 surface에서 reference–impact sign을 쌍으로 읽고 sensor 구매 정책을 비교하는 데이터가 아니다. [원문](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf) |

### 2–3. Literature review와 knowledge map

**Surface의 latent law → 같은 session에서 얻는 관측 수 → 식별할 수 있는 mixture 정보 → action별 expected loss → 측정 비용** 순서로 연구를 연결할 수 있다. 앞의 두 단계는 grouped sampling에 관한 이론, 가운데 두 단계는 finite-sample decision theory, 마지막 단계는 EVSI와 trial design의 문제다. 이 연결을 따르면 e를 더 정확히 추정한 정책이라도 두 행동의 Brier 차이가 작으면 지불한 impact의 가치가 작을 수 있다. 이는 v56에 대한 해석이다. [Vandermeulen & Scott](https://arxiv.org/pdf/1502.06644) · [Stoye](https://cemmap.ac.uk/wp-content/legacy/resources/files/manski09_7.pdf) · [Heath et al.](https://doi.org/10.1177/0272989X19837983)

### 4–7. 공백, 연구 간 차이, 방법론과 근거 강도

- **확립된 부분.** Mixture의 component를 식별하려면 어떤 관측들이 같은 component에서 나왔는지에 대한 실험 설계가 중요하다. 통계적 추정 정확도와 행동을 잘못 골랐을 때 치르는 regret은 구별할 수 있다. [Vandermeulen & Scott](https://arxiv.org/pdf/1502.06644) · [Stoye](https://cemmap.ac.uk/wp-content/legacy/resources/files/manski09_7.pdf)
- **상충하는 듯한 결과.** 표본을 더 얻을 때 기대 정보 가치가 생긴다는 EVSI 결과와, 실제 threshold rule이 작은 K에서 한쪽 상수 정책보다 나쁠 수 있다는 결과는 양립한다. EVSI의 decision maker는 정보를 받은 뒤 기대 효용이 가장 큰 행동을 고를 수 있다. Frozen rule에는 그 최적성이 주어지지 않는다. 문헌에 동일한 contact task에서 상반된 결과를 낸 직접 비교는 없다. [Heath et al.](https://doi.org/10.1177/0272989X19837983) · [Stoye](https://cemmap.ac.uk/wp-content/legacy/resources/files/manski09_7.pdf)
- **측정의 공백.** 물리 simulator와 실물 사이에 남는 오차는 중요한데, 기존 실물 impact trajectory에는 v56에 필요한 독립 pilot·향후 session·동일 예산의 두 action label이 갖춰져 있지 않다. 직접 포착되지 않은 impact sign을 solver state로 대신할 때 generalization의 대상이 좁아진다. [Acosta et al.](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf)
- **방법별 해석.** 이론적 식별 보장은 무한히 많은 group에서 가능한 것에 대한 진술이다. Exact small-sample risk는 지정한 observation law에서의 성능이며, 실물 trajectory 비교는 센서와 solver에 대한 현장 증거를 준다. 이 세 종류의 표본 수를 합산할 수 없다. 실제 surface의 mixture 빈도와 reset 비용에 대한 근거는 부족하다. [Vandermeulen & Scott](https://arxiv.org/pdf/1502.06644) · [Stoye](https://cemmap.ac.uk/wp-content/legacy/resources/files/manski09_7.pdf) · [Acosta et al.](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf)

### 8. 공백에서 도출한 후속 연구 질문 열 가지

| 질문 | 중요한 이유 | 변수와 검증 방법 |
|---|---|---|
| 1. 독립 session당 한 번 읽으면 행동에 필요한 mixture 정보가 충분한가? | 조성이 다른 population이 같은 관측을 줄 수 있다. [Vandermeulen & Scott](https://arxiv.org/pdf/1502.06644) | 평균 mismatch를 고정한 두 mixture의 최적 행동을 fresh Brier로 비교. v56에서 진단. |
| 2. 같은 session에서 두 물체를 읽는 편이 나은가? | Mismatch의 첫 두 moment를 분리할 수 있다. [Vandermeulen & Scott](https://arxiv.org/pdf/1502.06644) | 총 impact 수 고정, 한 번씩 넓게 읽기와 두 번씩 묶어 읽기 비교. |
| 3. 같은 두 건의 mismatch도 어느 쪽 행동의 손실이 더 큰가에 따라 결정이 달라지는가? | 틀린 선택의 비용이 비대칭일 수 있다. [Stoye](https://cemmap.ac.uk/wp-content/legacy/resources/files/manski09_7.pdf) | Source의 conditional cost로 고정한 rule을 새 surface에서 검증. |
| 4. 몇 번의 impact를 사야 측정 시간이 정당화되는가? | 정확도 개선을 장비 비용에 연결한다. [Heath et al.](https://doi.org/10.1177/0272989X19837983) | K별 loss와 장비 시간을 동시 기록해 cost frontier 추정. |
| 5. 한 표면에서 뒤이어 운용할 session 수가 적어도 pilot가 유리한가? | Setup 비용이 여러 배포에 걸쳐 분담된다. [Heath et al.](https://doi.org/10.1177/0272989X19837983) | 같은 surface 재사용 횟수별 no-pilot 정책과 net benefit 비교. |
| 6. 판독 실패가 mismatch와 연관되는가? | 쉽게 읽히는 impact만 남기면 빈도가 편향될 수 있다. [Acosta et al.](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf) | Independent high-speed readout과 대조해 censoring을 e·속도·표면별 추정. |
| 7. 동일 표면의 sign이 시간에 따라 변하는가? | 고정된 구성비를 공유한다는 가정을 시험한다. [Acosta et al.](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf) | Pilot과 deployment 시간 간격을 무작위화하고 연속 날짜별 외부 검증. |
| 8. Near-crossing이 아닌 surface만 골라 더 빨리 멈출 수 있는가? | 의사결정 가치가 작은 구간에 불필요한 probing을 줄인다. [Heath et al.](https://doi.org/10.1177/0272989X19837983) | Dynamic stopping을 source에서 고정하고 equal total contact budget에서 비교. |
| 9. Tail loss의 최적 switch point가 mean과 다른가? | 평균에 맞춘 threshold가 드문 큰 손실을 줄이지 못할 수 있다. [Stoye](https://cemmap.ac.uk/wp-content/legacy/resources/files/manski09_7.pdf) | Source에서 quantile criterion을 고정하고 독립 session의 p90과 mean을 함께 평가. |
| 10. 다른 contact solver·실물에서 최적 행동의 부호가 유지되는가? | Sensor 신호가 충분해도 outcome model이 달라지면 정책이 바뀐다. [Acosta et al.](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf) | 동일 fixture의 paired impact·trajectory, 서로 다른 solver, binary Brier를 대조. |

### 9. 논문 심사 관점

단순히 mismatch 빈도를 세거나 threshold 정책을 제안하는 것의 독창성은 낮다. 이 작업이 추가하는 증거는 독립적인 물리 outcome에서 **통계적으로 식별 가능한 정보와 정책 전환으로 회수할 수 있는 Brier gain을 같은 sensing budget에서 함께 정량화**하는 데 있다. v55에서 정한 교차점이 개별 surface에 정확히 맞는지, 물체의 판독 가능성이 mismatch와 독립인지, 몇 개의 실제 impact로 값을 측정할 수 있는지가 남는다. 같은 session에서 여러 독립 물체를 probing하고 실측 sensor timing을 넣는 비교가 이 한계를 직접 시험한다. 이 네 편은 연구 방향을 판단하기 위한 선택적 검토이며 전체 분야의 최초성을 확증하지 않는다. [Vandermeulen & Scott](https://arxiv.org/pdf/1502.06644) · [Stoye](https://cemmap.ac.uk/wp-content/legacy/resources/files/manski09_7.pdf) · [Heath et al.](https://doi.org/10.1177/0272989X19837983) · [Acosta et al.](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf)
