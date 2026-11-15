# v30 · 추가 calibration probe의 Brier gain을 학습하면 새 contact law에도 전이되는가

**같은 probe budget에서 추가 접촉의 Brier gain을 학습한 scorer는 두 unseen contact law에서 fixed 4나 uncertainty-only보다 held-out Brier를 낮췄다고 확인되지 않았다.** 한 환경의 과거 물체에 접촉하고, 이후 다른 물체에서 query view와 probe 중 무엇을 쓸지 고르는 실험이다. 새 환경 256곳, 평가 물체 32,768개에서 learned의 common-decoder batch Brier는 0.13421, fixed 4는 0.13389, uncertainty-only는 0.13423이었다. Learned − fixed 4는 **+0.00032 [−0.00074, +0.00138]**, learned − uncertainty-only는 **−0.00001 [−0.00086, +0.00083]**이었다. 두 simultaneous 95% interval 모두 0을 포함한다. 이 두 법칙의 평균에서는 사전에 정한 0.003 크기의 개선도 구간 밖이다.

**배분은 달라졌지만, 접촉이 더 가치 있는 환경을 안정적으로 찾지 못했다.** Learned와 uncertainty-only는 환경 42.2%에서 서로 다른 횟수를 배정했다. 첫 2회 뒤 실제 gain이 더 큰 환경을 맞힌 pairwise concordance는 51.9% [46.5%, 57.1%], 첫 stage를 통과한 4곳에서 8회까지 계속할 대상을 정할 때는 46.9% [39.1%, 54.7%]였다. 같은 환경 내에서 더 많은 접촉을 산 것이 항상 이득이라는 가정도 데이터가 보장하지 않았다.

**GPU에서의 차이는 calibration setup에 집중됐다.** 128개 world 전체의 warm p90은 learned 428.36 ms, fixed 4 427.92 ms, uncertainty-only 428.40 ms였다. 각 환경의 posterior fit·scoring·selector bank 생성을 포함하면 각각 512.07, 488.09, 509.02 ms였다. Learned는 fixed 4보다 4.9% 더 걸렸고 uncertainty-only와의 p90 차이는 명확하지 않았다. 세 정책 모두 block마다 calibration 32회와 test probe 104회를 썼다. 물체 reset·sensor 대기 시간은 포함하지 않았다.

## 1. 서베이가 바꾼 실험 질문

네 연구자의 최근 약 10년 대표작을 읽고, 네 개의 citation 경로를 두 단계씩 추적했다. Yale Song의 연구에서 *도움이 필요해 보이는 상태*와 *실제로 도움이 되는 행동*을 구분하는 질문을 가져왔다. Seon Joo Kim의 연구에서는 training과 streaming에서 들어오는 정보의 차이, Seungryong Kim의 연구에서는 internal score가 실제 대응 관계를 보존하는지 독립적으로 확인하는 방법, Lingjie Liu의 연구에서는 한 scene을 설명하는 모델이 새로운 힘에서도 맞는지 검증하는 방법을 가져왔다. 이 네 의견은 논문을 읽고 구성한 가상 지도 의견이다. 연구 질문·방법·표본·결과·한계 비교표, 상충하는 결과의 해석, 10개 후속 질문과 출처는 별도 서베이에 담았다. [네 연구자 서베이 및 방법론 비교](SURVEY_KO.md)

가장 직접적인 선례는 LAL의 error-reduction regression이다. Action-BED는 acquisition과 decision의 co-adaptation까지 교차 평가했고, PhysiCalWorld는 같은 scene에서 과거 interaction을 재사용했다. 이 연구가 차별화될 수 있는 지점은 **같은 표면 분포의 과거 물체에 접촉해서, 나중에 만난 다른 물체에서 어떤 sensor를 쓸지 더 잘 선택하는가**다. 이를 budget, decoder, hidden contact law를 통제해서 묻는다. [LAL](https://papers.neurips.cc/paper/7010-learning-active-learning-from-data.pdf) · [Action-BED, Appendix G](https://arxiv.org/html/2606.23662v1) · [PhysiCalWorld, §§3–4](https://arxiv.org/html/2610.30125v1)

## 2. 학습 target, 예산과 평가 분할

한 world는 geometry·friction·restitution이 숨은 충돌 장면이다. 새 world에서는 정밀 boundary profile을 읽는 query view 또는 별도 충돌의 속도를 읽는 probe 한 개를 골라 다른 충돌의 stick probability를 예측한다. Calibration probe는 같은 물성 분포의 다른 과거 world에 가한다. 최종 label은 simulator에서 계산한 실제 binary stick event이며 Brier loss는 예측 확률과 event의 차이 제곱이다. 각 action에 대한 32개 sensor noise draw를 공유했다. 실물 접촉에서 얻은 label은 없다.

**무엇을 학습했나.** V29의 Coulomb law에서 생성한 24개 independent blocks를 v30의 development data로 재정의했다. 과거 v29에선 평가에 쓰인 데이터이며 v30의 신규 평가와 분리했다. 환경 192곳 각각에서 probe가 2→4회 또는 4→8회로 늘었을 때 같은 128개 미래 worlds의 선택을 다시 계산했다. 한 환경마다 가장 큰 risk gap을 가진 13곳에 test probe를 배정했다. 같은 decoder로 평가한 이전 prefix의 Brier에서 이후 prefix의 Brier를 뺀 **signed gain**을 회귀했다. 관측을 더 받고 성능이 떨어지면 음의 target 그대로 남겼다. 이 target은 실제로 실현된 하나의 calibration future와 하나의 미래 batch에서 측정한 값이다. Scorer의 예측 대상은 그 조건부 평균이다.

같은 world의 binary label을 y, 추가 calibration 전후의 선택한 관측으로부터 얻은 확률을 p_old, p_new라 하면 관측 한 쌍의 Brier 감소량은 (p_old−p_new)(p_old+p_new−2y)이다. Action이 같으면 common decoder의 예측도 같아 이 값은 0이다. Action이 달라져도 새 예측이 실제 y에서 멀어지면 음수다. V29의 flip score에 없던 부호가 여기에 들어간다.

각 stage에서 별도 ridge regression을 학습했다. 입력 15개는 paid prefix의 coupling posterior 평균·표준편차·상호작용과 측정한 probe 속도 요약, factual/probe 속도 공분산이다. 현재까지 구입한 2개 또는 4개의 응답만 입력한다. 각 fold의 feature 표준화와 ridge penalty 선택은 훈련 blocks 안에서 처리했다. 외부 6-fold nested grouped validation으로 source 성능을 추정하고, 전체 24개 source blocks에서 정해진 penalty로 최종 계수를 동결했다. Test contact law, 환경 이름, 현재까지 구입하지 않은 probe, 미래 stick label은 score 입력에 들어가지 않는다. [구현과 features](score30.py) · [동결된 계수와 OOF predictions](results/model_lock.json)

**무엇을 같게 했나.** 한 allocation block에 독립 환경 8곳을 두고, 환경별로 factual worlds 1,024개를 먼저 본다. Learned와 uncertainty-only는 처음 2회씩 접촉한 뒤 네 곳에 2회, 그중 두 곳에 4회를 더 쓴다. 후자는 friction coupling posterior의 표준편차로 순위를 정한다. Fixed 4는 여덟 곳에 4회씩 쓴다. 세 방법 모두 calibration 32회와 quota test probe 104회, 합계 136회다. 중단한 환경도 평가에서 제외하지 않는다. 두 test law에 16개 independent blocks씩, 총 256개 환경·32,768개 held-out worlds를 평가했다. 각 block마다 새로운 수치 bank pair를 사용했고 source와 숫자 seeds가 겹치지 않는지 검산했다. 128개 세계가 동시에 모이는 batch quota가 primary다. 자유 선택 streaming에서는 calibration 32회만 같고 실제 test probe 수는 별도로 센다. [설정](design.json) · [사전 분석 계획](ANALYSIS_PLAN.md)

**어떤 decoder로 채점했나.** Primary common decoder는 각 환경의 실제 population coupling을 입력받지만 새 contact law의 ID와 개별 물체 물성은 모른다. 모든 정책에 동일한 source Coulomb likelihood·source coupling-conditional label model을 적용하고 실제 target-law label로 채점한다. 보조 평가에서는 관측 선택을 그대로 고정한 채 해당 law를 아는 matched decoder로 다시 채점한다. 두 decoder 모두 population 정보를 받는 privileged comparator다. Law-aware는 selector나 scorer를 재학습시키지 않는다. Allocation을 먼저 hash로 동결하고, test factual을 받은 뒤 observation 선택도 동결한 다음, 가능한 관측과 label을 evaluation에만 공개했다. [GPU runner](runner30.py) · [contact law와 decoder](contact30.py)

## 3. Contact law가 바뀐 것을 무엇으로 확인했나

Source Coulomb model의 friction coefficient를 μ, 접선 속도를 vₜ라 한다. 첫 law에서는 stick threshold에 1.20μ, 미끄러질 때의 impulse에 0.80μ를 썼다. 두 번째에서는 q=exp(−|vₜ|/0.45)로 두고 각각 μ(1+0.22q), μ(0.64+0.25q)를 썼다. 비교할 때 geometry, restitution, population panel, noise 모델을 유지했다. 같은 물체의 factual 충돌은 stick하므로 두 law에서 같은 관측을 준다. 충분한 접선 속도를 가한 probe는 slip하고, query 위치의 다른 충돌은 stick/slip 경계가 이동한다. 아래는 같은 숨은 물체에서 law만 source로 되돌린 paired 진단이다. Δ velocity는 noise를 더하기 전 2차원 차이의 L2 norm 평균이다.

| Target law | Target stick | 같은 물체의 source stick | Stick label 변경 | Factual Δ | Probe Δ |
|---|---|---|---|---|---|
| Static–kinetic split | 61.13% | 50.10% | 11.04% | 0.00000 | 0.08532 |
| Speed weakening | 56.95% | 50.89% | 6.07% | 0.00000 | 0.14547 |

### 실제 probe response의 source 분포 이탈

Ridge scorer의 입력에 포함되는 probe velocity 두 성분의 prefix 평균을 원래 단위로 확인했다. v2는 scorer가 probe_tan_mean이라 부르는 두 번째 좌표다. Local surface normal로 회전시킨 새로운 변수는 만들지 않았다. 비교 기준은 같은 횟수의 Coulomb training 환경에서 계산한 5·95 percentile이다. 아래 백분율은 해당 구간을 벗어난 환경 비율이다.

| Law | Prefix | 환경 수 | v2 평균 [source 5%, 95%] | v2 이탈 | v1 이탈 | 같은 물체의 factual v1 이탈 | 실제 평균 Brier gain |
|---|---|---:|---|---:|---:|---:|---:|
| Coulomb (source) | 2→4 | 192 | -0.6899 [-0.8717, -0.4823] | 10.4% | 10.4% | 10.4% | -0.00030 |
| Coulomb (source) | 4→8 | 192 | -0.6810 [-0.8132, -0.5364] | 10.4% | 10.4% | 10.4% | +0.00071 |
| split_static | 2→4 | 128 | -0.7493 [-0.8717, -0.4823] | 10.9% | 13.3% | 9.4% | +0.00032 |
| split_static | 4→8 | 128 | -0.7484 [-0.8132, -0.5364] | 21.9% | 10.9% | 8.6% | +0.00064 |
| speed_weak | 2→4 | 128 | -0.8153 [-0.8717, -0.4823] | 26.6% | 7.8% | 1.6% | +0.00082 |
| speed_weak | 4→8 | 128 | -0.8109 [-0.8132, -0.5364] | 49.2% | 4.7% | 10.9% | -0.00006 |

Source의 각 192개 환경, 두 target law 각각의 128개 환경을 모두 사용했다. 4회 prefix는 중단한 곳에도 가상으로 공개한 사후 평가다. 실제 gain은 같은 128개 worlds에서 두 prefix의 common-decoder quota Brier를 빼 계산했다. 과거와 현재의 평균 차이에는 contact law와 새로 추출한 유한 표본의 차이가 함께 들어간다. 이탈은 training support 밖이라는 판정이 아니라 source 중앙 90%를 벗어났다는 뜻이다. 각 변수의 주변분포를 따로 보므로 이 표로 joint shift의 크기, 성능 하락의 원인, 배포용 alarm의 정확도를 추정할 수는 없다. [감사 수치](results/shift_audit.json)

Frozen linear scorer에서 source 평균 예측과 target 평균 예측의 차이는 계수 × 입력 평균 변화의 합으로 정확히 나뉜다. Probe velocity 평균·표준편차와 factual/probe covariance 등 다섯 sensor features의 합, 나머지 posterior features의 합을 원래 Brier gain 단위로 표시했다. 서로 상관된 feature가 함께 달라지므로 이 분해는 계수에 따른 산술적 기여다.

| Law | Prefix | 평균 예측 변화 | Sensor features 기여 | Posterior features 기여 |
|---|---|---:|---:|---:|
| split_static | 2→4 | +0.00032 | +0.00007 | +0.00025 |
| split_static | 4→8 | -0.00026 | -0.00027 | +0.00001 |
| speed_weak | 2→4 | +0.00028 | +0.00006 | +0.00022 |
| speed_weak | 4→8 | -0.00044 | -0.00044 | +0.00000 |

## 4. Held-out Brier · 관측 선택 효과

| 운영 조건·decoder | Learned | Fixed 4 | Uncertainty-only | Learned − Fixed 4 [95%] | Learned − Uncertainty [95%] |
|---|---|---|---|---|---|
| Batch · source decoder | 0.13421 | 0.13389 | 0.13423 | +0.00032 [-0.00074, +0.00138] | -0.00001 [-0.00086, +0.00083] |
| Batch · law-aware decoder | 0.10185 | 0.10220 | 0.10197 | -0.00035 [-0.00101, +0.00040] | -0.00012 [-0.00081, +0.00055] |
| Streaming · source decoder | 0.13132 | 0.13178 | 0.13148 | -0.00045 [-0.00120, +0.00029] | -0.00015 [-0.00069, +0.00039] |
| Streaming · law-aware decoder | 0.09928 | 0.09969 | 0.09962 | -0.00041 [-0.00093, +0.00010] | -0.00033 [-0.00090, +0.00019] |

차이가 음수이면 learned에 유리하다. 두 law의 평균에 동일한 가중치를 주었다. 서로 다른 seeds로 생성한 두 law 각각에서 paired block 전체를 10,000회 재추출했다. 첫 행은 두 대비에 대한 simultaneous 95% interval, 나머지는 pointwise 95% interval이다. Noise 32회를 독립적인 labels로 세지 않았다. 두 law는 고정한 두 사례이며 새로운 contact law 전체로 interval을 일반화할 수 없다.

| Law · batch quota | Decoder | Learned | Fixed 4 | Uncertainty-only | Learned − Fixed 4 [95%] | Learned − Uncertainty [95%] |
|---|---|---|---|---|---|---|
| Static–kinetic split | Source | 0.13845 | 0.13849 | 0.13842 | -0.00003 [-0.00133, +0.00147] | +0.00003 [-0.00063, +0.00071] |
| Static–kinetic split | Law-aware | 0.09600 | 0.09692 | 0.09583 | -0.00092 [-0.00181, +0.00010] | +0.00017 [-0.00081, +0.00101] |
| Speed weakening | Source | 0.12997 | 0.12930 | 0.13003 | +0.00067 [-0.00055, +0.00188] | -0.00006 [-0.00142, +0.00119] |
| Speed weakening | Law-aware | 0.10770 | 0.10747 | 0.10810 | +0.00022 [-0.00075, +0.00135] | -0.00040 [-0.00141, +0.00057] |

각 stage의 예산 histogram을 같은 확률로 무작위 배정했을 때의 정확한 기대 Brier는 0.13411였다. Learned와 이 기대값의 차이는 +0.00010 [-0.00048, +0.00070]다. 자유 선택 streaming에서 learned, fixed 4, uncertainty가 실제로 probe를 고른 비율은 각각 9.75%, 9.73%, 9.80%였다. Batch에서 learned가 고른 관측이 fixed 4, uncertainty의 선택과 다른 비율은 8.12%, 4.47%였다. [전체 risk·비교 수치](results/analysis30.json)

아래 값은 고정 4회에서 일찍 멈춘 환경의 손익과, 8회까지 계속한 환경의 손익을 여덟 환경 전체에서 평균한 기여분이다. 두 행을 합치면 primary 차이가 정확히 복원된다. 어떤 환경이 일찍 중단되는지 정책마다 달라지는 점을 포함한다.

| 환경별 재배분 | Learned − Fixed 4의 기여 [pointwise 95%] | Learned − Uncertainty의 기여 [pointwise 95%] |
|---|---|---|
| 2회에서 중단 | +0.00027 [-0.00045, +0.00105] | -0.00000 [-0.00056, +0.00053] |
| 8회까지 관측 | +0.00004 [-0.00059, +0.00068] | -0.00001 [-0.00065, +0.00060] |

[재배분 기여분 검산](results/allocation_parts.json)

같은 world에서 query view만 또는 probe만 받았다고 가정한 두 potential predictions의 Brier도 따로 계산했다. 아래 표에서는 관측 종류를 고정했으므로 선택 정책의 영향 없이 decoder mismatch가 어느 측정값 해석에 집중되는지 볼 수 있다. 이 두 행동은 동일 quota로 경쟁하는 정책이 아니므로 primary 순위에 넣지 않았다.

| Target law | Source decoder · query | Source decoder · probe | Matched decoder · query | Matched decoder · probe |
|---|---|---|---|---|
| Static–kinetic split | 0.13461 | 0.22035 | 0.09690 | 0.16966 |
| Speed weakening | 0.11639 | 0.26091 | 0.10390 | 0.18576 |

## 5. 직접 학습한 gain은 얼마나 예측 가능한가?

| Source · stage | 환경 수 | Ridge penalty | OOF gain correlation | OOF MSE | Train-fold mean MSE | Split-half gain rank correlation [95%] |
|---|---|---|---|---|---|---|
| 2→4 | 192 | 3.0 | 0.112 | 0.0000717 | 0.0000730 | 0.099 [-0.008, 0.211] |
| 4→8 | 192 | 3.0 | -0.136 | 0.0000473 | 0.0000467 | 0.010 [-0.133, 0.139] |

예측값을 실제 배분에 적용한 비교도 source에서 확인했다. 각 과거 block에 대해 그 block을 통째로 제외하고 학습된 두 stage의 OOF predictions로 2/4/8회 배분을 재구성했다. Source batch Brier는 learned 0.10772, fixed 4 0.10806, uncertainty-only 0.10722였다. 따라서 unseen law에서 차이가 나더라도 그 원인을 전부 contact-law shift에 귀속할 수 없다. Fold 간 학습 표본이 겹치고 별도의 source replication이 없어 이 비교는 기술적 참고값으로 제시한다. [OOF 정책 재생](results/source_oof_policy.json)

Source target을 진단하려고 이미 동결된 128개 test actions를 유지한 채, 두 64-world 부분표본에서 gain을 따로 계산했다. 위 상관은 두 부분표본이 환경의 순서를 얼마나 일관되게 매기는지 나타낸다. 두 부분표본이 전체 batch의 13개 quota 결정과 하나의 calibration future를 공유하므로, 이는 새 calibration futures 간 신뢰도의 추정치도, 엄밀한 성능 상한도 아니다. [Split-half 원수치](source_target_noise.json) · [진단 코드](target_noise30.py)

128개의 world와 calibration future를 고정한 뒤 32회 sensor noise만 다시 측정했을 때의 환경별 gain 표준오차(RMS)는 2→4에서 0.00027, 4→8에서 0.00025이다. 이 진단은 기존 source data에서 계산했다. 같은 label을 공유하는 noise 반복을 32배 많은 독립 물체로 취급하면 표기되는 정밀도가 과장된다.

| Unseen law · stage | 예측 gain | 실제 gain | 예측 MAE | 항상 0 MAE | Source mean MAE | 환경 순위 일치율 [95%] | 계속−중단 실제 gain [95%] |
|---|---|---|---|---|---|---|---|
| 2→4 · 후보 전체 | 0.00000 | 0.00057 | 0.00694 | 0.00688 | 0.00690 | 0.519 [0.465, 0.571] | +0.00005 [-0.00221, +0.00220] |
| 4→8 · 첫 단계 생존 환경 | 0.00038 | -0.00051 | 0.00704 | 0.00703 | 0.00706 | 0.469 [0.391, 0.547] | +0.00067 [-0.00236, +0.00369] |

Gain은 동일한 stage에서 추가 probe를 가상 구입한 환경의 Brier 감소량이다. 환경 둘의 실제 gain 순서를 맞히면 1, 동률은 0.5로 계산했다. 둘째 stage는 learned가 첫 stage에 남겨둔 네 환경을 평가한다. 같은 stage에서 큰 예상값을 고르는 정책은 모든 점수가 음수여도 정해진 예산을 사용한다. 이 예산 계약으로는 자유로운 조기 종료나 접촉 가격을 별도로 최적화했다고 볼 수 없다.

**사후에 가장 좋아 보이는 환경을 골라도 이득이 재현되는가?** 2·4·8회라는 같은 배정 분포를 지키는 420개 배분 모두를 평가했다. 128개 미래 물체의 정답을 전부 열고 가장 작은 Brier를 선택한 낙관값은 0.12976이었다. 서로 다른 64개 물체 중 한쪽의 정답만 보고 배분을 정한 뒤 반대쪽에서 채점하고 역할을 바꾸자 0.13431였다. Fixed 4와의 차이는 +0.00042 [-0.00063, +0.00146]다. 동일 factual batch의 quota와 calibration future를 공유하는 탐색적 진단이며, 배포 정책이나 독립적인 성능 상한이 아니다. [배분 재평가 원수치](results/half_replay.json)

## 6. Compute와 probe 비용

| GPU p90 (ms) | Learned | Fixed 4 | Uncertainty-only | Learned/Fixed [95%] | Learned/Uncertainty [95%] | 정책당 시료 수 |
|---|---|---|---|---|---|---|
| Streaming 1개 · warm | 3.563 | 3.561 | 3.557 | 1.001 [0.998, 1.003] | 1.002 [0.999, 1.003] | 384 |
| Batch 128개 · warm | 428.363 | 427.925 | 428.396 | 1.001 [0.998, 1.003] | 1.000 [0.997, 1.002] | 48 |
| Streaming 1개 · setup/128 합산 | 4.220 | 4.019 | 4.177 | 1.050 [1.039, 1.173] | 1.010 [0.997, 1.135] | 384 |
| Batch 128개 · 전체 setup 합산 | 512.072 | 488.094 | 509.020 | 1.049 [1.041, 1.187] | 1.006 [0.998, 1.143] | 48 |

GPU는 NVIDIA RTX PRO 6000 Blackwell Server Edition였다. 각 law의 첫 8개 block에서 사전 seed로 지정한 환경 하나씩, 총 16곳에서 측정했다. Warm-up 뒤 방법 순서를 섞고 host 입력부터 선택·decoder·host 반환까지 CUDA synchronization을 포함했다. Timing 때의 action과 stick probability를 저장한 예측과 검산했다. Batch는 128건 전체의 wall time이고 streaming은 1건의 응답시간이다. setup은 경로별 posterior fit, scored allocation, selector bank 생성이다. Streaming의 setup은 128개 world로 상각했으며 batch에는 전액 가산했다. 95% interval은 같은 환경 내 paired timings을 보존한 law-stratified block bootstrap이다. Source 파일 로딩·validation·fit는 2.13초, 공통 estimator 초기화는 159.19 ms였다. 과거 v29 label 생성, 공통 decoder 생성, 센서·물체 reset은 timing에서 제외했다. 따라서 이 표는 실물 end-to-end wall time을 뜻하지 않는다. [Timing 기록](results/timing_stats.json)

![학습한 probe value와 새 물리 법칙에서의 관측 선택](results/v30.png)

## 7. 논문으로 주장할 수 있는 범위와 다음 결정

**정책 선택.** 두 unseen laws와 이번 비용 조건에서는 fixed 4를 기본 정책으로 두는 편이 타당하다. 셋의 primary Brier 차이는 불확실하고 fixed 4의 setup 포함 p90이 더 짧았다. 단순 uncertainty-only에서도 비슷한 Brier가 나왔다. 작은 차이의 부호를 근거로 세 정책의 일반적인 성능 서열이나 동등성을 주장할 수는 없다. Streaming, 개별 law, matched decoder 결과는 사후에 좋은 칸을 골라 primary를 대체할 수 없다. 두 법칙 모두 같은 2D 충돌 모델의 마찰 변형이고, detector는 원래 Coulomb likelihood를 사용했다. 현재 결론의 적용 범위는 이 조건들이다.

**논문에서 의미 있는 관찰은 세 가지다.** 첫째, 같은 물체의 factual 입력을 그대로 유지하면서 물리 법칙만 바꿔도 평가 label이 11.04%, 6.07% 바뀌었다. 둘째, learned가 고른 관측을 고정하고 decoder에 실제 law를 알려주자 batch Brier가 0.13421에서 0.10185로 0.03236 낮아졌다. 같은 source decoder에서 allocation을 바꾼 효과보다 훨씬 큰 차이다. 이때 decoder는 probe likelihood와 stick label 계산을 **모두** 바꾸므로 어느 하나만의 개선이라고 해석할 수 없다. 또 privileged sensitivity이므로 이 차이를 배포 가능한 성능으로 제시할 수 없다. 셋째, 같은 예산의 420개 배분 중 정답을 다 보고 고르면 0.12976까지 낮아지지만, 미래 물체 절반에서 고른 배분을 나머지 절반에 적용하면 0.13431이었다. Finite batch에서 retrospectively 좋은 배분이 있다는 사실만으로 환경별 gain을 예측할 수 있다고 말하기 어렵다.

**무엇 때문에 이득이 사라졌는지는 완전히 분리되지 않았다.** Source에서도 4→8 OOF correlation은 −0.136이었고 split-half rank correlation은 0.010이었다. 따라서 실패를 모두 새로운 contact law 탓으로 돌릴 근거는 없다. 한편 speed weakening에서는 4회까지 받은 probe velocity의 두 번째 성분 평균이 source 중앙 90% 범위를 벗어난 환경이 49.2%였다. 분포 이동과 불안정한 gain label이 함께 관측됐다. 이 환경에서 learned scorer는 192개 source 환경의 15개 feature를 사용하는 ridge regression이었다. 더 풍부한 nonlinear representation도 실패할 것이라는 결론은 나오지 않는다.

**다음 연구의 진행 기준.** 나는 더 큰 scorer를 바로 학습하기보다 **같은 paid prefix의 gain을 독립적으로 두 번 측정해 재현되는지** 먼저 검증하겠다. 동일 환경·동일 2회 prefix에서 갈라지는 여러 독립 calibration futures를 만들고, 각각에 서로 겹치지 않는 복수 deployment batches를 연결한다. 같은 future에서 batch만 바꾼 경우와 같은 prefix에서 future까지 바꾼 경우를 비교해 어느 단계의 변동이 큰지 추정한다. 그 뒤 source에서 얻은 높은 gain 순위가 별도 batch에서도 유지될 때, 새 물리 family를 통째로 제외한 평가에서 fixed 4와 uncertainty-only를 다시 비교한다. Sensor-likelihood만 바꾼 decoder와 label rule만 바꾼 decoder를 추가하면 위 0.03236 차이의 기원도 검증할 수 있다.

**현재 논문 기여 가능성.** Learned acquisition이 강한 baseline을 이긴다는 방법 논문의 핵심 증거는 부족하다. 반면 물리적 비식별성, 실제 probe가 제공하는 정보, 그 정보가 다음 물체의 관측 선택으로 전달될 때의 한계를 **서로 다른 측정으로 검증한 실험 연구**로는 구체적인 결과가 있다. 실물 또는 독립 simulator에서 측정한 contact-law mismatch와 target reliability가 이를 재현하면 연구 주장이 더 강해진다. 기존 error-reduction regression과 decision-aware design과의 차이, 연구자별 가상 지도 의견, 여전히 부족한 증거는 동반 서베이에 정리했다. [문헌 분석과 연구 질문](SURVEY_KO.md)

## 재현 자료

서버 폴더는 `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/learned_probe_value_v30`이다. Source는 같은 프로젝트의 `adaptive_stopping_v29/results`를 사용한다. [GPU 실행 entry point](run_gpu.sh) · [사전 CPU 검산](preflight.json) · [GPU 완료 기록](results/completion.json) · [GPU guard 종료](gpu_worker/results/worker_exit.json) · [통계 재계산 코드](analyze30.py) · [GPU timing 분석](timing_stats30.py) · [9개 항목을 반영한 동반 서베이](SURVEY_KO.md)
