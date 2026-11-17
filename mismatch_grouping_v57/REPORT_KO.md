# v57 · 같은 평균 mismatch에서 여덟 calibration impact를 어떻게 배분해야 하는가?

**4개 session에서 서로 다른 물체를 두 번씩 측정하면 어느 정책이 유리한지 판별하는 정확도가 50.00%에서 54.17%로 올랐다. 그러나 더 가능성 높은 환경의 정책을 그대로 쓰면 항상 v54 gate를 쓸 때보다 평균 held-out Brier loss가 0.000004387 높아졌다.** 오판 비용까지 반영하면 차이는 0.000000049 개선으로 바뀌었다. 두 방법 모두 session loss p90 개선을 입증하지 못했다. 이 연구에서 grouping의 가치는 environment identification과 downstream decision에 대해 다르다. [서버 채점 결과](score57.json)

## 1. 표면에 대해 무엇을 알고 어떤 결정을 내리는가

교체된 표면에서 2.0 N reference 때의 변형 방향과 실제 impact 때의 방향이 어긋나는 사건을 **mismatch**로 정의했다. 각 session에서 mismatch가 일어날 확률을 e라 하자. 환경 A에서는 session의 42%가 e=0이고 58%가 e=0.5다. B에서는 84%가 e=0.25이고 16%가 e=0.5다. 두 환경의 평균은 정확히 29%다. v56의 분석에 근거해 A에는 항상 마지막 impact를 사는 **fixed 3R+I**, B에는 처음 세 reference를 본 뒤 네 번째를 reference 또는 impact로 정하는 **v54 source-risk gate**가 유리하다고 결과 전에 지정했다. v57의 새 48개 surface instance 모두에서 이 방향이 재현됐다. 여기서 ‘최적’은 이 두 행동 중에서의 평균 Brier를 뜻한다. [사전 평가 계획](SCORE_PLAN_KO.md) · [v56 손익](../mismatch_switch_v56/identification56.json)

두 설계 모두 서로 다른 sacrificial object 여덟 개에 reference와 impact를 각각 한 번 가한다. 8×1은 물체마다 session을 다시 시작하고, 4×2는 같은 latent session state를 유지한 채 두 번째 물체로 교체한다. Policy에는 짝의 배정과 부호 판독만 제공한다. 두 후보의 mismatch 분포와 균등 prior는 주어지지만 실제 환경 이름, session의 e, deployment outcome은 가린다. Pilot이 끝난 뒤 별도 물체·session에서 선택된 정책으로 stick 확률을 예측한다. 네 번째 deployment sensor를 고르는 내부 gate와 pilot을 보고 두 정책 중 하나를 고르는 외부 selector는 서로 다른 결정이다. [생성 계획](WORLD_PLAN_KO.md) · [동결한 selector](policy57.py)

**왜 두 설계의 정보가 다른가?** 8×1에서 판독 한 개의 mismatch 확률은 모두 0.29다. 독립 session의 부호 개수를 늘려도 이 두 가설의 likelihood는 같다. 반면 한 session의 두 물체가 모두 mismatch일 확률은 E[e²]이다. A는 14.50%, B는 9.25%다. 실측된 합성 pilot에서 4×2의 double mismatch는 각각 **14.34%, 9.22%**, 독립 session의 인접 두 slot을 짝지으면 **8.30%, 8.40%**였다. 한쪽 또는 양쪽 판독이 없으면 해당 짝의 likelihood ratio는 1이다. 여덟 번을 모두 지불했지만 31.63%의 pilot에는 두 물체가 모두 판독된 짝이 하나도 없었다. [생성 감사](GENERATION_KO.md) · [정보량 사전 계획](INFORMATION_PLAN_KO.md)

![두 물체의 mismatch 분포와 환경에 따른 fixed 3R+I 선택률](grouped_pilot57.png)

## 2. 독립 검증과 평가 단위

24개 fresh catalog에서 두 surface instance, 두 rate law를 교차했다. Deployment에는 총 12,288개 새 물체를 썼고 support를 포함한 36,864개 physical parameter vector는 감사한 이전 표본과 중복이 없었다. 각 surface·환경·설계마다 4,096개 pilot episode를 생성했다. 두 설계는 서로 대응하는 specimen과 noise를 사용했고 pilot과 향후 deployment draw는 독립으로 뽑았다. 새로운 instance들은 이전 연구와 같은 합성 contour·law family에 속한다. Pilot의 부호는 sensor model로 생성하고 deployment 정답은 PyBullet의 반사실적 deformation 방향별 simulation에서 얻었다. Stick은 residual tangential speed ≤0.020 m/s다. [표본 명세와 checksum](world_audit57.json) · [생성 코드](generate57.py)

모든 정책은 같은 frozen terminal decoder로 예측한다. Primary mean은 held-out object의 binary stick outcome에 대해 알려진 전이확률을 적분한 expected Brier다. Session loss p90은 같은 128개 물체의 binary deployment draw에서 각 session의 평균을 낸 뒤 구한 90번째 백분위수다. 4,096개 pilot을 독립 surface로 세지 않고, 24개 catalog를 단위로 5,000회 paired bootstrap했다. Mean과 p90에 대해 각각 두 대조의 동시 95% CI를 계산했다. 독립 binary replay와 solver refinement도 별도로 확인했다. [채점 명세](RESULTS_KO.md) · [원시 집계](score57_cells.npz)

## 3. 평균 손실과 tail이 서로 다른 답을 주는 이유

| Pilot과 배포 정책 | 환경 판별 balanced accuracy | Fixed 3R+I로 전환 | 평균 held-out Brier ↓ | Session loss p90 ↓ |
|---|---:|---:|---:|---:|
| 8×1 MAP | 50.000% | 0% | 0.158347117 | 0.190866411 |
| 4×2 MAP | 54.171% | 40.742% | 0.158351504 | 0.190864995 |
| 4×2 loss-weighted | 50.205% | 0.446% | 0.158347068 | 0.190866411 |
| 항상 v54 gate | 50.000% | 0% | 0.158347117 | 0.190866411 |
| 항상 fixed 3R+I | 50.000% | 100% | 0.158364159 | 0.190873131 |
| 환경을 아는 privileged 선택 | 100% | 50% | 0.158325079 | 0.190833300 |

각 환경 비중은 50%다. 상수 정책의 판별률은 그 정책에 대응하는 환경을 항상 예측한다고 해석했다. 8×1 MAP는 모든 관측에서 likelihood가 같아 사전 지정한 tie action인 gate를 쓴다. 두 상수 정책도 공정한 예산 비교에서는 같은 pilot 비용을 지불한다. [전체 결과와 CI](RESULTS_KO.md)

- **판별 효과.** 4×2 MAP의 판별률 개선은 +4.171 percentage points [95% CI +4.075, +4.267]였다. A recall은 44.91%, B recall은 63.43%다. 같은 material·환경에서 서로 다른 pilot끼리 두 번째 물체를 바꿔 끼우면 정확도는 50.005%로 떨어졌다. 같은 비율로 fixed를 고르는 random policy와 비교하면 pairing을 사용한 MAP가 평균 Brier를 2.556×10⁻⁶ 낮췄다. [Permutation 및 random control](RESULTS_KO.md)
- **Primary decision.** 4×2 MAP에서 gate를 뺀 평균 Brier 차이는 +4.387×10⁻⁶ [동시 95% CI +1.748, +6.984]이고 fixed를 뺀 차이는 −12.655×10⁻⁶ [−16.710, −8.507]이다. A에서 올바른 전환 한 번이 회수하는 평균 손실은 44.08×10⁻⁶, B에서 잘못 전환할 때 지불하는 평균 손실은 78.16×10⁻⁶이었다. 전체에서 회수한 9.897×10⁻⁶보다 잃은 14.284×10⁻⁶이 컸다. [손익 분해](opportunity57.json)
- **Secondary decision.** 판별 odds가 1을 넘을 때마다 바꾸는 대신 v56 손익으로 고정한 2.016을 넘을 때만 바꾸면 gate 대비 −0.049×10⁻⁶ [−0.075, −0.024], fixed 대비 −17.091×10⁻⁶ [−23.760, −10.319]이었다. 이 정도면 환경을 완벽하게 알 때 가능한 개선의 **0.223%**다. 두 independent binary replay 중 한 번은 gate 대비 CI가 0을 포함했다. [동결한 손실 기준과 replay](RESULTS_KO.md)
- **Tail.** MAP의 session p90 차이는 gate 대비 −1.416×10⁻⁶ [−19.215, +12.473], fixed 대비 −8.136×10⁻⁶ [−26.182, +18.605]다. Loss-weighted의 gate 대비 차이는 0.000×10⁻⁶ [−1.207, 0.000]이다. 이 표본에서 두 상수 정책보다 tail까지 개선됐다고 말할 근거는 없다. [상세 수치](score57.json)

## 4. 물리 예산과 GPU 비용

Pilot 자체는 양쪽 모두 **8 impacts, 8 references, 16 resets**다. 별도 deployment session 한 개의 calibration까지 합하면 **24 contacts·24 resets**다. 4×2가 줄이는 것은 독립 pilot session을 시작하는 횟수 네 번이다. 한 pilot을 두 deployment session이 공유하면 총 32 contacts·32 resets다. 교체 전 공통 calibration 및 deployment object별 공통 weak readout은 이 증분 비교에서 제외했다. [Sensing ledger](sensing57.json)

Reference+reset과 impact+reset을 각각 1.04초로 놓고 추가 setup 시간을 0으로 두면 모든 설계가 deployment 한 번에 22.56초다. Impact+reset이 2.08초라면 같은 padded cap은 31.92초이고, 실제 route 비율로 계산한 평균은 8×1 MAP 31.25초, 4×2 MAP 31.52초, 4×2 loss-weighted 31.25초, fixed 31.92초다. 실물의 setup·readout·reset 시간은 계측하지 않았다. 상수 정책은 운영 시 pilot 16 contacts·16 resets를 전부 생략해도 같은 예측을 낸다. 따라서 동일 예산에서의 작은 Brier 이득을 새 pilot을 구입할 경제적 근거로 해석하기 어렵다. [비용 가정과 amortization](GPU_KO.md)

Node56의 idle NVIDIA RTX PRO 6000 Blackwell Server Edition에서 512개 물체를 한 batch로, 정책별 120회 측정했다. Pilot 판독이 메모리에 있는 시점부터 policy 선택·공통 decoder·GPU forecast 종료까지 양끝에서 CUDA를 동기화했다. 24개 catalog의 실행 순서를 교차시켰다. 별도 idle GPU인 node55가 평가용 forecast를 생성했다. [측정 조건과 원자료](GPU_KO.md)

| 정책 | GPU p90 (ms/512 objects) | 4×2 MAP − 해당 정책 (ms, 동시 95% CI) |
|---|---:|---:|
| 8×1 MAP | 440.218 | +0.355 [−0.392, +0.789] |
| 4×2 MAP | 440.573 | 기준 |
| 4×2 loss-weighted | 440.413 | +0.161 [−0.747, +0.689] |
| 항상 v54 gate | 440.536 | +0.038 [−0.409, +0.953] |
| 항상 fixed 3R+I | 440.322 | +0.251 [−0.526, +0.899] |

모든 primary GPU latency 차이의 interval이 0을 포함했다. Timing에 쓰인 48개 material pilot 중 loss-weighted가 fixed로 전환한 경우는 0개였다. 별도 96개 rare-switch case에서 route와 terminal posterior가 replay와 일치했지만, 이 branch만의 GPU p90은 추정하지 못했다. 물리 sensor와 reset 시간은 위 ledger에서 따로 다뤘다. [Timing 원자료](timing57.json) · [검산](timing_audit57.json) · [Timing checksum](timing_results57.sha256)

## 5. 근거의 강도와 논문 판단

**강한 근거는 sampling 설계가 어떤 정보를 만들어 내는가에 있다.** 8×1의 식별 불가능성, 4×2의 joint mismatch 차이, pairing을 끊으면 사라지는 accuracy 이득이 이 두 환경에서 일치한다. 판독률 50%일 때 정확한 likelihood로 열거한 4×2 accuracy 상한은 54.23%다. 실제 54.17%가 이에 가깝다. 모든 물체에서 부호를 읽을 수 있어도 같은 여덟 impact와 같은 두 환경 아래 상한은 59.58%다. [정확 열거](information57.json)

**제약은 실제 session이 무엇을 공유하는지에 있다.** 두 번째 물체가 첫 물체와 같은 e를 유지할 확률을 100%에서 50%로 내리면 계산상 accuracy는 52.11%가 되고 source-risk threshold의 gate 대비 작은 이득도 사라진다. 반대로 B에서 두 번째 mismatch 판독이 26.58% 확률로 첫 판독을 복사하는 가상 sensor artifact는, 평균과 둘 다 mismatch일 확률을 A와 같게 만든다. 이때 두 번의 부호만으로 물리적 persistence와 판독 상관을 구별할 수 없다. 26.58%는 반례에 필요한 설정값이다. 실제 sensor의 추정 오류율은 측정하지 않았다. [가정과 계산](information57.py) · [독립 sign noise 검증](pilot_noise57.json)

**환경 비중도 결과를 바꾼다.** 동결된 4×2 MAP를 재학습하지 않고 A/B의 비중만 사후에 바꿔 계산하면, A 비중이 약 59.1–67.1%일 때 두 상수 정책보다 평균 손익의 점추정치가 유리하다. 이 구간은 v57 결과를 이용해 구한 탐색적 경계다. 독립 배포 환경에서 확인하거나 범위의 불확실성을 추정하지 않았다. [사후 비중 분석](prevalence57.json)

**논문에서는 같은 sensing budget의 정보 가치와 정책 가치를 분리한 결과를 채택할 만하다.** 실용적인 adaptive policy 성능 주장은 아직 약하다. Opposite policy optima를 가진 환경은 사전 설계했고, 후보 분포와 비용을 알고 시작했으며, independent unit은 24개 synthetic catalog다. 표면의 실제 다양성, session drift, 물체 순서에 따른 wear, shared sensor error는 남아 있다. 다음 확인 실험은 서로 다른 두 센서로 같은 접촉의 부호를 기록하고 물체 순서와 session 간격을 무작위화하는 것이다. 같은 센서와 다른 센서에서 얻은 joint mismatch, 이후 독립 stick outcome을 함께 얻으면 어느 상관이 실제 정책 이득으로 이어지는지 검사할 수 있다. 새로운 물리 결과 없이 현재 결론을 hardware 성능으로 확장하기는 어렵다. [문헌과 후속 질문](LITERATURE_KO.md)

## 6. 재현 자료

서버의 실험 폴더는 `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/mismatch_grouping_v57`다. 생성, 채점, timing을 별도로 동결·실행했다. 물체와 pilot 생성, paired forecast, seed·checksum, guard log, catalog별 점수와 timing 표본을 보관했다. 생성 전에 checksum helper 오타를 수정했고, timing launcher 수정 이력도 보존했다. [생성 감사](GENERATION_KO.md) · [채점 결과](RESULTS_KO.md) · [Timing 감사](GPU_KO.md) · [생성 checksum](generation_results57.sha256) · [채점 checksum](score_results57.sha256)


## 문헌과 논문 기여 심사

### 1. 논문별 질문·방법·표본·근거·한계

| 연구 | 질문과 방법 | 표본·단위 | 결과와 결론 | v57에 적용할 때의 한계 |
|---|---|---|---|---|
| Vandermeulen & Scott, *On the Identifiability of Mixture Models from Grouped Samples* | 같은 latent component에서 나온 여러 관측을 묶으면 mixture가 식별되는가? Grouped distribution을 연산자로 표현해 충분조건과 반례를 증명했다. | m개 probability measure에 대한 수학적 정리. 물리 피험 표본은 없다. | 일반적인 mixture에는 group당 2m−1개의 관측으로 식별이 가능하고 더 적을 때 구별할 수 없는 사례가 있다. | v57은 두 개의 주어진 후보를 비교하고 e의 support도 안다. 일반 정리는 두 번이면 모든 contact mixture를 학습한다는 근거가 아니다. [원문](https://arxiv.org/pdf/1502.06644) |
| Stoye, *Minimax Regret Treatment Choice with Finite Samples* (2009) | 유한 표본으로 두 행동을 고를 때 어떤 규칙이 최대 regret을 줄이는가? Small-sample decision risk를 직접 계산했다. | 두 treatment와 여러 표본 배정에 대한 이론적 실험. 실물 cohort는 없다. | 일부 소표본 설계에서 empirical success와 minimax regret 규칙의 행동·regret이 달라진다. | v57의 source-risk 규칙은 두 환경과 prior를 고정한 expected Brier 기준이다. 이 논문은 그 규칙의 minimax 최적성이나 실제 slip 비용을 보장하지 않는다. [원문](https://cemmap.ac.uk/wp-content/legacy/resources/files/manski09_7.pdf) |
| Luo et al., *PhysiCalWorld* (2026, preprint) | Test scene 안에서 exploratory interaction clip을 얻어 새로운 물리에 맞게 예측을 조절할 수 있는가? 영상에서 physics descriptor를 추출해 action-conditioned rollout에 재사용한다. | 네 환경의 rollout prediction과 visual planning. 각 환경에서 변화시키는 물리 속성이 다르다. | 탐색 영상을 이용하면 test scene의 dynamics와 미래 예측을 더 잘 맞춘 사례를 보고했다. | Clip의 frame은 하나의 session에 속할 수 있다. 그것이 곧 독립 물체 두 개의 mismatch 표본에 해당하지 않는다. 해당 연구에는 v57의 동등 8회 예산·Brier 대조가 없다. [원문](https://arxiv.org/html/2610.30125v1) |
| Acosta, Yang & Posa, *Validating Robotics Simulators on Real-World Impacts* (2022) | Parameter를 맞춘 simulator가 실제 충돌을 얼마나 재현하는가? 실측 trajectory를 simulator 간 대조했다. | 550개 cube toss와 Cassie의 22개 jump. Cube는 3개 engine, Cassie는 2개. | 비탄성 충돌과 전체 운동은 재현되기도 하지만 elasticity 및 착지 dynamics에서 차이가 남았다. | 이 자료에는 같은 표면의 짝지어진 2.0 N reference–impact 방향과 이어지는 독립 deployment outcome이 없다. v57 정책의 hardware 검증을 대체하지 못한다. [원문](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf) |

### 2–3. 주제별 종합과 knowledge map

이 문제는 **같은 상태를 공유하는 물체의 범위 → 그 범위 안에서 수집한 관측의 분포 → 상태 또는 환경에 대한 evidence → 비용을 고려한 행동 → 실제 impact outcome**으로 연결된다. Grouped-sample theory는 앞의 두 단계를, finite-sample decision theory는 posterior evidence를 행동으로 바꿀 때 생기는 손실을 연구한다. Exploratory interaction 연구는 물리적 observation을 실제 prediction에 넣는 방법을 제공하고, simulator validation은 마지막 비교에 외부 측정이 필요한 이유를 보여준다. [Vandermeulen & Scott](https://arxiv.org/pdf/1502.06644) · [Stoye](https://cemmap.ac.uk/wp-content/legacy/resources/files/manski09_7.pdf) · [Luo et al.](https://arxiv.org/html/2610.30125v1) · [Acosta et al.](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf)

### 4–7. 연구 공백, 비교되는 가정과 근거 강도

- **식별 가능성과 유용성.** 동일 component에서 독립적으로 얻은 관측 여러 개가 mixture를 구분하는 정보를 갖는다는 점은 이론에 근거한다. 한 표면당 8회에서 얻을 수 있는 accuracy나 Brier 개선 크기는 별도로 계산해야 한다. 같은 상태를 더 잘 맞혀도 특정 오판의 비용이 크면 downstream decision은 악화될 수 있다. [Vandermeulen & Scott](https://arxiv.org/pdf/1502.06644) · [Stoye](https://cemmap.ac.uk/wp-content/legacy/resources/files/manski09_7.pdf)
- **방법에 따른 결과 차이.** Latent rollout을 학습하는 방법과 부호의 joint likelihood를 쓰는 방법은 정보량도 endpoint도 다르다. 후자는 어떤 두 물체가 같은 session이었는지 교란할 수 있어 pairing 효과를 분리한다. 전자는 속도·변형 크기 등 binary sign 외의 단서를 활용할 수 있다. 두 논문 사이에 동일 실험 조건에서 관찰한 직접적인 모순은 없다. [Luo et al.](https://arxiv.org/html/2610.30125v1) · [Vandermeulen & Scott](https://arxiv.org/pdf/1502.06644)
- **미해결 부분.** 같은 session에서 두 물체가 같은 e를 공유하는지, sensor drift·wear·첫 impact가 다음 물체의 판독을 바꾸는지, 그 의존성이 deployment로 전이되는지는 이 네 연구로 결정할 수 없다. 실물 반복에서 센서와 실제 dynamics를 별도로 읽어야 한다. [Acosta et al.](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf) · [Vandermeulen & Scott](https://arxiv.org/pdf/1502.06644)
- **근거의 범위.** 일반 이론, 네 환경의 rollout 평가, 실제 cube와 로봇의 trajectory 검증에는 서로 다른 독립 단위와 가정이 있다. 이 연구들의 sample size 또는 성공률을 합치거나 한 방법의 성능을 다른 방법의 Brier 효과로 환산할 근거가 없다. [Stoye](https://cemmap.ac.uk/wp-content/legacy/resources/files/manski09_7.pdf) · [Luo et al.](https://arxiv.org/html/2610.30125v1) · [Acosta et al.](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf)

### 8. 검증 가치가 있는 후속 질문 열 가지

| 질문 | 다루는 공백과 의미 | 변수·연구 방법 |
|---|---|---|
| 1. 두 물체가 공유하는 상태는 어느 시간 간격까지 유지되는가? | Latent component 공유가 grouping의 전제다. [Vandermeulen & Scott](https://arxiv.org/pdf/1502.06644) | 간격과 물체 순서를 무작위화하고 조건부 joint mismatch를 측정. |
| 2. 두 부호 판독의 상관이 센서에서 생기는가? | Shared sensor noise가 physical persistence와 비슷한 통계를 줄 수 있다. [Acosta et al.](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf) | 독립 camera와 gauge를 교차하고 동시 오판 확률을 추정. |
| 3. 두 환경의 정의를 모를 때도 grouping이 도움이 되는가? | 알려진 두 서명 중 고르는 문제보다 unknown mixture 학습이 어렵다. [Vandermeulen & Scott](https://arxiv.org/pdf/1502.06644) | Source에서 mixture support를 추정해 고정하고 새 material에서 검증. |
| 4. 8회 중 어느 grouping이 실제 decision loss를 최소화하는가? | 명목상 식별과 작은 표본의 regret이 다르다. [Stoye](https://cemmap.ac.uk/wp-content/legacy/resources/files/manski09_7.pdf) | 8×1·4×2·2×4 설계의 held-out loss를 같은 시간·횟수에서 비교. |
| 5. Active 판독 가능성이 e에 따라 달라지는가? | 비무작위 누락은 관측 likelihood를 바꾼다. [Vandermeulen & Scott](https://arxiv.org/pdf/1502.06644) | 독립 reference instrument로 모든 시도에 audit label을 붙여 수율을 stratify. |
| 6. 오판 비용을 source에서 배운 threshold는 다른 surface에도 맞는가? | 판별 신호와 cost transfer를 분리한다. [Stoye](https://cemmap.ac.uk/wp-content/legacy/resources/files/manski09_7.pdf) | 동일 pilot에 frozen source cost와 known-target cost를 교차. |
| 7. 같은 횟수의 영상에서 부호 이상의 단서를 쓸 수 있는가? | Sign-only pilot이 버리는 정보의 가치를 잰다. [Luo et al.](https://arxiv.org/html/2610.30125v1) | 동일한 여덟 contact를 영상·부호로 표현하고 공통 terminal에서 Brier 비교. |
| 8. 여러 deployment session이 하나의 pilot을 공유하면 tail이 어떻게 변하는가? | Margin에서 평균한 session p90과 fleet 전체 위험은 다르다. [Stoye](https://cemmap.ac.uk/wp-content/legacy/resources/files/manski09_7.pdf) | Surface당 배포 수를 바꾸고 worst-session 및 surface 평균을 따로 평가. |
| 9. Session setup 시간이 grouping의 경제성을 바꾸는가? | 같은 contact 횟수에서도 작업자는 표면을 세팅하는 시간을 지불한다. [Acosta et al.](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf) | Setup·reset·readout을 독립 계측하고 deployment 횟수별 실제 시간을 보고. |
| 10. 다른 contact solver 및 실물에서도 같은 action이 이기는가? | Readout이 같아도 target outcome의 정의가 달라질 수 있다. [Acosta et al.](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf) | 동일 물체·surface의 두 센서와 미래 stick을 독립 측정해 simulator와 비교. |

### 9. 연구 심사

Grouping으로 mixture를 식별하는 원리 자체와 실험으로 test-scene physics를 보정하는 아이디어는 이미 연구됐다. v57의 상대적인 기여는 **같은 수의 물리 관측으로 얻은 두 번째 moment가 환경 판별과 실제 sensor 선택에 각각 얼마의 가치를 갖는지, 두 값이 언제 다른 방향으로 움직이는지**를 한 과제에서 검증하는 데 있다. Grouping을 모르는 permutation control, 같은 비율로 전환하는 random control, 두 기본 정책, source cost를 쓰는 selector를 유지해야 그 주장을 평가할 수 있다. [Vandermeulen & Scott](https://arxiv.org/pdf/1502.06644) · [Stoye](https://cemmap.ac.uk/wp-content/legacy/resources/files/manski09_7.pdf) · [Luo et al.](https://arxiv.org/html/2610.30125v1)

실물에서 동일 session의 latent state 공유, 센서의 조건부 독립, setup·reset의 재현성 중 어느 것이 성립하는지 아직 검증해야 한다. 주장을 현재의 binary pilot, 두 알려진 후보 환경과 합성 contact task로 한정한다. 이 문헌 검토는 네 원문에 대한 목적별 비교이며 분야 전체의 최초성을 판정하지 않는다. [Acosta et al.](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf) · [Vandermeulen & Scott](https://arxiv.org/pdf/1502.06644)
