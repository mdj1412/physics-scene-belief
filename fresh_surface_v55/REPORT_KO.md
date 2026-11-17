# v55 · 새 표면에서 네 번째 calibration sensor를 고르는 규칙의 재현성

**결과.** v54에서 고정한 gate는 새 material 48개의 균등 가중 평가에서 평균 held-out Brier를 fixed 3R+I보다 **0.00001850**, 구매율을 똑같이 맞춘 random보다 **0.00002318** 줄였다. Session loss p90은 random보다 낮았고 fixed 3R+I와의 차이는 확인되지 않았다. 환경 조성을 바꾸면 두 비교가 갈린다. Reference가 impact 부호와 절반씩 어긋나는 session의 비중이 약 43.5%를 넘을 때에는 매번 impact를 측정하는 쪽이 더 나았다.

## 무엇을 확인했는가

표면 교체 후 세 물체에서 weak preload(0.4 N)와 reference(2.0 N)의 displacement를 읽는다. 네 번째 물체에서 다시 weak + reference를 살지(R), weak + impact 방향을 살지(I)가 유일한 선택이다. 모든 정책은 교체 후 정확히 네 쌍을 구입한다. 목표는 이후 물체에서 impact 뒤의 잔여 접선 속도가 0.020 m/s 이하가 될 확률을 예측하는 것이다. 본문에서 이 positive event를 stick이라고 부른다. Brier는 확률과 binary label 차이의 제곱이며 작을수록 좋다.

- **Frozen gate.** v54 source-risk gate의 코드와 source risk를 고정했다. Gate는 교체 뒤 첫 세 weak/reference 쌍의 연속 판독값만 받는다. 네 번째 weak reading, 교체 전 이력, 참 mismatch, material 이름과 target label은 decision에 들어가지 않는다. Acquisition은 reference–impact 방향이 뒤집힐 확률 e를 Uniform(0, 0.5)로 적분하지만, 구입을 끝낸 뒤에는 세 정책 모두 교체 전후 이력을 같은 v53 nominal continuous joint decoder로 처리한다. [v54 정책](../adaptive_witness_v54/POLICY_KO.md) · [평가 코드](score55.py)
- **Fixed 3R+I.** 앞의 세 쌍이 어떻든 마지막에 I를 산다. 이 대조는 언제나 실제 impact를 본다고 결정한 정책을 선별적 구매가 이기는지 묻는다.
- **Rate-matched random.** Frozen gate가 새 전체 prefix에서 선택한 I의 비율만 알려준다. 이 확률로 각 session에서 독립적으로 I를 고른다. 분석에서는 두 행동 각각의 loss와 loss 분포를 그 확률로 적분해 추첨 noise를 없앴다. GPU 시험에서는 같은 수의 I ticket을 실제로 섞었다. 두 정책의 평균 차이는 정확히 같은 구매율에서 어떤 session에 I를 배정했는지에만 기인한다. 실제 배포에서 이 비율을 미리 아는 상황은 별도의 가정이다. [사전 채점 계획](SCORE_PLAN_KO.md) · [GPU 계획](TIMING_PLAN_KO.md)

## 새 평가 표본

두 새 합성 surface family인 **braid**와 **seam**을 설계했다. Braid는 서로 다른 공간 주파수가 겹쳐 마찰이 변하고, seam은 국소적인 경계 주변에서 마찰이 변한다. 각 catalog마다 진폭·위치·폭 등 material parameter를 새로 뽑았다. 24개 독립 catalog × 두 재질 = **48개 material realization**이다. 재질을 기존 두 contact-rate law와 교차해 96개 environment, environment마다 별도 deployment object 128개를 확보했다. 이전 catalog의 측정이나 정답을 재사용하지 않았다. 교체 전후 calibration과 deployment의 support 및 seed도 분리했다. [Generator](generate55.py) · [독립성 기록](independence55.json)

Catalog마다 4개 environment, 256개 post-exchange prefix draw, 5개 weak→reference 방향 유지율 조건으로 총 **122,880개 prefix 조건**을 얻는다. 같은 draw를 유지율·mismatch·drift 조건과 정책에 짝지어 재사용하므로 독립 추론 단위는 **24개 catalog**다. Calibration gauge와 impact sign은 synthetic instrument model에서 생성한다. 두 새 surface field는 PyBullet의 공간별 마찰에 들어가고 이후 stick label에 영향을 준다. 주어진 e와 session drift가 surface field로부터 자동 추정되는 설계는 아니다. 따라서 새 물체·재질에서의 simulation replication과 물리적 sign-readout 검증은 별개다. [생성 및 표본 감사](generate55.py)

## Primary와 secondary endpoint

등록한 두 비교는 gate−fixed 3R+I, gate−rate-matched random이다. e∈{0, 0.25, 0.5}, session drift 확률 g∈{0, 0.25, 0.5, 1}, 교체 전 유지율 κ∈{0, 0.25, 0.5, 0.75, 1}을 각각 균등 가중한다. 평균 held-out Brier는 각 simulator binary potential outcome을 고정하고 방향 전이의 randomness를 적분한 expected loss다. Session loss p90은 해당 environment의 finite pool에서 128개 deployment draw를 복원추출해 얻은 session 평균 Brier의 90번째 백분위수다. Random의 p90도 두 행동의 loss CDF를 섞어 계산한다. Mean과 p90의 단위 모두 Brier이며 GPU p90만 ms다.

Catalog 단위 paired bootstrap 5,000회에서 random 구매율을 매번 다시 추정했다. 두 primary mean contrast의 familywise 95% interval 상한이 모두 0보다 낮을 때 방향 재현으로 판정한다. 양쪽 모두 −0.00001 아래인지도 별도로 본다. 이 수치는 v54 gain의 대략 절반을 보존하는지 확인하는 개발 기준이다. 실물에서 가치가 생기는 최소 크기는 아직 정해져 있지 않다. 두 p90 대조는 별도 secondary family다. [사전 분석 계획](SCORE_PLAN_KO.md) · [배포 분포 sensitivity 계획](TRANSFER_PLAN_KO.md)

## 결과와 판정

| Policy | I 구매율 | Mean held-out Brier ↓ | Session loss p90 ↓ | 두 번째 deployment draw의 session p90 |
|---|---:|---:|---:|---:|
| Frozen source-risk gate | 34.659% | **0.156878466** | **0.188775987** | 0.188756526 |
| Fixed 3R+I | 100% | 0.156896966 | **0.188775987** | 0.188776151 |
| Rate-matched random | 34.659% | 0.156901644 | 0.188802511 | 0.188790784 |
| Fixed 4R · 진단 | 0% | 0.156904126 | 0.188818560 | 0.188794896 |

| Gate에서 대조를 뺀 값 | 평균 차이 [simultaneous 95% CI] | Session p90 차이 [simultaneous 95% CI] | 평균 우세 catalog |
|---|---:|---:|---:|
| Gate − fixed 3R+I | −0.00001850 [−0.00002408, −0.00001359] | 0.00000000 [−0.00003647, +0.00002810] | 23/24 |
| Gate − rate-matched random | −0.00002318 [−0.00002486, −0.00002145] | −0.00002652 [−0.00004626, −0.00000748] | 24/24 |

**두 primary 비교에서 mean 개선이 재현됐고, 사전에 정한 절대 Brier 0.00001 기준도 충족했다.** v54에서 두 개선은 각각 0.00001773, 0.00002252였고 v55에서는 0.00001850, 0.00002318이었다. 128개 deployment outcome을 실제로 뽑은 두 seed에서도 gate−fixed의 평균 차이는 각각 −0.00001937, −0.00001722였고 gate−random은 −0.00002270, −0.00002304였다. Session loss p90은 random 대비 개선됐다. Fixed 3R+I 대비 p90이 첫 seed에서 같은 것은 두 분포가 동등하다는 검정 결과가 아니다. 두 번째 seed의 차이는 −0.00001962였다. [원시 요약](score55.json) · [Catalog별 paired 결과](clusters55.npz)

구매율만으로 설명할 수 있는지 한 번 더 검증했다. 같은 catalog 안의 **같은 surface와 law**에서 구매율을 맞춘 lottery를 쓰면 gate−random은 −0.00002314 [탐색적 95% CI −0.00002459, −0.00002162]였다. Gate는 같은 sensor budget으로 어느 session을 고를지에서 이득을 얻었다. 네 surface-family × rate-law 조합 각각에서도 gate가 random보다 낮은 mean을 보였다. 이 네 조합은 같은 24개 catalog를 공유한다. [독립 검산](audit55.json)

Fine과 ultra 두 PyBullet resolution에서 stick label은 24,576개 endpoint 중 794개(3.23%)가 달랐다. Ultra에서 gate−fixed는 −0.00002067 [−0.00002563, −0.00001595], gate−random은 −0.00002358 [−0.00002509, −0.00002200]이었다. 같은 solver 내 해상도를 바꿔도 mean 차이는 남았다. [Physics 감사](physics_audit55.json) · [Ultra 결과](score55.json)

## 언제 fixed impact가 더 좋은가

![새 catalog별 차이와 high-mismatch session 비율에 따른 변화](transfer55.png)

왼쪽은 새 catalog 하나당 paired mean Brier 차이, 오른쪽은 reference와 impact의 부호가 절반씩 달라지는 **e=0.5 session의 비율**을 바꾼 결과다. 0 아래면 gate의 mean이 더 낮다. 나머지 질량은 e=0과 0.25에 절반씩 배정했다. 등록된 평가에서는 e=0.5의 비율이 1/3이다. 이 비율이 **43.5%**를 넘으면 평균 Brier에서 fixed 3R+I가 앞선다. Catalog bootstrap으로 구한 교차점의 탐색적 95% 범위는 41.2–45.9%였다. Gate는 같은 구매율의 random보다 모든 배합에서 낮은 mean을 유지했다. 이 교차점은 지정된 mixture의 모집단 평균에 대한 값이며, 현장에서 개별 session을 분류할 threshold가 아니다. [사전 계획](TRANSFER_PLAN_KO.md) · [계산 및 교차점](transfer55.json)

| 실제 reference–impact mismatch e | Gate − fixed 3R+I (×10⁻⁶) | Gate − rate-matched random (×10⁻⁶) |
|---|---:|---:|
| 0 | −41.87 | −14.21 |
| 0.25 | −116.67 | −23.55 |
| 0.5 | **+103.04** | −31.77 |

이 두 비교는 다른 질문을 푼다. e=0.5에서는 gate가 I를 산 집단과 R을 산 집단 모두에서 평균적으로 I가 유리했다. I의 평균 이득은 각각 0.0002980과 0.0001577이었다. Gate가 impact가 특히 유용한 session을 고르는 능력이 남아도 나머지 session에서 impact를 생략한 비용이 생긴다. 이를 같은 구매율에서는 이기고 항상 impact를 쓰는 정책에는 지는 사례로 해석할 수 있다. 두 집단의 gap은 simulator의 두 potential outcome을 이용해 평가한 것이다. [조건부 risk 분해](transfer55.py)

## Sensing·reset 비용

교체 이후 세 정책은 session마다 4회의 weak 측정과 후속 측정 4회에 **8 contacts와 8 resets**를 지불한다. 교체 전의 공통 비용도 8 contacts와 8 resets다. Weak+reset 0.44초, reference+reset 1.04초, impact+reset T초를 가정하면 교체 후 예상 시간은 5.92 + r(T−1.04)초이며 r은 I 구매율이다.

| T 시나리오 | Frozen gate | Rate-matched random | Fixed 3R+I |
|---|---:|---:|---:|
| 0.52초 | 5.740초 | 5.740초 | 5.400초 |
| 1.04초 | 5.920초 | 5.920초 | 5.920초 |
| 2.08초 | 6.280초 | 6.280초 | 6.960초 |

Gate와 matched random은 어느 T에서도 기대 sensing·reset 비용이 같다. T가 2.08초면 gate는 fixed 3R+I보다 session당 약 0.680초를 절약한다. T가 0.52초면 오히려 약 0.340초가 더 든다. 물리 장비의 wall-clock을 측정한 결과는 없으므로 이 수치로 hardware speedup을 주장할 수 없다. [비용 원장](sensing55.json)

## GPU p90

| Policy | 512-object GPU p90 | Gate에서 대조를 뺀 차이 [동시 95% CI] |
|---|---:|---:|
| Frozen source-risk gate | 435.90 ms | — |
| Fixed 3R+I | 435.78 ms | +0.12 ms [−0.83, +2.71] |
| Rate-matched random | 435.73 ms | +0.17 ms [−1.46, +3.44] |

Datacenter3의 빈 RTX PRO 6000 Blackwell에서 24개 catalog의 새 fixture마다 네 session을 묶고 세 정책의 실행 순서 여섯 가지를 교차했다. 정책당 warmup 제외 144개 timed batch에서 CPU의 실제 센서 선택, 구입한 판독값 처리, 연속 posterior 갱신, GPU prediction과 synchronization을 포함했다. 각 반복에서 96개 timing session 중 gate와 random은 모두 정확히 35개에 I를 배정했다. Sensor readout을 만드는 시뮬레이션과 실물 capture·reset은 이 시간에 들어가지 않는다. Output replay 오차는 0이었고 측정 중 다른 GPU process는 없었다. GPU p90 차이는 두 비교 모두 불확실하다. [GPU 상세 및 비용](GPU_KO.md) · [측정 원자료 요약](timing55.json)

## 논문 기여와 남은 판정

직접적인 기여는 sensor 선택의 **세 가지 질문**을 같은 예측기로 분리한 데 있다. 첫째, 제한된 impact를 어느 session에 할당할지는 prefix를 보는 편이 무작위보다 낫다. 둘째, 그럼에도 impact를 매번 사는 선택과의 손익은 실제 mismatch의 빈도에 따라 뒤집힌다. 셋째, 평균을 줄이는 acquisition이 session loss p90도 항상 줄인다는 증거는 없다. v54에서 탐색한 규칙 하나를 v55 표본을 보기 전에 고정하고, 경험적으로 중요한 대조와 실패 조건을 함께 제시했다.

실용적 효과는 여전히 작다. 평균 개선은 fixed 3R+I 대비 약 0.012%, matched random 대비 약 0.015%이며, 더 나은 예측이 실제 slip 사고나 장비 비용을 얼마나 줄이는지 측정하지 않았다. 촉각 또는 영상으로 약한 preload의 방향과 실제 impact의 방향을 함께 읽을 수 있는 분리된 실물 session에서 e의 빈도, 오독률, reset 시간을 측정하는 것이 다음 검증이다. 일정 구매 예산이면 같은-rate random, 평균 예측이 주목적이고 impact가 충분히 저렴하면 fixed impact를 각각 유지해야 한다. 이 결과는 paper의 독립 simulated replication과 population sensitivity로 반영했다.

서버 작업 경로는 `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/fresh_surface_v55`다. 실행 순서와 seed·support, panel SHA-256, 재채점과 GPU 명령은 [표본 및 실행 기록](GENERATION_KO.md), [정량 검증](RESULTS_KO.md), [GPU 실행 기록](GPU_KO.md)에 있다. 논문 본문 및 Appendix D *Replicating acquisition gains on fresh surfaces*에 주요 비교와 한계를 수록했다.

## 문헌과 연구 심사

### 1. 논문별 질문·방법·표본·결론

| 논문 | 연구 질문과 방법 | 표본과 측정 단위 | 결과와 결론 | v55 적용의 한계 |
|---|---|---|---|---|
| Fishel & Loeb (2012) | BioTac이 얻은 tactile evidence를 바탕으로 다음 stroking movement를 고르면 texture를 더 잘 식별하는가? Bayesian exploration을 cycling·random과 비교했다. | 후보 117개 중 16개 texture의 별도 validation 측정으로 8,000회 virtual exploration을 수행했다. | Median 5회 동작으로 정분류율 95.4%를 얻었다. 무엇을 다음에 측정하는지가 texture identification에 영향을 줬다. | 반복 탐색은 독립 재질 8,000개에 해당하지 않는다. 목표가 texture ID여서 downstream slip probability의 개선 크기를 가져올 수 없다. [원문](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2012.00004/full) |
| Luo et al., *PhysiCalWorld* (2026, preprint) | Test scene에서 얻은 interaction clip을 physics descriptor token으로 압축해 model weight를 고정한 채 rollout을 개선할 수 있는가? | 서로 다른 dynamics variation을 가진 네 환경에서 rollout prediction과 visual planning을 평가했다. | Scene별 탐색 정보를 다시 사용해 rollout·planning 성능을 개선했다. 같은 물체의 추가 상호작용으로 물리를 추정한다는 발상은 이미 선행연구에 있다. | 평가 단위는 주로 trajectory와 planning task다. 네 번째 센서의 구매 횟수를 맞춰 slip의 conditional Brier gain을 분리한 실험과 직접 환산하기 어렵다. [원문](https://arxiv.org/html/2610.30125v1) |
| Koh et al., *WILDS* (ICML 2021) | 실제 발생하는 distribution shift에서 기존 방법이 얼마나 전이되는가? 현장 환경별 split을 표준화했다. | Hospital, camera trap, 촬영 장소와 시점 등 domain이 구별되는 10개 dataset. | 여러 dataset에서 OOD 성능이 떨어졌고 기존 robust training도 격차를 남겼다. 어떤 단위를 분리했는지가 평가 해석에 중요하다. | Contact policy나 Brier gain을 실험하지 않았다. 합성 surface의 family split에 외부 타당성을 자동으로 부여할 수 없다. [원문](https://proceedings.mlr.press/v139/koh21a.html) |
| Acosta, Yang & Posa (RA-L 2022) | 추정한 contact parameter로 실제 impact를 얼마나 재현하는가? 동일한 실물 trajectory와 simulator를 대조했다. | Acrylic cube의 550개 toss trajectory와 Cassie의 22개 jump. Cube는 Drake·MuJoCo·Bullet, Cassie는 두 solver. | 비탄성 충돌과 전체 움직임을 잘 재현하는 사례가 있었지만 elasticity 및 기체 모델 차이에서 오차가 남았다. | 실물 trajectory에는 동일 prefix에서 reference 또는 impact sensing을 고른 뒤의 paired forecast가 없다. v55의 acquisition 정책 자체를 이 데이터로 검증할 수 없다. [원문](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf) |

### 2–3. 문헌 종합과 지식 지도

연구 질문을 따라가면 **prefix 관측 → 다음 측정 선택 → 미래 운동 예측 → 평가 환경의 변화**로 연결된다. Fishel & Loeb은 앞의 두 단계에서 탐색을 바꿨다. Luo 등은 이미 얻은 scene-specific interaction이 세 번째 단계의 정보를 제공함을 보였다. 두 설계는 상보적이다. 달라진 측정으로 학습한 decoder만 비교하면, 어떤 sensor를 샀기 때문에 개선되었는지 식별하기 어렵다. [Fishel & Loeb](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2012.00004/full) · [Luo et al.](https://arxiv.org/html/2610.30125v1)

검증의 마지막 연결은 **training에서 분리한 단위 → 새 domain의 error → 물리 측정과의 일치**다. WILDS의 domain split은 같은 환경 속 trial만 바꾸는 방법의 한계를 보여준다. Acosta 등의 결과는 물체와 trial을 새로 생성해도 공통 solver가 실물과 다를 수 있음을 보여준다. v55는 새 synthetic surface에서 gate를 고정하는 검증을 담당한다. [Koh et al.](https://proceedings.mlr.press/v139/koh21a.html) · [Acosta et al.](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf)

### 4–7. 연구 공백·상충하는 결과·방법론·근거의 강도

- **어떤 입력이 필요한가.** 원 논문들에서 물리 탐색이 분류나 rollout을 도운 사례는 일관된다. 그러나 texture class·latent trajectory가 목표일 때와 calibrated slip probability가 목표일 때 유용한 센서가 같다는 근거는 없다. 미래 forecast를 고정하고 action을 바꾸는 것이 v55의 직접 비교다. [Fishel & Loeb](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2012.00004/full) · [Luo et al.](https://arxiv.org/html/2610.30125v1)
- **성공과 실패가 양립하는 이유.** 개별 scene에서 contextual calibration이 성공할 수 있는 것과 새 domain에서 전체 성능이 떨어질 수 있는 것은 서로 다른 모집단과 endpoint에 대한 결과다. 이 논문들 사이에서 동일한 task·sample에 대한 직접적인 상반 증거는 확인하지 못했다. v55는 aggregate mean, upper-tail quantile과 환경 조성별 mean을 구분해 보고한다. [Luo et al.](https://arxiv.org/html/2610.30125v1) · [Koh et al.](https://proceedings.mlr.press/v139/koh21a.html)
- **표본과 한계.** Virtual 재탐색, simulated trajectory, real-world dataset, 실물 impact는 각기 다른 표본 단위다. Sensor 선택의 효과를 실물에서 주장하려면 같은 물체·표면 session 안에서 두 행동을 비교하고 label과 시간도 별도로 측정해야 한다. 네 원문은 그런 조건에서의 tail gain, sensor 복구 비용, reference–impact sign mismatch의 현장 빈도를 결정하지 못한다. [Fishel & Loeb](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2012.00004/full) · [Acosta et al.](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf)

### 8. 후속 질문 열 가지

| 질문 | 메우는 공백과 중요성 | 변수와 검증 방법 |
|---|---|---|
| 1. 구매율이 같아도 prefix의 가치는 새로운 surface에서 유지되는가? | 탐색 자체와 구매 총량을 분리한다. [Fishel & Loeb](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2012.00004/full) | 구매율 고정, paired mean Brier 비교. v55에서 수행. |
| 2. Reference–impact mismatch의 현장 빈도는 얼마인가? | 합성 평가의 regime 가중치에 실증 근거를 준다. [Acosta et al.](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf) | 표면별 paired reference·impact readout을 별도 표본에서 수집하고 domain별 빈도를 추정. |
| 3. 같은 접촉 예산에서 learned descriptor가 analytical gate보다 유리한가? | 표현을 학습할 때의 추가 가치를 측정한다. [Luo et al.](https://arxiv.org/html/2610.30125v1) | Source-only encoder, 동일한 세 쌍·공통 decoder·동일한 평가 session. |
| 4. Readout noise가 집단별 이득의 부호를 바꾸는가? | 탐색 신호의 판독 오차가 어느 정도까지 허용되는지 알 수 있다. [Fishel & Loeb](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2012.00004/full) | 동일 물리 trial의 복수 센서 판독, jitter 수준별 risk. |
| 5. 다른 solver에서도 같은 prefix가 같은 행동을 정당화하는가? | Solver error가 큰 regime의 순위 반전을 찾는다. [Acosta et al.](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf) | 엔진과 restitution을 교차하고 같은 action의 realized gain을 대조. |
| 6. 교체 뒤 표면 마모가 gate를 무효화하는가? | 독립 session 효과와 시간 drift를 구분한다. [Koh et al.](https://proceedings.mlr.press/v139/koh21a.html) | 같은 표면을 날짜·마모 차수로 분리해 순차 holdout. |
| 7. Tail을 직접 목표로 한 acquisition은 평균을 얼마나 희생하는가? | Prediction 개선과 나쁜 session 보호의 차이를 측정한다. [Luo et al.](https://arxiv.org/html/2610.30125v1) | Source에서 quantile objective를 고정하고 새 domain의 mean·p90을 함께 비교. |
| 8. 첫 세 쌍의 위치 다양성이 한 쌍의 강도보다 중요한가? | 탐색 순서를 설계할 때 바꿔야 할 동작을 찾는다. [Fishel & Loeb](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2012.00004/full) | Site spacing과 preload를 factorial design으로 교차하고 probing 횟수 고정. |
| 9. 실물 reset 시간까지 포함하면 어느 정책이 유리한가? | Sensor 가격 차이가 작은 Brier 이득의 적용 여부를 바꾼다. [Acosta et al.](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf) | Readout과 reset을 별도 timestamp로 재고 time-matched risk frontier 산출. |
| 10. Marginal purchase rate를 새 fleet에서 알 수 없어도 비교가 유지되는가? | 평가용 대조와 현장 배포 기준을 연결한다. [Koh et al.](https://proceedings.mlr.press/v139/koh21a.html) | 구매율은 이전 fleet에서 추정해 고정한 뒤 시간순 새 fleet에서 calibration과 비교. |

### 9. 연구 심사

새로울 수 있는 부분은 세 번의 저렴한 관측이 **어느 session에 더 강한 관측을 배정할지** 알려주는 효과를, 같은 구매율·동일 decoder·분리된 surface라는 대조로 식별하는 것이다. Active tactile exploration이나 test-time physics calibration 자체는 이미 발표되었다. 현재 가장 가까운 learned 대안은 source interaction으로 학습해 test scene의 dynamics를 요약하는 descriptor다. 동일한 입력·label·terminal decoder에서 직접 비교하기 전에는 학습 방법보다 우수하다는 주장이 성립하지 않는다. [Fishel & Loeb](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2012.00004/full) · [Luo et al.](https://arxiv.org/html/2610.30125v1)

독립 synthetic catalog에서 재현되는 효과와 다른 물리 엔진·실물에서 재현되는 효과를 구분해야 한다. 서로 다른 표면 재료와 설치 날짜에 대해 독립적인 holdout을 구성하고, 결과를 보지 않고 정한 실용적 loss margin과 실제 elapsed sensing time을 함께 보고하는 것이 다음 검증이다. 네 편에 근거한 관련 연구 검토이므로 최초성에 대한 포괄적인 판단은 유보한다. [Koh et al.](https://proceedings.mlr.press/v139/koh21a.html) · [Acosta et al.](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf)
