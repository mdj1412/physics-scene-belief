# v47 · 같은 접촉을 보고도 어느 contour 정보를 버려야 하는지가 달라지는가?

**새 정책은 frozen v46보다 나았고, 항상 전체 profile을 유지하는 always-original에는 졌다.** 12개 새 catalog에서 held-out Brier loss는 always-original 0.151709, v47 observation-only 0.151723, frozen v46 0.153199였다. v47이 projection을 거의 쓰지 않는 것이 v46 대비 개선의 주된 이유였다. 반면 **기전 실험에서는 같은 고정렬 profile을 지우는 효과가 −0.009763에서 +0.027425로 반전됐다.** 앞의 경우 변형은 sensor에만 나타났고, 뒤에서는 하중을 받은 접촉면과 stick/slip 결과까지 바꿨다. 아래에서 음수는 loss 감소다. [Primary scores](score47.json) · [사전 분석 계획](PLAN_KO.md)

## 1. 물체와 측정에서 무엇을 독립적으로 바꿨나

같은 물체를 같은 위치로 접근하는 tip을 상상하면 된다. 카메라로 본 윤곽은 같아도 range sensor에 나타난 작은 굴곡이 광학적 효과일 수 있고, 하중을 주면 실제 표면이 휘는 방향과 연결될 수도 있다. 이 두 경우에 같은 residual을 없애는 것이 유리한지 시험했다. 매 trial의 예측 대상은 imposed impact 후 residual tangential speed가 0.020 m/s 이하인 사건의 확률이다. **이 실험에서 observation selection은 이미 끝났다.** 각 정책은 같은 factual evidence와 같은 구매 결과를 받고, terminal decoder에서 7점 profile 정보를 유지할지 결정한다.

| 독립 요인 | 두 수준 | 바뀌는 항목 | 짝지어 고정한 항목 |
|---|---|---|---|
| Optical alignment | frozen source axis와 실제 변형 방향의 squared cosine 0.10 / 0.90 | 동일 norm 1.20 mm인 signed vector를 paid range profile에 가하는 방향 | 객체, 표면, sensor noise, 접촉 전 영상, physical label |
| Mechanical coupling | inert / strong | 하중을 받을 때 prescribed local slope 변화 0 또는 nominal site에서 ±0.14. 네 family의 patch 내부 bending profile을 다르게 설정 | 동일한 factual evidence, paid profile/probe, 영상, 선택 및 decoder 확률 |
| Signed variant | hollow / bulge | Primary에서는 optical 변화와 loaded deformation의 부호를 정방향으로 연결 | 부호 대응만 뒤집은 사후 대조를 §5에 제시 |

새 contour family는 **keel, crown, channel, bead**다. 외곽에서는 높이 profile이 달라지고, 19·31 mm의 nominal site 전후 2.5 mm에서는 unloaded surface의 위치와 기울기가 같다. Keel은 loaded slope가 국소적으로 일정하고, 나머지는 patch 내에서 이차식·선형식·주기 함수에 따라 달라진다. 서로 다른 실물 재료 4종이라는 뜻은 없다. 높이 2개 × offset 3개(−0.5, 0, +0.5 mm) × finite pressure radius 2개(0.25, 2 mm)를 사용했다. 두 surface와 두 contact law의 네 조합을 네 environment에 하나씩 배정하고 catalog마다 순환했다. 이 설계는 surface·law별 단독 효과 추정을 위한 완전교차 설계는 아니다. [World generator](world47.py) · [Finite-pressure evaluator](mechanics47.py)

**조작이 입력에서 성립했는지 직접 확인했다.** 동일 물체·높이·offset에서 네 family의 영상 추정 x 차이는 모두 0이었다. 단, 참 x와의 MAE는 0.09392 mm였다. 추정 normal variance의 family 간 범위는 median과 p90이 0, p99가 0.00000728 rad², 최대가 0.003423 rad²였다. 같은 family의 factorial twins는 6,144개 별도 재렌더링 대조에서 pixel 불일치가 없었다. 실제 저장한 paired query에서 두 alignment 수준의 최대 squared-cosine 오차는 3.68×10⁻⁷이었다. Inert에서는 hollow↔bulge로 전환해도 label이 한 건도 변하지 않았고, strong에서는 **33.72%**가 바뀌었다(pooled catalog 95% CI 33.01–34.42%). 같은 signed variant에서 coupling만 바꿔 label이 변한 비율은 **16.86%**다. [Optical audit](observed_optics47.json) · [Physics audit](outcomes_checks47.json) · [형상별 결과](PHYSICS_KO.md)

평가 단위는 12개 독립 catalog support(700–711)다. 4 environment × 128개씩 총 6,144회 추출했고, support 안 중복을 제거한 prototype은 5,955개다. Primary에 배정한 1,179,648개 physical condition에는 의도적으로 동일하게 만든 null control이 포함된다. 정확히 같은 null 조건을 한 번만 세면 663,552개다. 네 번의 paid-noise 반복이나 두 optical alignment를 독립 물리 표본으로 세지 않았다. 두 catalog에서 quadrature를 12×6에서 24×12로 높여 196,608개 조건을 검산했고, 달라진 binary label은 0개였다. [GPU archive](gpu_audit47.json) · [Quadrature 검산](outcomes_checks47.json)

## 2. 정책과 정보 예산

- **Always-original**은 factual evidence로 고른 observation을 구매한 뒤 seven-point profile의 전체 likelihood를 쓴다. **Projected decoder**는 같은 profile에서 v45 source의 높이별 whitened axis 한 방향을 제거한다. Probe를 구매한 경우 두 forecast는 일치한다. 모든 정책은 같은 K256 acquisition을 쓰며, 환경당 128개 중 115개는 query view, 13개는 deployment probe를 받았다.
- **Frozen v46**은 동일한 세 pre-impact frame에서 나온 x와 주변 surface normal angle variance로 기존 lookup을 적용한다. v47에서 threshold를 다시 맞추지 않았다.
- **V47 observation-only**는 이미 구매한 한 번의 profile에서 상수·선형·이차 성분을 제거한 뒤 residual과 source axis 간 squared cosine 및 residual RMS를 구한다. Source(v45+v46)의 실제 paired Brier 차이로 x·기존 camera route·cosine·RMS 구간별 conditional risk를 학습한다. 25%를 parent mean 쪽으로 shrink하고 128개 미만의 구간은 parent로 돌아간다. 타깃의 family·axis·coupling·paired variant·label은 입력에서 제외했다. 유효한 query가 없는 probe arm에서는 original을 선택한다.
- **같은 source로 다시 학습한 camera-only ablation**도 계산했다. Profile feature를 제외하자 자료가 있는 8개 camera cell의 평균과 빈 cell에 적용한 parent 추정치가 모두 양수여서 이 규칙은 항상 original을 선택한다. v47이 frozen v46를 이기는 것만으로 profile feature의 추가 가치가 입증되지는 않는다. [정책 구현 및 훈련](ROUTER_KO.md) · [Frozen weights](policy47.json) · [Ablation](camera_ablation47.json)

실제 source에서 144개 중 89개 구간에 자료가 있었고 projection이 허용된 구간은 하나였다. 31 mm, v46이 original을 선택한 경우, residual cosine² ≤0.125, RMS (0.30, 0.60] mm인 구간이다. Source가 예측한 Brier 개선은 겨우 0.00001576이었다. 따라서 이번 policy는 “추정 alignment가 높으면 projection”이라는 고정 규칙이 아니다. [Source fit](fit_route47.py) · [Lookup](policy47.json)

## 3. Held-out Brier loss와 GPU p90

| 배포 정책 | Held-out Brier ↓ | GPU p90, ms / 512 objects [95% CI] ↓ | 전체 projection 선택률 |
|---|---:|---:|---:|
| Always-original | **0.151709** | **467.98** [467.37, 471.80] | 0% |
| Frozen v46 | 0.153199 | 475.25 [474.39, 475.61] | 21.072% |
| V47 observation-only | 0.151723 | 475.30 [473.26, 487.39] | 0.925% |

| 사전 지정 비교 | Paired Brier 차이 [동시 95% CI] | V47이 낮은 catalog | GPU p90 차이, ms [paired 95% CI] |
|---|---:|---:|---:|
| V47 − always-original | **+0.00001370** [+0.00000379, +0.00002360] | 2/12 | **+7.33** [+3.48, +19.53] |
| V47 − frozen v46 | **−0.00147622** [−0.00167104, −0.00128141] | 12/12 | +0.05 [−1.74, +12.29] |

Brier는 조건과 관측 반복을 catalog 안에서 먼저 평균한 뒤 같은 비중으로 합쳤다. 두 primary 차이에는 Bonferroni simultaneous paired t interval(df=11)을 사용했다. GPU는 할당 상태·메모리·process를 확인한 RTX PRO 6000 Blackwell 두 장에서 forecast를 나눠 계산하고, p90은 같은 GPU 한 장에서 512개씩 측정했다. 12개 대표 batch에서 방법별 warm-up 후 순서를 무작위로 바꿔 5회씩, 총 60회 측정했다. CI는 16,000회 paired catalog bootstrap이다. 시간에는 공통 acquisition, 구매 결과 masking, 필요한 raw frame 전송과 feature 계산, 단일 선택 decoder 및 결과 회수가 들어간다. 촬영·렌더링·실물 reset 및 label 생성은 제외했다. Archived forecast와 online latency 경로의 최대 probability 차이는 3.33×10⁻¹⁶이었다. [Primary scores](score47.json) · [Timing 및 비용 범위](GPU_KO.md) · [Timing 원자료](timing)

**무엇이 개선을 설명하는가?** V46이 projection하던 사례를 original로 되돌린 효과는 −0.00148992이고, v47이 새로 projection한 사례는 +0.00001370을 더했다. 두 gate는 서로 다른 구간에서 projection한다. V47의 위치·형상·구매 예산 내 projection 빈도만 보존해 무작위 배정한 보조 대조보다 실제 선택의 loss가 0.00002354 낮았다(보조 95% CI 0.00001516–0.00003193). 즉 같은 비율의 무작위 선택보다는 나았지만, 한 번도 projection하지 않는 편이 더 정확하고 빨랐다. 절대 차이 0.00001370을 실물 로봇의 비용이나 안전성으로 환산할 자료는 없다. [분해 및 빈도 대조](score47.py) · [결과](score47.json)

## 4. Axis alignment가 같아도 projection의 가치가 반전됐다

강제로 전체 query의 source axis를 지운 decoder와 original을 같은 대상에서 비교했다. Probe arm은 두 decoder가 같으므로 원래 정책과 같은 비율로 포함된다.

| 실제 optical alignment | Mechanical coupling | Original Brier | Projected Brier | Projected − original [네 cell 동시 95% CI] |
|---|---|---:|---:|---:|
| Low (0.10) | Inert | 0.143224 | 0.146228 | +0.003004 [+0.002056, +0.003952] |
| High (0.90) | Inert | 0.150969 | 0.141206 | **−0.009763** [−0.011075, −0.008452] |
| Low (0.10) | Strong | 0.166191 | 0.183518 | +0.017327 [+0.016306, +0.018347] |
| High (0.90) | Strong | 0.146453 | 0.173878 | **+0.027425** [+0.025759, +0.029091] |

Inert 조건에서는 alignment를 높일수록 projection 손익이 **−0.012768**만큼 개선됐다. Strong 조건에서는 같은 변화가 **+0.010098**만큼 악화됐다. 두 차이의 차이인 alignment × coupling interaction은 **+0.022866** [95% CI +0.022017, +0.023714]이다. 동일 high-alignment 측정에서 상대 Brier의 부호가 반전된다는 점은 두 요인이 서로 다른 역할을 한다는 직접적인 증거다. 다만 이 이득을 배포하려면 *어떤 물체에 도움이 되는지 관측만으로 알아내야* 한다. 네 cell을 사후 분리해 표시한 표 자체는 정책이 아니다. [Factorial 결과](score47.json)

![Projection의 factorial 효과와 polarity sensitivity](factorial_v47.png)

그림 A의 양수는 전체 profile을 유지하는 쪽이 유리하다는 뜻이다. 그림 B는 primary에서 고정한 optical–mechanical 부호 대응을 바꾸어 다시 채점한 보조 결과다. A의 구간은 네 비교를 동시에 보정했고, B의 각 구간은 해당 interaction의 개별 catalog 95% CI다.

## 5. 관측으로 구별되는 것과 구별되지 않는 것

**Alignment는 부분적으로 읽힌다.** 단일 paid profile의 residual cosine²만으로 high/low를 가르는 stratified AUC는 **0.7563** [95% CI 0.7555, 0.7571]이었다. 같은 물체·noise의 high 점수가 low 점수보다 높은 확률은 81.59%였다. 반면 19 mm channel/bulge에서는 AUC가 약 0.505까지 떨어졌다. Residual RMS는 오히려 low에서 더 컸다(0.486 대 0.359 mm). 원래 7차원에서 perturbation norm을 일치시켜도 quadratic 성분을 제거한 공간에서 energy가 같다는 보장은 없다. AUC는 true factor의 판별력이며 projection 이득의 AUC는 아니다. [Proxy audit](proxy_audit47.json) · [Feature 분석](ROUTER_KO.md)

**Coupling은 이 설계의 관측으로 식별되지 않는다.** Coupling을 바꿔도 같은 물체의 모든 paid observation, pre-impact pixels, route, 후보 확률은 그대로이고 evaluator의 loaded mechanics만 바뀐다. 동일 입력을 받는 함수는 두 twin에서 다른 결정을 낼 수 없다. 그렇다고 모든 observation-only 정책이 항상 original에 진다는 명제는 나오지 않는다. 분포의 평균과 관측 가능한 하위 집단에 따라 같은 결정을 내리는 규칙도 이득을 얻을 수 있다.

이를 수치로 확인하기 위해 탐색적 **leave-one-catalog-out group rule**을 만들었다. 다른 11개 target catalog의 label로 family·true alignment·height·offset 구간별 평균 projection cost를 계산하고 남은 catalog를 채점했다. 두 rule 모두 privileged group 정보를 쓰되 하나에만 coupling을 추가했다. Brier는 0.151402에서 0.148815로 감소했고 차이는 **−0.002588** [−0.002860, −0.002315]였다. 이는 이 coarse rule이 추가 정보에서 회수한 양이다. 배포 정책의 성능이나 완벽한 정보의 이론적 상한으로 쓸 수 없다. [Group rule 구현](score47.py) · [결과](score47.json)

**변형의 부호 대응이 기전 해석을 바꾼다.** Primary에서는 optical hollow/bulge의 부호를 실제 load-induced 변화의 부호와 고정하여 연결했다. 같은 물체의 두 loaded outcome을 맞바꾸면 optical alignment·noise·decoder 확률·개별적인 변형의 크기·coupling별 label flip 비율은 모두 유지하면서 부호의 대응만 바꿀 수 있다. Target 결과를 확인한 뒤 시행한 이 사후 replay에서 high/strong의 projection cost는 **+0.027425 → −0.043393**으로 바뀌었다. 정방향과 역방향을 50:50으로 섞으면 high/strong은 −0.007984, interaction은 **+0.002607**로 작아진다. 이로써 “변형이 stick/slip을 바꾸면 정렬된 축을 버릴 때 항상 손해”라는 해석을 배제했다. 원래 interaction이 정방향으로 연결된 synthetic population에서 관찰됐다는 조건이 중요하다. [Polarity replay](polarity47.py) · [사후 결과](polarity47.json)

수학적으로도 이유가 보인다. 같은 예측 확률에서 projection의 상대 loss를 D라고 두면, **D(strong)−D(inert) = −2 × (p_projected−p_original) × (Y_strong−Y_inert)**가 각 trial에서 성립한다. Label이 바뀐 횟수가 같아도, 어느 쪽으로 바뀌었는지와 decoder의 확률 이동 방향이 어떻게 짝지어졌느냐에 따라 평균 효과가 달라진다. Pair polarity 재채점에서 이 항등식과 실제 Brier 차이를 대조했다. [검산 구현](polarity47.py)

## 6. 논문 기여와 다음 의사결정

**정책 성능에 대한 판정은 부정적이다.** 새 정책은 두 baseline 모두에서 개선돼야 한다는 기준을 충족하지 못했다. Source와 동일한 정보로 다시 fit한 camera-only ablation도 always-original이므로 v46 대비 개선을 새로운 profile feature의 성과로 주장하기 어렵다. 강한 대안 설명은 source에 이미 있던 “대부분의 조건에서 유지하는 편이 낫다”는 정보다. 실제로 gate가 추가한 projection은 작은 양의 loss를 만들었다.

**논문 본문에는 측정·작동 기전을 분리한 결과가 들어갈 만하다.** 같은 접촉 영상·같은 optical alignment에서 load-bearing 여부가 projection 손익을 반전시킬 수 있었고, 그 반전이 optical–mechanical polarity에 의존한다는 것까지 확인했다. Primary 2×2와 polarity control을 함께 제시해야 독자가 synthetic generator가 부여한 상관구조의 역할을 판단할 수 있다. v46에서 관찰한 OOD 오류의 원인 전부를 v47이 입증한 것은 아니다. v47은 별도의 설계에서 해당 경로가 실제로 존재할 수 있음을 보인다.

**다음 한 번은 약한 preload로 숨은 정보를 관측하는 실험이 적절하다.** 서로 다른 optical finish와 backing stiffness를 조합한 별도 물체군 또는 독립 compliant solver에서, 같은 pre-impact profile을 가진 쌍에 약한 preload 1회만 준다. Displacement 혹은 force로 loaded slope의 방향을 추정하는 rule, 같은 예산의 fixed probe, always-original, contact-aware learned decoder를 미리 고정한다. 물체 family와 surface session 전체를 hold out하고, (1) coupling과 polarity를 구분하는지, (2) 최종 Brier를 always-original보다 낮추는지, (3) sensing/reset까지 포함한 비용이 정당화되는지를 따로 판정한다. 현재 네 family는 local base와 물성 생성기를 공유하고, finite-pressure solver에 정해진 deflection을 넣었다. 실물 영상·FEM 검증과 독립 camera session은 남아 있다. [NDCF 실물 visuo-tactile 연구](https://www.roboticsproceedings.org/rss19/p080.pdf) · [실제 impact에서 simulator 검증](https://arxiv.org/abs/2110.00541)

## 선행연구 6편에서 본 v47의 위치

아래는 이 실험의 가설 또는 비교 방법을 바꾸는 6편을 선별한 검토다. 표의 표본 단위는 원 연구마다 다르다. 서로 다른 endpoint의 수치를 직접 순위 매기지 않았다.

| 연구 | 연구 질문·방법 | 표본·평가 | 결과와 해석의 경계 |
|---|---|---|---|
| Kim et al., *Im2Contact* | RGB-D만으로 로봇이 쥔 물체의 외부 contact를 국소화할 수 있는지 물었다. Simulation에서 학습한 visual model을 실제 로봇에서 평가했다. | Simulation 평가 500 episodes, 실제 상호작용 30 episodes·12,362 frames | 여러 형태의 contact를 국소화했다. 현재 외부 contact 위치를 찾는 성능으로 미래 stick에 대한 decoder 선택의 가치를 추론할 수는 없다. [원문](https://proceedings.mlr.press/v229/kim23b/kim23b.pdf) |
| Pfrommer et al., *ContactNets* | Signed distance와 contact-frame Jacobian을 trajectory로 학습해 impact·stiction의 불연속성을 표현했다. | 실제 cube toss 750회에서 정제한 570개 trajectory. 여러 training-set 크기로 unstructured baseline과 비교 | 구조가 있는 모델이 물리적으로 타당한 trajectory를 data-efficient하게 예측했다. 여러 compliant family에서 부분 profile 제거의 조건부 Brier는 별도 문제다. [원문](https://proceedings.mlr.press/v155/pfrommer21a/pfrommer21a.pdf) |
| Van der Merwe et al., *Neural Deforming Contact Fields* | Partial point cloud와 force/torque로 loaded shape와 contact patch를 함께 추정했다. Simulated supervision을 사용했다. | 실물 검증은 Franka에 장착한 46 mm cube sponge, 두 depth camera와 force/torque sensor. Material fitting에 flat press 16회 | Sim-to-real에서 point-cloud baseline보다 나은 결과를 보고했다. 접촉 **후**의 wrench는 v47의 pre-impact observer가 갖지 못하는 정보다. 이 차이가 matched-coupling 실험에 직접적인 후속 관측을 제시한다. [원문](https://www.roboticsproceedings.org/rss19/p080.pdf) |
| Acosta et al., *Validating Robotics Simulators on Real-World Impacts* | 실제 impact trajectory로 contact parameter를 맞춘 뒤 Drake·MuJoCo·Bullet을 대조했다. | Tossed cube와 착지하는 Cassie biped, 두 물리 시스템 | 비탄성·탄성 regime 및 전체 운동과 세부 운동에서 fidelity가 달랐다. 하나의 finite-pressure solver에서 식별한 손익의 크기를 실물 보정량으로 사용할 근거는 약하다. [원문](https://arxiv.org/abs/2110.00541) |
| Veitch et al., *Counterfactual Invariance to Spurious Correlations in Text Classification* | 어떤 feature 변형을 무시해야 하는지를 causal structure로 정의하고 OOD 조건을 분석했다. | 이론과 text-classification 실험. 물리 접촉 표본은 없다. | 어떤 부분을 바꿔도 되는지는 label과 feature가 생기는 경로에 따라 달라진다. v47에서 sensor 변형과 loaded shape 변형을 따로 조작한 이유다. 이 이론이 v47에서 특정 router의 개선을 예측하지는 않는다. [원문](https://proceedings.neurips.cc/paper_files/paper/2021/file/8710ef761bbb29a6f9d12e4ef8e4379c-Paper.pdf) |
| Luo et al., *PhysiCalWorld* (2026 preprint) | Exploratory interaction clip에서 physics token을 만들어 test scene의 action-conditioned rollout에 재사용했다. | 물성이 달라지는 네 환경에서 latent prediction과 downstream task를 평가 | Interaction이 알려주는 instance-specific dynamics를 활용할 수 있음을 보고했다. 추가 clip을 허용하는 setting과 같은 paid prefix에서 terminal decoder만 바꾸는 setting은 정보 예산이 다르다. [원문](https://arxiv.org/abs/2610.30125) |

**문헌의 구조와 겉보기 충돌.** 연구들을 하나의 사슬로 놓으면, unloaded geometry의 관측(Im2Contact의 visual localization과 연결) → load 이후 contact patch·force(NDCF) → impact outcome(ContactNets, Acosta et al.) → 특정 feature를 버릴 수 있는지에 대한 판단(Veitch et al.)으로 이어진다. PhysiCalWorld의 exploratory clip은 이 사슬에 추가 interaction을 제공한다. NDCF와 Im2Contact의 긍정적 결과와 v46의 transfer 실패는 같은 가설의 상반된 결과가 아니다. 센서가 사용 가능한 시점, geometry를 재구성하는지 future event를 채점하는지, 비교 모델과 접촉 재질이 다르기 때문이다. [Im2Contact](https://proceedings.mlr.press/v229/kim23b.html) · [NDCF](https://roboticsconference.org/2023/program/papers/080/) · [PhysiCalWorld](https://arxiv.org/abs/2610.30125)

**어디까지 근거가 있는가.** 휴지 상태의 contour와 하중을 받은 contact surface를 구분해야 한다는 점은 visuo-tactile 연구와 실물 simulator 검증 모두에 부합한다. Learned dynamics와 실제 exploratory measurement가 도움을 줄 수 있다는 근거는 과제와 데이터에 조건부다. 선정한 논문들은 same location·matched normal variance에서 profile axis와 loaded deformation을 교차해 동일 decoder의 Brier를 재는 비교를 제공하지 않는다. 이 실험 설계의 추가 가치는 그 조건부 차이를 직접 측정하는 데 있다. 문헌 전체에 대한 최초성, real-world 안전성, 일반적인 OOD 불가능성은 이 표에서 도출되지 않는다. [NDCF](https://roboticsconference.org/2023/program/papers/080/) · [ContactNets](https://proceedings.mlr.press/v155/pfrommer21a.html) · [실물 impact](https://arxiv.org/abs/2110.00541) · [Counterfactual invariance](https://proceedings.neurips.cc/paper_files/paper/2021/hash/8710ef761bbb29a6f9d12e4ef8e4379c-Abstract.html)

### 후속 연구 질문 10개

아래는 추가 실험을 고르기 위한 제안이다. 각 행의 오른쪽에 가장 직접적인 관련 문헌을 표시했다.

| 질문 | 미해결 부분과 필요한 실험 |
|---|---|
| 1. 작은 preload 하나로 hidden coupling을 구별하는가? | 동일 prefix에 load 전후 displacement나 wrench 한 번을 추가한다. Paired Brier gain과 sensing/reset 비용을 함께 재면 현재 정보 한계가 실험적으로 풀리는지 알 수 있다. [NDCF](https://roboticsconference.org/2023/program/papers/080/) |
| 2. 실제 물체에서도 sensor axis와 loaded mode가 분리되는가? | 서로 다른 표면 마감과 backing stiffness를 교차한 시험편으로 optical profile, force, slip을 독립 측정한다. 시험편·날짜 단위 holdout을 쓴다. [NDCF](https://roboticsconference.org/2023/program/papers/080/) |
| 3. Query height가 어긋나도 같은 결론인가? | Contact localization error를 독립 calibration session에서 추정한 분포로 주입해 route와 Brier의 민감도를 추적한다. [Im2Contact](https://proceedings.mlr.press/v229/kim23b.html) |
| 4. 물성이 달라져도 paired-mechanism 결과가 남는가? | Compliant simulator와 실제 elastic/inelastic material을 나눠 정렬×변형 interaction을 비교한다. [실물 impact](https://arxiv.org/abs/2110.00541) |
| 5. Contact-aware learned decoder가 projection의 이득을 흡수하는가? | 동일 paid measurement와 source training budget으로 contact Jacobian 기반 predictor를 학습한 뒤 held-out Brier를 비교한다. [ContactNets](https://proceedings.mlr.press/v155/pfrommer21a.html) |
| 6. 소수의 큰 오차는 sensor signal-to-noise ratio가 설명하는가? | Optical amplitude를 고정하고 range noise를 바꿔 true alignment에 대한 proxy discrimination과 route regret을 따로 잰다. [Im2Contact](https://proceedings.mlr.press/v229/kim23b.html) |
| 7. 자연스러운 object population에서도 같은 결정을 하는가? | 실제 독립 표본에서 alignment·coupling prevalence를 추정하고, 고정 policy의 Brier를 그 분포로 재가중한다. [실물 impact](https://arxiv.org/abs/2110.00541) |
| 8. 표면 session마다 소량의 calibration을 사는 것이 이득인가? | 같은 총 관측 예산으로 clip 기반 latent descriptor, 명시적 stiffness estimate, fixed decoder를 대조한다. [PhysiCalWorld](https://arxiv.org/abs/2610.30125) |
| 9. 동일한 proxy라도 causal pathway가 달라지면 선택이 달라져야 하는가? | Sensor-only 변형과 load-bearing 변형의 비율을 독립 randomization하고, source에서 정한 손익 예측의 부호가 target에서 재현되는지 확인한다. [Counterfactual invariance](https://proceedings.neurips.cc/paper_files/paper/2021/hash/8710ef761bbb29a6f9d12e4ef8e4379c-Abstract.html) |
| 10. Binary stick 기준이 결과를 결정하는가? | 사전 지정한 residual-speed threshold 범위와 continuous velocity error를 함께 평가하고, 독립 실물 velocity 계측의 해상도와 비교한다. [ContactNets](https://proceedings.mlr.press/v155/pfrommer21a.html) · [실물 impact](https://arxiv.org/abs/2110.00541) |

## 재현 경로

서버 작업 디렉터리는 `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/factorial_projection_v47`이다. Source-only router를 동결하고 GPU/평가 코드 및 물리 코드의 checksum을 확인한 뒤 2026-11-16 11:32:48 UTC에 target label 생성을 허용했다. Primary Brier는 서버의 해당 폴더와 기존 project 환경에서 `../continuous_v2/.venv/bin/python score47.py`로 재계산할 수 있다. GPU p90은 `gpu_audit47.py`, polarity sensitivity는 `polarity47.py`로 기존 archive에서 재계산한다. 멀티 에이전트가 physics·영상 검산, policy·source ablation, GPU forecast·timing을 각각 담당했고 primary 채점은 별도 코드로 수행했다. [실행 동결](master47.sha256) · [Target release](TARGET_GO) · [GPU run metadata](runs) · [Catalog별 prediction](forecast) · [Physical outcome](labels)
