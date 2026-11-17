# v71 · 새 racket rubber에서 release 방향과 rebound 크기를 몇 번의 impact로 보정할 수 있는가?

Grass D.TecS의 새 impact에서 v70의 예측 부호를 뒤집어야 한다는 점은 calibration 4회로 대부분 감지됐다. 그러나 8회에서 routing만 뒤집었을 때의 Grass spin MAE는 40.89 rad/s였다. 같은 관측으로 두 expert의 연속 출력까지 보정하면 23.70 rad/s로 줄었다. 여섯 제품으로 범위를 넓히면 다른 결론이 나온다. 8회에서 두 expert에 같은 방식의 empirical Bayes prior를 주어도 single affine에 비해 velocity는 다소 나쁘고 spin은 다소 좋았다. 네 계수의 elastic impulse model이 같은 예산에서 가장 작은 제품 평균 velocity·spin 오차를 냈다. [전체 수치](analysis/summary.json)

## 1. 무엇을 반전이라고 부르는가?

접선 방향의 상대속도를 impact 전에는 $s=v_x-Rω_y$, 이탈 시점에는 $s'=v'_x-Rω'_y$로 정의한다. Ball radius는 0.020 m다. v70의 두 expert는 $s'$의 부호로 나뉘고, 배포 시에는 입사 상태로부터 source gate가 해당 부호를 예측한다. v71의 calibration은 측정한 이탈 부호와 source gate가 얼마나 자주 어긋나는지 관찰한다. 제품명이나 sponge 이름은 detector에 입력하지 않는다.

Grass의 held-out 299회에서 source gate는 299회 모두 입사 부호와 반대인 이탈을 예측했다. 실측에서는 **28회만** 부호가 바뀌었고 **271회는 유지**됐다. 따라서 Grass에서 gate를 뒤집는다는 관측은 *이 제품은 대개 입사 방향을 유지하는데 source 모델은 방향이 바뀐다고 예측했다*는 뜻이다. 비교로 Hexer Duro에서는 506회 중 505회, Power 3에서는 321회 중 320회에서 실측 입·출사 부호가 달랐다. 이 데이터에서 부호는 궤적과 spin으로 계산한 이탈 상태다. 접촉 중 국소 변형이나 stick/slip 이력은 관측하지 않았다. [제품별 부호 집계](analysis/polarity.json) · [원자료 및 측정법](https://arxiv.org/html/2604.11349v1)

## 2. 제품 전체를 제외한 평가와 동일한 관측 예산

공개 racket dataset의 10개 configuration을 6개 rubber 제품으로 묶었다. 한 제품의 모든 configuration을 source 학습에서 제외하고 나머지 다섯 제품의 공식 train으로 prior를 만들었다. 제외한 제품의 **공식 train**에서만 $K=0,1,2,4,8,16,32$개의 calibration impact를 구매하고, 분리된 **공식 test**의 이탈 velocity와 spin을 예측했다. 여기서 구매는 기록된 impact의 정답을 공개하는 절차다. 매번 새 물체를 충돌시킨 실험은 아니다. 공개 split에 포함된 held-out impact는 Hexer Duro 506, Power 3 321, Grass D.TecS 299, Diamant 168, Blowfish 171, A-B-S II soft 158로 총 1,623회다. [공식 데이터와 버전](https://github.com/gossardt/tt_rackets/tree/159c6f040dd7832570b537b3b5fd00a97fc28c15)

제품별로 256개의 무복원 calibration 순열을 생성하고, 같은 순열에서 앞의 K개를 모든 방법에 제공했다. 순열 간 impact 재사용이 있으므로 256은 독립 rubber나 session 수가 아니다. 각 impact에서 $v'_x,v'_z$의 vector error와 $ω'_y$의 absolute error를 계산했다. 본문의 macro 평균은 test impact 수에 관계없이 여섯 제품을 동일하게 가중한다. 비교별 paired difference, test 크기로 가중한 micro 평균, 각 제품의 draw 분위수도 보존했다. 기존 v70에서 Grass의 반전과 공식 test를 확인했으므로 이번 평가는 모델 fitting에 대한 holdout이며, 현상 발견 후의 독립적인 신규 제품 검증은 남아 있다. [평가 계획](core/PLAN_KO.md) · [paired 결과](analysis/summary.json)

| 방법 | Calibration impact 한 건이 바꾸는 것 |
|---|---|
| Gate only | 실측 이탈 부호가 source gate와 다른 횟수가 절반을 넘으면 두 expert의 routing을 뒤집는다. 동률이면 원래 routing을 유지한다. |
| Gate + experts | 같은 vote를 쓰고, 관측한 분기로 나누어 shared affine residual과 분기별 차이를 학습한다. 분기별 항의 ridge penalty는 shared 항의 4배다. |
| Frozen routing + experts | 배포 중 쓰는 고정 partition으로 calibration도 분할한다. 부호를 명시적으로 맞춰야 연속 예측이 좋아지는지 보는 대조군이다. |
| Inclined rigid | 동일 관측으로 tilt, friction, restitution을 조정하고 source 예측에서 벗어나는 정도를 정규화한다. |
| Single affine | 같은 입력으로 세 출력을 함께 추정한다. 일반 ridge와, source 제품 사이 12개 계수의 covariance를 활용한 empirical Bayes(EB)를 모두 비교한다. |
| Gate + experts (EB) | 두 head에도 source 제품 간 coefficient covariance를 쓰고 single affine와 같은 절차로 prior 강도를 고른다. |
| Elastic impulse | 접선 impulse가 translation과 spin을 함께 바꾸도록 묶는다. 입사 속도에 따라 변하는 네 계수만 같은 관측으로 갱신한다. |

보정 강도 선택에는 outer target의 test를 쓰지 않았다. 기본 two-expert 방법은 다섯 source 안에서 다시 한 제품을 제외해 penalty를 골랐다. 여섯 fold 모두 λ=0.25였고, 더 작은 λ 두 개를 추가한 source-only 검산에서도 선택이 유지됐다. 추가 elastic impulse 비교는 모형식을 별도로 기록하고 평가했다. [방법 구현과 검증](core/CORE_KO.md) · [baseline 추정법](baselines/BASELINES_KO.md) · [추가 비교 설계](analysis/PLAN_KO.md)

## 3. Grass에서 몇 회가 필요한가?

| Calibration impacts | Gate를 뒤집은 draw | 고정 K 단측 기준 통과 | Held-out release sign accuracy | Gate only, velocity / spin | Gate + experts, velocity / spin |
|---:|---:|---:|---:|---:|---:|
| 0 | 0/256 | 0/256 | 9.36% | 1.584 / 119.10 | 1.584 / 119.10 |
| 1 | 237/256 | 0/256 | 84.60% | 0.736 / 46.41 | 0.664 / 41.73 |
| 2 | 222/256 | 0/256 | 79.84% | 0.790 / 51.01 | 0.668 / 41.94 |
| 4 | 249/256 | 0/256 | 88.41% | 0.694 / 42.73 | 0.532 / 30.20 |
| 8 | 255/256 | 227/256 | 90.32% | 0.672 / 40.89 | 0.455 / 23.70 |
| 16 | 256/256 | 253/256 | 90.64% | 0.668 / 40.58 | 0.388 / 19.96 |
| 32 | 256/256 | 256/256 | 90.64% | 0.668 / 40.58 | 0.346 / 17.93 |

한 번의 고정된 K에서 mismatch가 우연히 절반을 넘었을 가능성을 따로 계산했다. $H_0:p=0.5$인 one-sided binomial 기준(0.05)에서는 4/4가 불일치해도 p=0.0625다. 따라서 4회로 유용한 결정을 자주 내릴 수 있지만 이 기준으로 증거를 확보할 수 있는 최소 관측 수는 더 크다. 8회에서 7회 이상 불일치하면 통과한다. 이 기준은 반복 확인하며 멈추는 sequential test로 검증하지 않았다. [독립 검산](review/REFERENCE_KO.md)

Grass calibration pool의 불일치는 1,195회 중 1,111회다. 모든 가능한 무복원 표본을 고려한 hypergeometric 계산에서 8회로 gate를 뒤집을 정확한 확률은 **99.87%**, 단측 기준을 통과할 확률은 **89.63%**다. 같은 방식에서 A-B-S II soft의 flip 확률은 **34.17%**였다. 나머지 네 제품은 각각 0.001% 미만이었다. A-B-S II soft에서는 불일치가 절반 부근인데도 158개 test impact 전체에 반대 routing을 적용하면 원래 routing의 2.134 m/s·115.49 rad/s가 1.517 m/s·57.42 rad/s가 된다. 같은 한 표라도 잘못된 expert를 고른 피해 크기가 impact마다 다르다는 사례다. 이 강제 반전의 손익은 test를 본 후의 진단이다. [전수 확률과 두 routing의 오차](review/finite_reference.json)

## 4. 같은 8회에서 어느 방법의 연속 예측이 좋은가?

각 칸은 **held-out velocity vector error (m/s) / spin MAE (rad/s)**다. 두 값 모두 작을수록 좋다.

| 방법 | Grass D.TecS | 여섯 제품 macro 평균 |
|---|---:|---:|
| Frozen v70 (0회 대조) | 1.584 / 119.10 | 1.065 / 67.16 |
| Gate only | 0.672 / 40.89 | 0.878 / 50.76 |
| Gate + experts | 0.455 / 23.70 | 0.505 / 28.48 |
| Frozen routing + experts | 0.632 / 35.80 | 0.499 / 27.81 |
| Inclined rigid | 0.582 / 40.15 | 0.934 / 61.82 |
| Single affine (ridge) | 0.498 / 29.25 | 0.406 / 20.89 |
| Single affine (EB) | 0.436 / 25.70 | 0.348 / 18.15 |
| Gate + experts (EB) | 0.398 / 21.41 | 0.359 / 17.94 |
| Elastic impulse | 0.397 / 21.97 | 0.300 / 15.06 |

**Expert를 갱신하는 효과.** 기본 gate + experts는 gate only보다 여섯 제품 모두에서 두 오차를 줄였다. K=8의 paired macro 차이는 −0.372 m/s와 −22.28 rad/s였다. 그러나 같은 방식은 single affine (EB)보다 평균 **+0.157 m/s, +10.33 rad/s**였고 velocity는 0/6, spin은 1/6개 제품에서 이겼다. 제품 한 개씩을 빼도 두 차이의 범위는 각각 +0.133~+0.185와 +9.04~+12.79였다. 256 paired draw의 10–90 percentile은 +0.114~+0.206과 +7.01~+13.92다. 이 구간은 calibration 샘플 선택의 변동을 나타낸다. 새 제품 모집단에 대한 confidence interval로 해석할 수는 없다. [paired 결과](analysis/summary.json)

**Prior를 맞춘 대조.** Two-expert 쪽에도 EB prior를 주면 8회에서 single affine 대비 제품 평균 차이가 **+0.011 m/s, −0.21 rad/s**까지 작아진다. 16회에서는 두 방법이 각각 0.327 / 16.02와 0.329 / 17.12다. 이때도 제품별로 두 지표 각각 3승 3패다. 8회의 Grass만 보면 EB two-expert의 21.41 rad/s가 elastic impulse의 21.97 rad/s보다 조금 낮고, velocity는 두 방법이 0.398과 0.397 m/s로 비슷하다. 반면 8회에 A-B-S II soft의 spin은 inclined rigid가 20.00 rad/s로 elastic impulse의 23.99보다 낮다. [동일 prior 추정 절차](baselines/gated_eb.json) · [제품별 결과](analysis/summary.json)

![Calibration budget에 따른 gate 반전 확률과 제품 평균 velocity 및 spin 오차](analysis/calibration_curves.png)

왼쪽은 고정된 calibration pool에서 무복원 추출할 때의 정확한 확률이다. 가운데와 오른쪽은 여섯 제품의 macro 평균이다. 1–2회에는 single affine (EB)가 elastic impulse보다 두 오차가 작고, 4·8·16·32회에는 순서가 바뀐다. 8회의 elastic impulse는 single affine (EB)보다 평균 **−0.048 m/s, −3.09 rad/s**로, velocity는 6/6, spin은 4/6개 제품에서 개선됐다.

**물리식으로 확인한 대안 설명.** Thin-shell ball의 $I/(mR^2)=2/3$을 두고 tangential impulse를 $Δv_x=-αs$, $Δ(Rω_y)=1.5αs$로 쓰면 $s'=(1-2.5α)s$다. 따라서 $α$가 0.4를 지나는 것만으로 gate 없이도 이탈 방향이 바뀐다. $α$가 $|s|$에, normal restitution이 $|v_z|$에 각각 affine로 의존하게 했다. 두 intercept와 두 slope의 source prior를 같은 impact로 갱신했다. 이 식이 손실을 낮췄다는 사실은 translation과 spin을 묶는 제약을 지지한다. Ball과 pimple 사이에서 실제로 발생한 friction, contact patch, 변형의 기전을 유일하게 식별하지는 않는다. [구현](analysis/elastic.py) · [원 연구의 state-dependent impulse 모델](https://arxiv.org/html/2604.11349v1)

## 5. 같은 제품에서 sponge만 바꾸면 무엇이 전이되는가?

Grass의 두 configuration에서 calibration과 test의 구성을 교차했다. 같은 열에서는 동일한 held-out impact를 평가한다. K=8, 기본 gate + experts의 결과다.

| Calibration 구성 | Test 5, 1.2 mm sponge | Test 6, OX |
|---|---:|---:|
| 5, 1.2 mm sponge | 0.403 / 18.60 | 0.505 / 28.09 |
| 6, OX | 0.537 / 28.18 | 0.390 / 20.78 |

Calibration 구성 5에서 98.83%, 구성 6에서 100%의 draw가 같은 gate 반전을 선택했다. 어느 쪽으로 calibration해도 부호는 비교적 쉽게 옮겨갔지만, **동일한 test를 두고 calibration 구성을 바꾸면 두 방향 모두 연속 예측 오차가 커졌다.** 같은 Grass 제품의 두 configuration이므로 이 차이를 독립적인 long-pips 제품 두 개의 반복 검증으로 세지는 않았다. [전체 교차 실험](analysis/sponge.csv) · [구성 메타데이터](https://github.com/gossardt/tt_rackets/tree/159c6f040dd7832570b537b3b5fd00a97fc28c15)

## 6. 논문에서 유지할 주장과 다음 확인 실험

가장 방어 가능한 기여는 **새 surface에서 쉽게 알아낼 수 있는 release 부호와, 같은 관측 예산으로 실제 rebound 크기를 잘 예측하는 능력을 분리해 정량화한 평가**다. Grass의 방향 전환, 같은 제품의 sponge 교차, A-B-S II soft의 비대칭적인 오차가 서로 다른 근거를 준다. 물리식을 쓰거나 source covariance를 학습한 강한 대안이 평균 순위를 바꿨다는 결과는 이 분해를 본문에 넣을 이유다. 원 저자도 이미 online adaptation을 평가했고 ALPaCA에서도 source prior로 소수 관측에 적응했다. 신규성은 few-shot update 자체나 gate의 보편적 우위로 방어하기 어렵다. [Gossard §VI](https://arxiv.org/html/2604.11349v1) · [ALPaCA](https://arxiv.org/abs/1807.08912)

다음 확인 실험으로는 **v70·v71에서 보지 않은 long-pips 및 anti-spin 제품**에서 K=4·8을 미리 고정하고 majority vote, paid impact의 연속 loss 차이를 정규화한 routing, single affine EB, elastic impulse를 비교하는 것이 적절하다. 제품마다 독립된 시편·날짜를 확보하고 입사 속도 및 spin 범위를 맞춘다. 주 지표는 제품별 paired velocity·spin 오차, 진단 지표는 Grass형 부호 감지와 $|s'|$가 작은 impact를 제외한 sensitivity로 둔다. 동일 제품의 256 draw가 줄여 주지 못한 product·session uncertainty를 정면으로 평가할 수 있다. 새 데이터 전까지 active probe selection, 실측 contact patch의 독립적인 정보량, 비용·latency 우위는 보류한다. v71은 무작위로 기록된 impact를 공개했고 연산 예산 동일화나 물리적 reset 비용 측정은 시행하지 않았다.

## 7. 실행과 검산

Slurm allocation 2744471의 node56에서 compute process가 없고 utilization 0인 NVIDIA RTX PRO 6000 Blackwell Server Edition을 확인한 뒤 CUDA float64로 실행했다. 핵심 방법, baseline, EB 대조군을 별도 작업으로 실행했다. 독립 구현은 10,752개 prefix에서 flip과 gate-only loss를 검산했고 최대 차이는 velocity 1.33×10⁻¹⁵ m/s, spin 7.11×10⁻¹⁴ rad/s였다. Elastic impulse의 별도 CUDA replay도 10,752개를 모두 검산했으며 최대 차이는 5.87×10⁻¹⁴ m/s, 5.45×10⁻¹² rad/s 이하였다. 이 검산은 부동소수점 구현 일치를 검사한다. [실행 환경](core/fit.json) · [gate 검산](review/draw_audit.json) · [elastic impulse CUDA 검산](analysis/elastic_cuda.json)

재현 bundle에는 v70 frozen coefficient, v71이 읽은 공개 train/test CSV와 upstream license, 실제 draw index, 결과, 그림, 코드가 들어 있다. 두 버전 폴더를 나란히 둔 뒤 v71 안에서 실행한다. CUDA가 있는 Slurm 환경에서 아래 순서를 사용한다. 사용한 PyTorch는 2.7.1+cu128이다. 그림 생성에는 Matplotlib가 필요하다.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python core/study.py
python baselines/compare.py
python baselines/summarize.py
python baselines/gated_eb.py
python analysis/elastic.py
python analysis/verify_elastic_cuda.py
python analysis/expert_cov.py
python analysis/sponge.py
python analysis/polarity.py
python core/sensitivity.py
python review/finite_reference.py
python review/check_draws.py
python analysis/aggregate.py
python analysis/figure.py
```

## 부록 · 문헌 검토와 연구 평가


### 1. 연구별 질문·방법·표본·결과·한계

문헌의 공통 질문은 새 환경에서 적은 관측으로 **어느 부분을 바꿔야 하는지**이다. 아래에서는 원 연구 결과와 v71에서 확인할 질문을 다른 열에 두었다. Jia 등의 논문은 발행사 공개 abstract 범위로 한정했다.

| 원 연구와 질문 | 방법과 표본 | 관찰과 결론 | v71에 적용되는 범위 |
|---|---|---|---|
| Gossard (2026, preprint). Racket마다 다른 bounce dynamics를 예측하고 온라인에서 보정할 수 있는가? | 10개 구성, 6개 rubber 제품의 planar impacts. 공식 배포 원본 8,184행. 실측 velocity와 spin에서 state-dependent restitution 및 impulse response를 fit했다. §VI는 inverted 구성 1·2·3·4·9의 train data로 prior를 구성하고 test observations를 순차 반영했다. | Online 계산에 (1, 표면 상대속도, normal velocity) 특징의 linear prior를 썼다. 저자는 약 10회 burn-in, 약 30회 이후 velocity 오차 개선, 평균 약 0.06 m/s·최대 약 0.17 m/s 감소를 보고했다. | 같은 rubber type의 prior, velocity 중심 online curve다. 제품 전체를 미리 제외한 Grass의 release 부호 감지, expert-only·gate-only 대조, spin 개선 여부를 이 수치로 정할 수 없다. [원문 §III·V·VI, Fig. 8–9](https://arxiv.org/html/2604.11349v1) · [공식 데이터](https://github.com/gossardt/tt_rackets/tree/159c6f040dd7832570b537b3b5fd00a97fc28c15) |
| Jia 등. 다른 rubber의 속도·각도·회전 반응은 어떤 실측으로 비교되는가? | 네 종류 표면, 무회전의 다섯 입사속도와 topspin/backspin의 다섯 회전속도, 1,500 fps 측면 촬영 후 SIMI-Motion 분석. | 무회전 조건에서 입사속도와 restitution 사이 linear fit의 R²가 0.84를 넘었다고 보고했다. | 다른 실험진에서 표면과 입사조건을 바꾼 사례다. 조건평균·회귀 적합도로 held-out spin error나 few-shot gate 효과를 추정하기는 어렵다. [발행사 원문 abstract](https://doi.org/10.1177/17543371241310338) |
| Rinaldi 등 (2019). Surface 층과 입사각이 rebound를 어떻게 바꾸는가? | Foam과 foam+compact를 비교. 14 m/s, 3개 각도·3개 spin 설정, high-speed 실측과 companion FE model. | 표면 층에 따른 spin ratio의 상대적 효과가 normal과 tilted incidence에서 달라졌다. 동일한 재료 구성에서도 입사 조건이 관측 관계를 바꾼다. | 측정된 spin 자체의 부호와 v71의 $v_x'-Rω_y'$ 부호는 서로 다른 양이다. v71의 Grass 실패 원인을 FE상의 pimple 변형 하나로 귀속하려면 event별 추가 관측이 필요하다. [원문 §2–4, Fig. 5·6·9](https://www.mdpi.com/2076-3417/9/1/158) |
| Harrison·Sharma·Pavone (2018). 이전 task에서 학습한 prior를 새 task의 소수 관측으로 어떻게 갱신하는가? | ALPaCA에서 task family의 feature와 last-layer prior를 offline 학습하고 새 context로 Bayesian linear regression. 사례에 mass/length가 다른 simulated pendulum 및 friction/torso가 다른 hopper가 있다. | 동역학 실험에서 초반 몇 관측만으로 오차와 predictive likelihood가 개선됐다. | Source response를 보존하면서 소수 관측으로 expert를 정규화해 갱신하는 선례다. 학습 prior와 target task의 관계, 센서와 목적함수가 다르므로 racket에서의 표본수·정확도를 예고하는 수치로 쓰지 않는다. [원문 §3–6 및 Fig. 4–5](https://arxiv.org/abs/1807.08912) |
| Gossard·Tebbe 등 (2023). SpinDOE가 회전을 얼마나 정확히 읽는가? | Dot 검출, geometric hashing, orientation sequence의 regression. Motorized ball spinner의 측정값으로 검증하고 10개 영상, 약 26 ms를 사용한 시험을 보고했다. | 원문은 각속도가 매우 낮거나 높은 영역의 relative error 증가, 검출점 수와 ball logo에 따른 orientation failure를 구별한다. | 별도 계측 검증이다. Racket dataset의 각 impact에 같은 오차를 부여할 근거는 없다. $v_x'$와 $Rω_y'$가 비슷한 행에서 release sign이 민감한 이유와 margin sensitivity의 필요성을 보여준다. [원문 §IV.B–C](https://arxiv.org/abs/2303.03879) |

### 2. 문헌을 주제별로 읽은 결과

**표면을 바꾸면 달라지는 항.** Gossard는 normal rebound와 tangential impulse의 상태 의존성을 분리한다. Rinaldi 등은 표면 층을 바꿨을 때 같은 변형의 효과가 입사각에 따라 달라질 수 있음을 보였다. Jia 등은 무회전에서 간단한 restitution fit이 설명할 수 있는 범위를 보였다. 따라서 부호 하나가 잘 전이되는지와, 그 부호를 맞혔을 때 실수값 velocity·spin까지 맞는지는 따로 평가해야 한다. [Gossard](https://arxiv.org/html/2604.11349v1) · [Rinaldi 등](https://www.mdpi.com/2076-3417/9/1/158) · [Jia 등](https://doi.org/10.1177/17543371241310338)

**적은 관측으로 조정할 항.** Gossard는 같은 rubber type의 prior를 현재 상호작용으로 갱신했다. ALPaCA는 많은 source tasks에서 학습한 feature와 weight prior가 작은 target context를 활용하게 한다. v71에서는 같은 calibration impacts에서 routing만 변경했을 때와 새 표면의 연속 response까지 갱신했을 때의 차이가 핵심 대조다. [Gossard §VI](https://arxiv.org/html/2604.11349v1) · [ALPaCA](https://arxiv.org/abs/1807.08912)

### 3. Knowledge map

Rubber·sponge·blade 및 입사 상태 → 접촉 동안의 변형·impulse → 이탈 translation과 rotation → 영상으로 추정한 $s'=v_x'-Rω_y'$ → calibration의 disagreement → gate 방향 결정 → expert의 연속 예측 → 실제 held-out velocity·spin 오차. ALPaCA의 prior는 마지막 예측과 갱신 사이에 대응한다. SpinDOE는 rotation 추정 단계에 오차를 도입할 수 있다. **직접 비교 가능한 연결**은 공개된 입·출력으로부터 계산한 disagreement, routing, endpoint loss다. **남는 연결**은 pimple 변형, 압력 분포, 국소 stick/slip이 해당 부호를 발생시킨 경로다. [Gossard 데이터](https://github.com/gossardt/tt_rackets) · [SpinDOE](https://arxiv.org/abs/2303.03879) · [Rinaldi 등](https://www.mdpi.com/2076-3417/9/1/158)

### 4. 근거에서 드러난 공백

- **추가 표면의 독립 반복.** 공개 데이터에서 Grass D.TecS는 같은 제품을 sponge 두께 1.2 mm와 0 mm로 바꾼 구성 두 개다. 이 둘에서 방향이 같아도 독립 제조사 long-pips 두 제품으로 계산할 수 없다. v70에서 이미 발견한 사례라는 점도 결과 해석에 남는다. [구성 표](https://github.com/gossardt/tt_rackets/tree/159c6f040dd7832570b537b3b5fd00a97fc28c15)
- **감지 대상과 최적 행동의 불일치.** $s'$ 부호를 틀렸는지에는 1표씩 주지만, 틀린 방향의 expert가 만드는 m/s·rad/s 오차는 shot마다 다르다. Gossard의 목표 지표는 실제 velocity·spin이다. 부호 정확도를 최종 목표로 두면 작은 오차를 여러 번 낸 경우와 큰 오차를 한 번 낸 경우를 구별하지 못한다. [Gossard §V.E](https://arxiv.org/html/2604.11349v1)
- **교정량의 식별.** 소수 calibration에서 비슷한 입사 조건만 관측하면 expert slope와 offset을 동시에 안정적으로 정하기 어렵다. Source를 따르는 regularization과 새로운 입사각·spin을 포함하는 calibration 설계를 분리해 볼 이유다. [ALPaCA](https://arxiv.org/abs/1807.08912) · [Rinaldi 등](https://www.mdpi.com/2076-3417/9/1/158)
- **Measurement uncertainty.** SpinDOE의 별도 성능은 ground truth가 알려진 장치에서 나온다. 각 impact의 joint velocity–spin uncertainty와 annotation repeatability는 공개 14열에서 확인할 수 없다. [SpinDOE](https://arxiv.org/abs/2303.03879) · [Gossard 자료](https://github.com/gossardt/tt_rackets)

### 5. 상충하는 듯한 연구의 비교

Gossard의 10–30회와 v71의 몇 회는 같은 sample-complexity 추정량이 아니다. 원 논문은 inverted 계열 prior에서 연속적인 restitution과 response를 새 관측으로 수정하고, velocity 오차의 평균 변화를 그렸다. 이미 학습된 gate의 이진 방향 교환 여부는 훨씬 제한된 가설이다. 이진 방향을 적은 샘플로 알아도 연속 예측이 충분히 정확해졌다는 결론은 별도 검증을 요구한다. Jia 등의 무회전에서 강한 선형 상관과 Rinaldi 등의 oblique 조건에서 관찰한 spin 효과도 서로 다른 조건·지표를 설명한다. [Gossard §VI](https://arxiv.org/html/2604.11349v1) · [Jia 등](https://doi.org/10.1177/17543371241310338) · [Rinaldi 등](https://www.mdpi.com/2076-3417/9/1/158)

### 6. 방법 비교와 독립 평가 감사

| 확인할 항목 | 공정한 해석 |
|---|---|
| Rubber holdout | Source gate와 source expert에서는 동일 제품을 모든 blade·sponge 조합에서 제외한다. Target calibration은 공식 train에서 구입한 prefix만, 최종 평가는 공식 test로 제한한다. [구성 목록](https://github.com/gossardt/tt_rackets/tree/159c6f040dd7832570b537b3b5fd00a97fc28c15) |
| Budget | 동일한 draw와 prefix를 gate-only, gate+expert, inclined rigid, single affine 모두에 준다. Frozen comparator와 달리 각 baseline도 같은 수의 target observations로 parameter를 갱신해야 한다. [ALPaCA](https://arxiv.org/abs/1807.08912) · [Gossard](https://arxiv.org/html/2604.11349v1) |
| 계측 시점 | Calibration shot의 $s'$는 그 shot 종료 후 알려진다. 이후 test shot은 incoming state로 예측한다. 이번 실험의 비용 단위는 이미 기록된 impact 수다. 실제 sensing latency·reset 비용은 별도 검증 대상이다. [Gossard §III](https://arxiv.org/html/2604.11349v1) |
| 반복 단위 | 같은 pool에서 뽑은 무작위 prefix 수는 calibration draw의 변동을 나타낸다. 예측한 test shots와 물리적 표면은 반복마다 같다. 새 표면의 유효 표본 수를 draw 횟수로 늘릴 수 없다. [Gossard 공개 자료](https://github.com/gossardt/tt_rackets) |
| 역사적 재사용 | v70에서 같은 test set으로 Grass를 실패 사례로 발견했다. v71 test labels를 개별 fit에서 숨기는 것과, 미지의 대상에서 처음 가설을 확인하는 것은 다른 검증 단계다. 결과는 알려진 개발 사례를 이용한 protocol replay이며, 최종 confirmatory test는 새 acquisition이 필요하다. [데이터 release](https://github.com/gossardt/tt_rackets/tree/159c6f040dd7832570b537b3b5fd00a97fc28c15) |

또 하나의 구별은 **source prediction에 대한 global disagreement**와 **같은 shot에서 incoming sign이 outgoing sign으로 바뀌는 물리적 reversal**이다. v71의 detector가 직접 검정하는 것은 전자다. Global swap이 평균적으로 맞는 제품에서도 특정 입사 조건의 모든 shot이 뒤집힌다는 뜻은 아니다. 한 제품 안에서 pooled calibration 자료의 빈도가 달라지면 관찰되는 전환율도 바뀔 수 있다. [Gossard §II–V](https://arxiv.org/html/2604.11349v1)

### 7. Evidence synthesis

**강하게 뒷받침되는 부분**은 장비가 측정한 rebound response가 rubber와 입사 조건에 따라 달라지고, prior에 새 표면의 관측을 넣어 오차를 줄일 여지가 있다는 점이다. **제한적 근거**는 gate disagreement의 공간적 범위와 지속성이다. 이 실험은 동일한 여섯 제품과 기록된 조건에서 비교할 수 있다. **남는 불확실성**은 새 제조사의 long pips, 완전히 다른 spin 구간, 다른 공·날짜·습도에서의 검출과 계수 전이다. 공개 자료에서 그 방향을 물리적으로 일으킨 구조나 실제 경기 성능은 확인되지 않는다. [Gossard](https://arxiv.org/html/2604.11349v1) · [Rinaldi 등](https://www.mdpi.com/2076-3417/9/1/158) · [Jia 등](https://doi.org/10.1177/17543371241310338)

### 8. 자료와 공백에 근거한 후속 질문 10개

| 질문 | 필요한 변수·이유 | 검증 방식과 근거 |
|---|---|---|
| 1. 다른 제조사의 long pips에도 같은 global flip이 필요한가? | 제품·batch, disagreement 빈도, velocity·spin 오차 | 미관측 제품과 acquisition session에 v71 rule을 고정하고 평가. [Gossard](https://arxiv.org/html/2604.11349v1) |
| 2. 같은 rubber의 OX에서 배운 방향이 sponge 장착 시에도 통하는가? | sponge 두께, 입사 상태, 방향·magnitude | 한 구성 calibration으로 다른 구성 test를 평가한 뒤 순서를 바꾼다. [Gossard 구성 표](https://github.com/gossardt/tt_rackets) |
| 3. 부호보다 예상 velocity·spin 손해로 투표하면 나아지는가? | 같은 disagreement에도 다른 endpoint 손실 | Source-only로 손실 차이를 예측하고 동일 prefix의 단순 투표와 대조한다. [Gossard §V.E](https://arxiv.org/html/2604.11349v1) |
| 4. $|s'|$가 작은 calibration label을 보류할 이득이 있는가? | spin 측정 오차·margin·구입한 총 impact 수 | 사전에 margin 규칙을 정하고 동일 sensing budget의 무작위 보류와 비교한다. [SpinDOE §IV](https://arxiv.org/abs/2303.03879) |
| 5. 실제 표면별로 global flip 하나면 충분한가? | 입사각·normal speed별 disagreement 빈도 | 입사 조건을 층화해 고정 gate swap과 source에서 지정한 조건별 swap을 비교한다. [Rinaldi 등](https://www.mdpi.com/2076-3417/9/1/158) |
| 6. 두 expert에 공유하는 보정만으로 충분한가? | 공유 offset, slope, branch별 residual | 같은 budget에서 shared-only, branch-specific, shrinkage update를 nested holdout한다. [ALPaCA](https://arxiv.org/abs/1807.08912) |
| 7. 새 표면에서 입사 범위를 넓게 고르면 필요한 횟수가 줄어드는가? | calibration state coverage·held error | Random sampling과 outcome-free space-filling sampling을 같은 pool에서 paired 비교한다. [Jia 등](https://doi.org/10.1177/17543371241310338) · [Gossard](https://arxiv.org/html/2604.11349v1) |
| 8. 같은 예산에서 physical parameter prior와 affine prior 중 어느 쪽이 유리한가? | contact type, restitution, tangential gain, shrinkage strength | 같은 source와 target calibration으로 update하고 제품별 두 metric을 비교한다. [Gossard §VI](https://arxiv.org/html/2604.11349v1) · [ALPaCA](https://arxiv.org/abs/1807.08912) |
| 9. 부호와 오차의 순위가 장비·spin 추정 절차를 바꿔도 유지되는가? | camera·marker, repeated annotation, confidence | 같은 impact에서 독립 spin 추정을 생성해 detector·loss의 불일치를 측정한다. [SpinDOE](https://arxiv.org/abs/2303.03879) |
| 10. 감지 증거가 충분해지면 일찍 멈춰도 되는가? | stopping time, false flip, session loss | Sequential rule을 source에서 고정하고 동등 평균 비용의 fixed budget과 새 acquisition에서 비교한다. [Gossard §VI](https://arxiv.org/html/2604.11349v1) |

### 9. 논문 기여에 대한 검토

Gossard가 이미 few-observation online adaptation을 제안했고, ALPaCA는 source prior와 새 관측으로 연속 출력을 갱신하는 일반 원리를 제시했다. v71의 기여 후보는 **같은 calibration evidence가 잘못된 routing을 얼마나 고치며, 고친 뒤에도 연속 response에 얼마의 오차가 남는지 실제 rubber 단위로 분해하는 것**이다. Downstream 성능에서는 단순 single affine 및 physical model이 동일 budget으로 적응했을 때의 결과를 함께 제시해야 한다. 불리한 제품이나 budget도 그대로 내야 새로운 방법의 적용 조건을 평가할 수 있다. [Gossard](https://arxiv.org/html/2604.11349v1) · [ALPaCA](https://arxiv.org/abs/1807.08912)

독립 검산에서는 v70의 source-only 예측식을 재구성하고, 공개 calibration pool을 전수 집계해 무복원 표본의 gate flip 확률과 gate-only endpoint loss의 기대값을 계산했다. 이 집계는 평가자만 접근한다. 실제 detector는 받은 prefix만 사용해야 한다. [독립 계산 원리와 결과](review/REFERENCE_KO.md)
