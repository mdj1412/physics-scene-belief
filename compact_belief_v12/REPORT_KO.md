# Compact multi-model belief v12: 독립적인 belief 압축과 model mismatch 검증

작성일: 2026-11-12 KST. 실제 실행: `datacenter3`를 통해 `node56`, 기존 allocation `2680491`의 idle GPU 1개. GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition. 설명은 한글, technical terms는 English로 작성했다.

## 1. 이번 실험의 결론

**Full joint posterior를 계산하거나 그 moments를 전달받지 않는 compact multi-model belief를 구현하고, 서버에서 독립적인 test worlds로 검증했다.** 4개의 model probabilities와 각 model의 3D Gaussian mean/covariance만 유지한다. Persistent numerical state는 world당 **416 bytes**다.

선택된 compact 방법은 matched, unseen-boundary, unseen-contact-law 조건 모두에서 QMC16384 및 full reference 대비 사전에 정한 **mean decision loss ratio의 upper CI ≤ 1.10** 기준을 통과했다. 그러나 **warm GPU latency는 29.750 ms로 QMC16384의 2.490 ms보다 약 11.95배 느렸다.** Accuracy와 2배 speedup을 함께 요구한 사전 screening은 모두 실패했다. 현재 결과로 주장할 수 있는 것은 작은 state로 decision accuracy를 보존한 사례이며, 강한 baseline 대비 compute 우위는 아니다.

더 중요한 실패는 contact-law mismatch였다. Boundary family만 바꾼 조건에서는 loss 증가가 확인되지 않았지만, contact law가 실제 바뀐 135 worlds에서는 compact의 mean loss가 matched 대비 **0.032700** 증가했다. Full reference와 QMC도 거의 같은 수준으로 악화되었다. Model menu 자체가 틀린 상황에서는 integration을 정밀하게 하거나 belief를 압축하는 것만으로 해결되지 않았다.

**현재 형태만으로 독립적인 method paper의 기여는 부족하다.** 다음 연구의 우선순위는 compact representation 자체의 발명보다, 작은 belief에서 model mismatch를 감지하고 필요한 observation 또는 model expansion을 결정하는 방법이다. 이는 이번 결과에서 도출한 다음 가설이며, 아직 검증된 성과는 아니다.

## 2. 무엇을 독립적으로 구현했는가

### 2.1 Compact state와 update

Discrete model은 boundary family 2개와 contact model 2개의 조합인 4개다. 각 component는 다음 state를 유지한다.

\[
q(m,u\mid D)=\pi_m\,\mathcal N(u;\mu_m,\Sigma_m),\qquad m\in\{1,2,3,4\},\ u\in\mathbb R^3.
\]

Latent coordinates는 Gaussian CDF를 이용해 physical parameters로 바꾼다.

\[
s=0.8\Phi(u_1)-0.4,\quad a=0.2\Phi(u_2)-0.1,\quad e=0.35+0.55\Phi(u_3).
\]

Initial standard-normal distribution을 연속적으로 변환하면 기존 bounded uniform prior와 일치한다. 실제 계산에서는 유한한 Gauss–Hermite quadrature를 사용하므로 integration approximation은 남는다.

Update는 Gauss–Hermite sigma points에서 state–measurement moments를 계산하는 Gaussian-sum assumed-density filter다. Factual boundary profile와 factual collision velocity를 하나의 11D measurement로 처리한다. 각 component의 mean/covariance는 Gaussian conditional update로, model probabilities는 approximate Gaussian predictive likelihood로 갱신한다. Likelihood tempering을 사용할 때 whitened observation covariance를 step 수만큼 늘리고 같은 measurement를 반복 반영한다. 단순히 원래 likelihood를 네 번 적용하는 방식은 아니다. Covariance는 symmetrization과 eigenvalue floor `1e-8`을 적용한다.

Dev 64 worlds에서 GH order `{3,5}` × tempering steps `{1,4}`를 비교해 **GH5, 4 steps**를 test 전에 선택했다. Component당 125개, 총 500개의 sigma points를 moment calculation마다 사용한다. State의 numerical payload는 `4 × (1 + 3 + 9) × 8 = 416 bytes`이며, shared sigma-point cache와 temporary arrays, Python/Tensor object overhead는 별도다.

이는 **approximate joint dependencies를 유지하는 compact belief**다. Geometry와 restitution의 correlation까지 제거한 factorized point estimate는 아니다. Full posterior의 array를 입력받거나 full posterior에서 압축한 representation이 아니라는 의미에서 독립적이다. Component 내부의 복잡한 multimodality를 정확히 보존한다는 주장은 하지 않는다.

### 2.2 Observation selection과 prediction

Target query velocity를 `T`, action `a`에서 얻을 measurement를 `Y_a`라고 하면 selection score는 다음 LMMSE gain이다.

\[
G(a)=\frac12\operatorname{tr}\left[C_{TY_a}(C_{Y_aY_a}+R_a)^{-1}C_{Y_aT}\right].
\]

Compact와 QMC/full reference는 **같은 score**를 사용한다. 이 식은 linear predictor의 expected MSE reduction에 해당하는 surrogate이며, nonlinear posterior update 이후의 exact expected decision loss를 직접 적분한 nested acquisition은 아니다. 이후 실제로 선택한 observation을 받아 belief를 갱신하고 posterior predictive mean을 출력한다.

이번 강한 QMC baseline은 prior Sobol bank에 exact likelihood reweighting을 적용한다. 이전 단계의 nested SNIS acquisition을 그대로 재사용하지 않았다. 따라서 cross-fitting으로 nested utility bias를 제거하는 실험이 아니라, 공통 LMMSE acquisition 아래에서 inference representation과 update를 비교하는 실험이다.

### 2.3 Full posterior 의존성을 어떻게 차단했는가

`compact.py`, `run_compact.py`, `physics.py`, `common.py`는 reference module을 import하지 않는다. 공유하는 것은 forward equations, prior, measurement noise와 public factual evidence다. Compact phase에서 dev 선택과 모든 test policies의 freeze 및 prediction 생성을 마친 뒤, 별도 phase에서 처음 `bank.py`를 import했다.

`compact_policy_freeze_done.json` → `compact_stage_done.json` → `reference_import.json`의 timestamps와 `no_bank_import: true`를 저장했다. Policy/public data의 SHA256을 보존했고, private test targets와 hypothetical observations는 policy freeze 이후 평가에서만 읽었다. 모든 action의 prediction을 평가용으로 저장했지만 selector에는 unselected observation이 전달되지 않는다. 이 절차는 기록과 code audit에 근거한 검증이며, cryptographic execution attestation을 의미하지 않는다.

## 3. Unseen boundary와 contact law를 분리한 실험

### 3.1 기본 physical setup

Synthetic 2D point-mass instantaneous collision을 사용한다. Boundary는 `x(y)=s y+a f_g(y)`이고 `L=25 mm`, `z=y/L`이다.

| Boundary family | 함수 |
|---|---|
| Candidate `g=0` | `f₀=L z²/2` |
| Candidate `g=1` | `f₁=L(z²/2−0.35z⁴)` |
| Withheld `g=2` | `f₂=L(z²/2−0.15z⁴+0.06z⁶)` |

Factual contact는 `y=0`, incoming velocity `(1, 0.35) m/s`다. Factual evidence는 `−10…10 mm`의 9개 boundary samples와 2D outgoing velocity다. Profile noise SD는 `1.2 mm`, velocity noise SD는 `0.07 m/s`다. Query는 모든 worlds에서 `y=25 mm`, incoming velocity `(1, 0.9) m/s`로 고정한다. Visibility와 candidate observations도 고정한다.

| Action | Measurement | Noise SD |
|---|---|---|
| `none` | 추가 observation 없음 | — |
| `query_view` | `19…31 mm`, boundary samples 7개 | `1 mm` |
| `factual_view` | `−8…8 mm`, boundary samples 7개 | `0.35 mm` |
| `probe` | `y=0`, incoming velocity `(1,−0.7)`의 outgoing velocity | `0.05 m/s` |

추가 action 사이의 acquisition cost는 동일하게 취급한다. 실제 camera 이동이나 physical reset 시간은 측정하지 않았다. `none`은 추가 sensing을 하지 않는 control이다. Image encoder나 raw RGB를 이번 단계에서 새로 검증한 것은 아니며, boundary-preserving profile samples를 직접 measurement로 제공했다.

### 3.2 세 paired conditions

새로운 independent base worlds **256개** 각각에 다음 세 조건을 만들었다. 같은 `s,a,e,c`, 같은 factual noise와 추가 observation noise를 공유한다. 각 action의 actual observation noise는 16회 반복한다. 따라서 condition마다 independent world 수는 256개이며, 16 repeats를 독립 worlds로 세지 않는다.

| Condition | 변경한 것 | 고정한 것 |
|---|---|---|
| Matched | 없음 | Candidate family에서 생성 |
| Boundary-only | Boundary family를 withheld `g=2`로 교체 | Contact law와 나머지 parameters/noise |
| Contact-only | `c=1`의 tangential contact law 교체 | Boundary와 나머지 parameters/noise |

Candidate frictional law는 Coulomb form이다.

\[
j_t=\operatorname{sign}(v_t)\min\{|v_t|,\;0.22(1+e)v_n\}.
\]

Contact-only shift에서는 이를 다음 synthetic viscous-like tangential impulse로 교체한다.

\[
j_t=\lambda v_t,\qquad\lambda\sim U[0.25,0.65].
\]

Normal restitution은 유지한다. 이는 Coulomb coefficient만 바꾸는 shift가 아니라 tangential response의 functional form을 바꾸는 shift다. `c=0`은 frictionless로 유지하므로 121 worlds가 정확히 변화 없는 negative control이고, 실제 law가 바뀐 `c=1` subset은 **135 worlds**다. Inference model menu에는 withheld boundary나 viscous-like law를 추가하지 않았다.

Pilot 16, dev 64, test 256은 서로 다른 seeds를 사용했다. Base seed는 각각 `1200101`, `1300101`, `1400101`이며 world마다 `137 × index`를 더한다. Boundary-only shift는 특정 smooth family 하나에 대한 검사다. Severe shape extrapolation이나 임의의 unseen geometry에 대한 robustness test로 확대 해석할 수 없다.

## 4. Strong baselines와 primary decision loss

Strong baselines는 stratified scrambled Sobol QMC banks 4,096/16,384/65,536개와 order61 full quadrature다. Full reference는 `4 × 61³ = 907,924`개의 nodes에서 같은 candidate model posterior를 적분한다. New observation 이후에도 weights를 갱신한다. OOD 조건에서 이 reference는 **candidate-model posterior reference**이며, 실제 shifted law를 알고 있는 oracle은 아니다.

추가로 v11에서 고정한 depth3 rule에 QMC16384 또는 reference decoder를 연결했다. 이 rule은 과거 offline reference-derived risk labels로 학습한 baseline이다. Compact에는 해당 teacher labels나 rule을 제공하지 않았다. 따라서 rule의 offline training cost는 별도이며, online cost만 보고했다.

Loss는 `||predicted query velocity − true query velocity||² / 2`다. 여기서 downstream decision은 continuous velocity prediction이며, robot control success나 physical task reward는 아니다. 아래 숫자는 world별 16 repeats를 평균한 뒤 256 worlds에서 평균한 MSE다. 단위는 `(m/s)²`이며 작을수록 좋다.

| Method | Matched | Boundary-only | Contact-only, 전체 |
|---|---:|---:|---:|
| **Compact GH5 / 4 steps** | **0.005859** | **0.005763** | **0.023104** |
| QMC4096 | 0.005857 | 0.005802 | 0.023013 |
| QMC16384 | 0.005898 | 0.005895 | 0.023249 |
| QMC65536 | 0.005906 | 0.005894 | 0.023269 |
| Full reference61 | 0.005906 | 0.005894 | 0.023270 |
| Frozen rule + QMC16384 | 0.005669 | 0.005610 | 0.023275 |
| Frozen rule + reference61 | 0.005672 | 0.005608 | 0.023292 |

Compact의 평균이 reference보다 조금 낮다는 사실을 superiority로 해석하면 안 된다. Six primary comparisons에서 모두 paired difference CI가 0을 포함했다. Compact와 QMC16384/full reference의 mean loss ratio에 30,000 paired world-bootstrap resamples와 Bonferroni-adjusted **99.1667% CI**를 적용했다.

| Condition | Baseline | Compact / baseline | Adjusted CI |
|---|---|---:|---|
| Matched | Full reference | 0.9922 | [0.9327, 1.0576] |
| Matched | QMC16384 | 0.9935 | [0.9355, 1.0567] |
| Boundary-only | Full reference | 0.9778 | [0.9216, 1.0397] |
| Boundary-only | QMC16384 | 0.9776 | [0.9232, 1.0361] |
| Contact-only | Full reference | 0.9929 | [0.9588, 1.0277] |
| Contact-only | QMC16384 | 0.9938 | [0.9601, 1.0285] |

모든 upper bound가 1.10 미만이므로 이번 diagnostic에서 정한 **one-sided noninferiority screening**은 통과했다. 10% margin은 실험 전에 정한 engineering threshold이며, application에서 허용되는 오차를 입증한 값은 아니다. Statistical equivalence나 universal accuracy preservation을 주장하지 않는다.

![Mean decision loss와 warm GPU latency](results/analysis/loss_latency_report.png)

그림은 means만 표시하며 각 panel의 y-axis scale이 다르다. 좁은 y-axis에서 보이는 작은 순위 차이의 통계적 판단은 위 CI와 함께 해야 한다. 원래 4개의 compact configurations와 diagonal ablation 전체 frontier는 `results/analysis/summary.json` 및 `loss_latency.png`에 보존했다.

## 5. Memory와 compute 비용

같은 GPU, float64, batch1에서 factual fit → observation selection → 선택한 observation 하나의 update/prediction을 측정했다. Fixed 16 cases × 3 timing repeats, warmup 이후 각 stage 전후 CUDA synchronization을 수행했다. 아래는 total latency의 median이며 setup은 포함하지 않는다.

| Method | GPU latency, ms | CPU latency, ms | Persistent state, bytes/world | Shared numerical cache, bytes |
|---|---:|---:|---:|---:|
| **Compact GH5 / 4 steps** | **29.750** | **5.101** | **416** | **4,000** |
| Compact GH3 / 1 step | 12.300 | 1.896 | 416 | 864 |
| QMC4096 | 2.105 | 0.560 | 32,768 | 7,110,656 |
| QMC16384 | 2.490 | 0.907 | 131,072 | 28,442,624 |
| QMC65536 | 3.548 | 3.214 | 524,288 | 113,770,496 |
| Full reference61 | 21.781 | 43.153 | 7,263,392 | 1,576,156,064 |
| Frozen rule + QMC16384 | 1.656 | 측정하지 않음 | 131,072 | 28,442,624 |

State 숫자는 numerical payload이며 전체 process memory가 아니다. Compact는 QMC16384보다 약 **315배**, reference보다 약 **17,460배** 작은 per-world state를 유지한다. Bank의 shared cache는 여러 worlds에서 재사용할 수 있으므로 world마다 전부 복제되는 비용으로 계산하면 안 된다. Rule artifact/Python object, common model constants 등은 이 cache 숫자와 별도다.

Selected compact의 GPU median stage latency는 factual fit **12.881 ms**, selection **5.681 ms**, update/prediction **11.321 ms**였다. QMC16384는 각각 **0.302 / 1.737 / 0.416 ms**였다. 각 stage median의 합과 total median은 일반적으로 일치하지 않는다.

Compact는 매 update마다 작은 matrix decompositions와 nonlinear moment evaluations를 여러 번 실행한다. Bank methods는 고정된 sensors/query에 대한 forward tables와 moment features를 cache하여 weight reweighting과 matrix products를 사용한다. 현재 implementation과 batch1에서는 작은 state가 낮은 latency로 이어지지 않았다. GPU dispatch overhead가 영향을 준다는 해석은 CPU 결과와 일관되지만, kernel profiler로 병목 비율을 분해한 것은 아니므로 확정 원인으로 제시하지 않는다.

GPU에서 측정한 setup은 compact **0.978 ms**, QMC16384 **6.266 ms**, reference **30.436 ms**였다. 이는 warm process 안의 method construction 비용이며, process startup/CUDA context initialization을 포함한 cold-start가 아니다. GPU warm latency는 cache가 존재하는 반복 사용 조건에 해당한다. Sensors/query가 계속 바뀔 때 cache 재생성 비용까지 포함한 break-even point는 아직 측정하지 않았다.

CPU timing은 GPU 결과를 본 뒤 추가한 **post-primary supplementary benchmark**다. 같은 allocated node의 CPU 2 threads, 같은 cases/repeats, 같은 code와 float64를 사용했다. Compact는 CPU에서 reference보다 약 **8.46배 빠르지만 QMC16384보다 약 5.62배 느렸다.** CPU와 GPU 사이 최대 prediction 차이는 전체 측정 methods에서 `1.53e-13`이었다. CPU의 유리한 비교만 골라 사전 GPU screening의 성공으로 바꾸지 않았다.

정리하면 accuracy margin은 통과했지만, latency ≤ baseline/2 조건까지 함께 요구한 **accuracy–cost screening은 six primary comparisons 모두 실패**했다. Faster compact configuration을 test 결과에 맞춰 다시 선택하지 않았으며, GH3/1 step은 dev에서 이미 accuracy가 나빴다. Method별 controlled batched throughput은 이번 결과에서 주장하지 않는다. 전체 evaluation 실행시간에는 서로 다른 batch size와 추가 diagnostics가 섞여 있어 serial latency와 별도로 보아야 한다.

## 6. Boundary shift와 contact-law shift에서 무엇이 달랐는가

| Method | Boundary-only − matched, 전체256 | Contact-only − matched, 실제 변경135 | Contact-only loss, 실제 변경135 |
|---|---:|---:|---:|
| Compact | −0.000096 | +0.032700 | 0.039576 |
| QMC16384 | −0.000003 | +0.032903 | 0.039641 |
| Full reference | −0.000011 | +0.032928 | 0.039672 |

Compact의 boundary paired effect에 대한 descriptive 95% CI는 **[−0.000727, +0.000586]**이다. 현재 sample과 shift 크기에서는 악화가 확인되지 않았다. 변경된 contact subset에서는 **[+0.026919, +0.038666]**으로 뚜렷하게 증가했다. 전체256에서 compact contact-only loss는 matched의 약 3.94배다. Frictionless negative-control 121 worlds의 paired loss difference는 모든 분석 methods에서 정확히 0이었다.

![Boundary와 contact-law shift의 분리](results/analysis/separated_shifts.png)

Boundary-only 결과가 양호한 것은 이번 withheld polynomial이 candidate families와 충분히 비슷하고, query 근처의 boundary observation이 local slope 정보를 제공하기 때문일 수 있다. 원인에 대한 이 설명은 가설이다. 모든 unseen shape를 다룰 수 있다는 증거는 아니다.

Contact law를 바꿨을 때 full reference까지 실패한 사실은, 현재 finite model menu 안에서 posterior를 더 정확히 계산하는 것만으로는 해당 error를 없애기 어렵다는 근거다. 다만 이것이 모든 가능한 acquisition이나 richer model의 실패를 증명하는 것은 아니다.

### 추가 observation과 adaptive selection의 이득을 분리

아래는 selected compact의 같은 decoder를 사용한 결과다. Fixed-action controls와 관련 CI는 primary six contrasts 이후의 **descriptive diagnostics**다.

| Policy | Matched | Boundary-only | Contact-only |
|---|---:|---:|---:|
| No additional observation | 0.014108 | 0.014008 | 0.025493 |
| Always query_view | 0.010078 | 0.009382 | 0.023797 |
| Always factual_view | 0.012032 | 0.011828 | 0.024892 |
| Always probe | 0.006422 | 0.007013 | **0.021160** |
| Adaptive compact | **0.005859** | **0.005763** | 0.023104 |

Adaptive compact는 `none` 대비 matched에서 약 **58.5%**, boundary-only에서 약 **58.9%** 낮은 loss를 보였다. Paired difference의 descriptive 95% CI도 각각 `[−0.010325,−0.006369]`, `[−0.010288,−0.006355]`다. 추가 observation의 유용성은 유지된다.

그러나 adaptive selection 자체의 가치는 더 제한적이다. Always-probe 대비 matched difference는 `−0.000563`, CI `[−0.001412,+0.000450]`으로 우위가 명확하지 않았다. Boundary-only에서는 `−0.001250`, CI `[−0.002157,−0.000191]`이었다. Contact-only에서는 오히려 **+0.001944**, CI `[+0.000051,+0.003921]`로 adaptive selection이 나빴다. 이는 multiple-comparison adjustment를 하지 않은 secondary finding이며 새 test로 재검증해야 한다. 그럼에도 유리한 `none` comparison만으로 observation selection의 보편적 성공을 주장할 수 없음을 보여준다.

Compact는 matched/boundary에서 각각 query_view 173회, probe 83회, contact-only에서 query_view 164회, probe 92회를 선택했다. Reference와의 action agreement는 **98.05%, 98.05%, 97.66%**다. Compact choices에 reference decoder를 연결한 loss는 `0.005908 / 0.005914 / 0.023252`였다. Reference choices에 compact decoder를 연결하면 `0.005854 / 0.005752 / 0.023129`였다. Compact가 새로운 acquisition 전략으로 크게 이기는 상황보다, reference의 decision behavior를 작은 state로 대체하는 상황에 가깝다.

강한 frozen rule + QMC16384 대비 compact의 mean differences는 matched `+0.000190`, boundary `+0.000153`, contact `−0.000171`이고 descriptive 95% CI는 모두 0을 포함했다. Rule 대비 superiority는 확보하지 못했다. 또한 within-component covariance를 diagonal로 제한한 ablation의 loss는 `0.005865 / 0.005799 / 0.022951`이었다. Full covariance의 필요성도 이번 결과만으로 주장할 수 없다. Single-model hard decoder는 `0.006611 / 0.006327 / 0.025044`, model-averaged point decoder는 `0.006053 / 0.005785 / 0.023553`였다. 이 값들은 별도의 primary hypothesis test를 하지 않은 descriptive ablations다.

## 7. Numerical validation과 해석 범위

Linear-Gaussian measurement로 바꾼 closed-form validation에서 1-step과 4-step update의 mean/covariance/model probability errors는 모두 `7.5e-15` 이하였다. Pilot에서 covariance PSD, model probability normalization, finite predictions, dissipative energy behavior를 확인했다. 저장한 모든 compact factual states도 PSD와 normalization을 검사했다. Pilot energy check를 모든 possible worlds에 대한 formal physical proof로 해석하지 않는다.

Reference61의 numerical accuracy는 사전에 고정한 first16 base worlds × 3 conditions, 48 cases에서 order81로 확인했다. All-action/all-noise prediction의 maximum difference는 **0.000474 m/s**, mean absolute difference는 **0.000001595 m/s**였다. Maximum single outcome loss change는 **0.000256**이므로 작은 error라고 일괄 생략하지 않았다.

Main selected actions에 한정하면 order61→81의 world-averaged loss change 최대값은 **0.0000006143**이었다. Condition별 mean changes는 matched `−3.97e-9`, boundary `+8.09e-9`, contact `+4.81e-8`이었다. 이 audit subset에서는 primary differences보다 작지만, 나머지240 worlds의 uniform convergence를 보증하지 않는다. Primary reference 및 선택을 audit 결과로 바꾸지 않았다.

그 밖의 한계는 다음과 같다.

- Single-contact synthetic dynamics이며 rotation, deformation, long-horizon rollout, real video noise를 포함하지 않는다.
- Finite model menu는 두 boundary families와 두 contact laws다. Contact-only에서 맞는 law는 menu 밖에 있다.
- Boundary-only와 contact-only를 분리했으며, 둘을 동시에 바꾸는 compound shift는 이번 test 범위가 아니다.
- QMC는 하나의 fixed scramble seed를 사용했다. Multiple QMC seeds와 multiple independent test cohorts에 대한 robustness는 아직 없다.
- Gaussian component 하나가 within-model multimodality를 충분히 표현하는지 별도의 posterior calibration metric으로 평가하지 않았다.
- LMMSE acquisition의 ranking bias와 inference approximation error를 완전히 분리하지 못했다. Exact nested acquisition을 이번 강한 baseline으로 실행한 것은 아니다.
- GPU 하나와 CPU 한 node, 고정된 method 실행 순서의 warm timing이다. Thermal variation, 다른 hardware, compilation, vectorized production implementation에 대한 일반적 속도 결론은 아니다.
- 실험 전 protocol의 10% loss margin/2× speedup 기준을 유지했다. CPU timing과 fixed-action 추가 통계는 결과를 본 뒤의 supplementary diagnostics로 구분했다.

## 8. 논문 기여 가능성에 대한 판단

### 기존 연구와의 경계

Gaussian assumed-density filtering, Gaussian-sum mixtures, Gauss–Hermite integration은 기존 방법이다. Ito & Xiong의 *Gaussian Filters for Nonlinear Filtering Problems*는 이러한 Gaussian/mixed-Gaussian filtering과 numerical integration을 다룬다. 따라서 4 Gaussian components를 만들고 moments를 update한 것 자체를 새로운 filtering algorithm으로 주장하지 않는다. [원문, IEEE TAC 2000](https://webdiis.unizar.es/~rmcantin/glrobot/sesion2/2000_Paper_Ito_TAC.pdf)

Downstream loss를 기준으로 observation을 설계하고 explicit posterior 계산에 의존하지 않는 방향도 기존 연구와 겹친다. *Action-BED: Task-Driven Bayesian Experimental Design with Singly Intractable Objectives*는 experiment policy와 downstream action policy를 joint optimization하는 접근을 제시한다. 현재 compact filter는 그 학습형 policy를 구현하거나 비교한 결과가 아니므로 해당 연구보다 좋다고 말할 수 없다. 2026-11-06의 arXiv v2를 확인했으며 conference acceptance는 주장하지 않는다. [원문](https://arxiv.org/html/2606.23662v2)

### 지금 확보한 것과 확보하지 못한 것

| 후보 claim | 현재 판단 | 근거 |
|---|---|---|
| Full posterior 없이 compact belief를 구성할 수 있다 | 구현·검증 완료 | Independent phase, source/policy hashes, 416-byte state |
| 작은 state로 reference 수준의 decision accuracy를 유지한다 | 제한된 test에서 지지 | Adjusted ratio upper CI < 1.10 |
| Strong QMC보다 compute-efficient하다 | 현재 실패 | GPU·CPU 모두 QMC16384보다 느림 |
| Adaptive selection이 strong rule/fixed action보다 일관되게 좋다 | 입증되지 않음 | Rule 우위 없음, contact shift에서 always-probe가 유리 |
| Geometry–physics full covariance가 반드시 필요하다 | 입증되지 않음 | Diagonal ablation도 유사한 loss |
| Novel filtering/acquisition 원리를 제안했다 | 아님 | Gaussian-sum/GH/LMMSE의 적용 |
| Model mismatch 대응이 중요한 다음 문제다 | 후속 가설로 유망 | Contact law 변경 시 모든 current methods가 악화 |

**연구를 계속한다면 주제를 “posterior를 더 작게 만들기”에서 “limited belief budget 아래 model mismatch를 감지하고 decision을 보호하기”로 좁히는 것이 낫다.** 예를 들어 compact components의 predictive residual이 일관되게 설명되지 않을 때 discrepancy component를 추가하거나, law를 구별할 수 있는 probe를 선택하도록 하는 것이다. 단순히 test에 나온 viscous law를 menu에 넣고 성공하는 것은 새로운 기여가 아니므로, held-out law families와 severity sweep에서 검증해야 한다.

다음 단계의 검증은 다음처럼 정하는 것이 합리적이다. 아직 실행하지 않은 계획이다.

1. **Model adequacy 검사:** Public factual evidence와 실제로 받은 observation의 standardized residual로 mismatch score를 만든다. Unseen law labels 없이 matched dev에서 threshold를 정하고, contact-family/severity별 false-alarm와 decision risk를 측정한다.
2. **Compact adaptive capacity:** Fixed four-component state, diagonal version, discrepancy-augmented compact state, richer QMC를 비교한다. New component는 held-out law 정답을 직접 삽입하지 않고 residual-driven model 또는 uncertainty inflation으로 구성한다.
3. **Observation과 model expansion의 분리:** 같은 belief에서 always-probe, frozen rule, LMMSE, mismatch-aware selection을 비교하고, 같은 selected observation에서 decoder를 교환한다. Correct law를 포함한 oracle menu는 diagnosis용으로만 둔다.
4. **Compute-aware test:** QMC4096/16384를 유지한 채 total persistent memory와 same-device wall-clock budgets를 맞춘다. Query/view가 변할 때 cache invalidation 비용을 포함하고, optimized compact implementation의 이득이 실제로 생기는지 확인한다.
5. **진행 기준:** 새 independent cohorts에서 strong QMC/rule 대비 paired decision loss 감소가 재현되고, meaningful memory 또는 latency constraint 아래 advantage가 남아야 method contribution으로 전진한다. 다시 rule이 충분하거나 mismatch detector가 false alarm만 늘린다면 compact filter는 baseline으로 남긴다.

현재 결과는 reproducible diagnostic과 강한 compact baseline으로는 가치가 있다. 다만 지금 바로 major-conference method paper의 중심 claim으로 삼기에는 novelty와 strong-baseline advantage가 부족하다는 것이 이번 평가다.

## 9. 실행 위치와 재현 자료

Server project directory:

```text
/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/compact_belief_v12
```

실행 환경은 Python 3.9.18, PyTorch 2.7.1+cu128, NumPy 1.26.4다. Existing environment `../continuous_v2/.venv/bin/python`을 사용했다. 다른 jobs나 allocation은 취소·수정하지 않았다. GPU workers는 foreign GPU process가 나타나면 자신의 worker를 종료하는 guard를 사용했고, 두 phase 모두 exit code 0 및 `foreign: []`를 기록했다. CPU supplementary run도 login node가 아닌 allocated compute node에서 실행했다.

핵심 파일:

| 파일 | 역할 |
|---|---|
| `PROTOCOL.md` | Test 전 design, screening, primary comparisons |
| `physics.py`, `compact.py`, `bank.py` | Forward model, standalone compact inference, strong baselines |
| `run_compact.py` | Data generation, dev selection, compact policy freeze/evaluation |
| `run_reference_phase.py`, `run_baselines.py`, `benchmark.py` | Reference/QMC evaluation과 synchronized GPU timing |
| `CPU_TIMING_ADDENDUM.md`, `benchmark_cpu.py` | Post-primary CPU timing과 provenance |
| `validate_linear.py` | Closed-form linear-Gaussian validation |
| `analyze.py`, `supplement_analysis.py` | Primary statistics, secondary controls와 plots |
| `results/data/`, `results/policies/`, `results/evaluations/` | Public/private data, frozen policies, per-action predictions |
| `results/analysis/summary.json`, `world_losses.npz`, `supplement.json` | Primary/secondary statistics와 world-level losses |
| `results/benchmark.json`, `benchmark_cpu.json` | 모든 timing rows와 summaries |
| `results/integrity.json`, `MANIFEST.json` | Source/policy/output checks와 SHA256 manifest |

재실행은 sealed original directory를 덮어쓰지 말고, 새 directory에 code와 `frozen_rule/`만 복사한 뒤 새 allocation 안에서 수행한다. Idle GPU 확인 후 compact phase를 먼저 실행하고, 완료 이후 reference phase를 실행해야 한다. 현재 allocation ID는 실행 기록이며 재사용을 전제로 한 영구 설정이 아니다.

```bash
# 새 allocated compute node / 새 results directory에서 순서대로 실행
python run_compact.py
python run_reference_phase.py
CUDA_VISIBLE_DEVICES= python validate_linear.py
CUDA_VISIBLE_DEVICES= python analyze.py
# 아래 두 단계는 post-primary supplementary diagnostics
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python benchmark_cpu.py
CUDA_VISIBLE_DEVICES= python supplement_analysis.py
python seal.py
```

이 명령의 `python`은 위와 같은 dependencies를 가진 environment를 의미한다. 실제 실행에는 GPU process guard와 Slurm `srun`을 사용했다. `results/compact_worker*.json`, `baseline_worker*.json`, phase markers, environment metadata가 실행 evidence다.
