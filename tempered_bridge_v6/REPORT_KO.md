# Factual posterior에서 hypothetical observation으로 이동하는 Tempered SMC bridge — v6 실험 보고서

설명은 한국어로, technical terms는 English로 작성했다. 본 보고서는 실제 server 실행 결과를 정리한다. 기존 v5 결과를 알고 설계한 후속 실험이며, 동일한 72 worlds를 재사용한 exploratory numerical validation이다.

## 1. 이번 실험에서 확인한 내용

기존 ESS collapse를 보였던 8개 hypothetical view를 다시 계산했다. 가장 큰 오류 사례인 world66/hypothesis110에서 기존 conditional importance estimator의 predictive mean RMSE는 0.267773 m/s였고, tempered bridge는 0.000964 m/s, 새 nested full SMC는 0.001789 m/s였다. Reference는 v5에서 별도 실행한 full SMC 4096 particles 두 runs의 평균이다. 이는 hidden truth와의 오차가 아니라 **posterior inference 간 불일치**다.

다만 같은 사례에서 **tempering 없는 factual-bank SNIS도 RMSE 0.001538 m/s**였다. 큰 개선의 상당 부분은 proposal을 factual posterior 쪽에서 시작한 효과다. Tempering의 추가 효과는 이 control 및 nested full SMC와 비교해야 하며, 기존 estimator 대비 개선 전체를 새로운 algorithm의 기여로 주장할 수 없다.

- **high**: adaptive-budget bridge EVSI의 test MSE는 0.008775, validation에서 고정한 family×query rule은 0.011213였다. 차이는 -0.002438 [-0.004496, -0.000504]이며 95% paired bootstrap interval 전체가 0 아래였다. Active queries에서 bridge/full의 top-action agreement는 98.8%였다.
- **pixel_only**: adaptive-budget bridge EVSI의 test MSE는 0.008213, validation에서 고정한 family×query rule은 0.010358였다. 차이는 -0.002145 [-0.004322, -0.000100]이며 95% paired bootstrap interval 전체가 0 아래였다. Active queries에서 bridge/full의 top-action agreement는 99.2%였다.

Stage ESS의 회복, inner posterior의 일치, outer acquisition ranking의 안정성, 실제 policy 이득은 서로 다른 검증이다. 아래에서 각각 구분한다.

**질문에 대한 직접적인 답은 조건부로 긍정적이다.** Practically stable queries만 보았을 때 `pixel_only`에서는 strong family×query rule 대비 이득이 남았고, `high`에서는 95% interval이 0을 포함했다. 두 조건 모두 factual-bank SNIS 대비 selection MSE의 우위는 확인하지 못했다. 따라서 이번 결과는 tempered bridge가 full SMC와 비슷한 선택에 도달한다는 근거이며, tempering의 추가 계산이 더 저렴한 posterior approximation보다 실제 prediction을 개선한다는 주장에는 아직 부족하다.

## 2. 문제 설정과 이번에 바뀐 부분

v5의 raw image에서 추출한 geometry baseline과 continuous physics model은 그대로 유지했다. Latent scene parameter는 geometry offset `g`, restitution `e`, friction `μ`, shape extent `d`, corner rounding `b`다. Factual grazing contact의 관측은 geometry와 restitution이 서로 보상하는 correlated posterior를 만들 수 있다. 새로운 view는 그 posterior의 일부만 지지하므로, fixed particles에 likelihood를 한 번 곱하는 방식은 소수 particle에 weight가 몰릴 수 있다.

이번 변경은 hypothetical observation마다 continuous posterior를 다시 계산하는 부분이다. `high`와 `pixel_only`의 boundary representation을 사용하며, low-resolution 및 bbox representation을 새로 학습하거나 바꾸지 않았다. `view`뿐 아니라 `left`, `right`, `center` probe도 같은 nested SMC framework에서 계산했다. 따라서 v5와의 변화에는 view estimator 교정과 probe estimator 교정이 함께 포함된다. 이를 구분하기 위한 mixed-estimator ablation도 계산했다.

`high`는 calibrated geometry-feature noise에 1.5 mm pre-render position/extent jitter가 포함된다. `pixel_only`는 같은 high-resolution rasterization에서 이 jitter를 제거한 조건이다. 둘 다 image-derived low-dimensional calibrated likelihood이며, unrestricted raw-pixel likelihood나 실제 camera model은 아니다.

재사용한 실제 factual/probe measurements는 v5의 synthetic RGB frames에서 color-threshold centroid를 추출해 계산한 값이다. Position noise와 rasterization/quantization을 거친다. 반면 이번 hypothetical dynamic observations와 inference는 σ=0.025 m/s의 continuous Gaussian likelihood를 사용한다. 따라서 measurement generator와 likelihood가 완전히 같은 model이라고 가정하면 안 된다. 같은 factual x-velocity 값이 여러 worlds에 반복되는 것도 이러한 discrete measurement 환경에서 나타난다. Known colors, camera calibration, scene family는 강한 실험 가정이다.

Uniform prior의 bounds는 `g∈[-8,8] mm`, `e∈[0.15,0.85]`, `μ∈[0,0.6]`, `d∈[22,34] mm`, `b∈[0,6] mm`다. Radius 20 mm sphere와 static rounded rectangles의 2D continuous collision detection을 사용하며 Coulomb impulse와 spin을 포함한다. Gravity는 없다. Probe shift는 −22/+22/+9 mm, query shift는 −30/−17/−7/+4/+15/+28 mm다. Query target은 0.8 s의 instantaneous velocity이고 observation은 마지막 50 ms의 finite-frame velocity다. MSE 단위는 `(m/s)²`다.

## 3. 구현한 posterior transport

SMC는 Sequential Monte Carlo, ESS는 Effective Sample Size, MH는 Metropolis–Hastings를 뜻한다. Acquisition 비교에는 Expected Value of Sample Information (EVSI)와 Expected Information Gain (EIG)을 사용하며, 추가 control인 SNIS는 Self-Normalized Importance Sampling이다.

정규화된 weights를 $w_i$라고 하면 $ESS=1/\sum_i w_i^2$다. 모든 weight가 같으면 N, 하나의 particle이 거의 모든 weight를 가지면 약 1이다.

Factual likelihood를 $L_f(\theta)$, hypothetical observation likelihood를 $L_a(\theta;y)$라고 쓰면 다음 두 target sequence를 비교했다.

$$\pi^{bridge}_{\beta}(\theta)\propto p(\theta)L_f(\theta)L_a(\theta;y)^{\beta},\qquad \beta:0\rightarrow1.$$

$$\pi^{full}_{\beta}(\theta)\propto p(\theta)[L_f(\theta)L_a(\theta;y)]^{\beta},\qquad \beta:0\rightarrow1.$$

Bridge는 factual posterior bank에서 시작한다. **Factual likelihood의 exponent는 모든 단계에서 1**이다. Full SMC는 매 hypothetical observation마다 prior에서 새로 시작한다. 최종 target은 같지만 접근 경로와 계산량이 다르다.

Bridge의 incremental weight는 $\exp[(\beta_{t+1}-\beta_t)\log L_a]$다. Full SMC는 이 식에 $\log L_f+\log L_a$를 사용한다. Adaptive bisection으로 다음 temperature를 정해 pre-resampling ESS가 0.75N 이상이 되게 했다. 각 단계에서 systematic resampling 후 covariance random walk, local random walk, global uniform-prior proposal을 번갈아 사용하는 MH rejuvenation을 수행한다. Bounds 밖 proposal은 reject한다.

Hypothetical observations를 batched independent SMC populations로 처리하며, population마다 temperature schedule과 resampling이 다르다. Resampling 이후 uniform weights의 ESS=N을 성공 지표로 삼지 않았다. `one_step_ess`, `min_stage_ess`, acceptance, ancestry, predictive mean 및 contact probability를 별도로 저장했다.

Tempering과 resample–move 자체는 기존 방법이다. 이번 구현을 새로운 SMC algorithm으로 주장하지 않는다. 일반적인 배경은 [Del Moral, Doucet & Jasra, Sequential Monte Carlo samplers (2006)](https://www.stats.ox.ac.uk/~doucet/delmoral_doucet_jasra_sequentialmontecarlosamplersJRSSB.pdf)에 해당한다.

## 4. 비교 조건과 leakage 방지

| 항목 | 설정 |
| --- | --- |
| Worlds | corner/gate/deflection 각각 24, 총 72 |
| Split | 각 family 앞 8 validation, 뒤 16 test: 24/48 |
| Queries | World당 6 counterfactual interventions |
| Factual posterior | v5 continuous SMC 4096 particles, shared bank |
| Main budget | 64 outer observations × 1024 particles × 12 MH moves/stage × 2 inner repeats |
| Methods | Tempered bridge와 prior-start nested full SMC |
| Actions | view/left/right/center; high 및 pixel_only view, dynamic probes는 공유 |
| Observation sensor | 50 ms finite-frame velocity: factual은 vx만, dynamic probes는 vx/vy; σ=0.025 m/s |
| Large-inner audit | case1/25/49/66/68,128 outer ×2048 particles ×18 moves ×2 repeats |
| ESS failure audit | 기존 case66/68의 8개 hypothetical views,4096 particles ×18 moves ×2 repeats |
| Policy evaluation | v5 actual-observation full SMC 1536 particles/12 MH의 score 재사용; no-observation은 factual bank 4096 |
| Uncertainty | 48 test worlds 단위 paired bootstrap 10000회,95% interval |

Methods와 actions는 동일한 outer latent draws와 Gaussian noise를 사용한다. Query truth, hidden scene parameters, 아직 선택하지 않은 actual observation은 acquisition에 전달하지 않았다. `run.py`는 기존 case JSON을 연 뒤 factual observation·family·split만 남기고 나머지를 버린다. `adaptive_plan.py`는 posterior summary만 읽는다. Outcome score는 `analyze.py`의 evaluation 단계에서만 읽는다.

Family rule과 더 강한 family×query rule은 validation 24 worlds의 actual-outcome MSE로 한 번 고정한다. 후자는 같은 query에서도 factual evidence에 따라 observation을 바꿀 수 있는 EVSI와 비교하기 위한 strong fixed rule이다. All-action random은 실제 random draw가 아닌 네 action의 outcome score 평균이다. 모든 observation의 비용은 같다고 가정했다.

결과 표의 `type_rule`은 family별 고정 action, `value_rule`은 family×query별 고정 action이다. 두 rule은 test factual observation에 따라 action을 바꾸지 않는다.

## 5. Acquisition estimate와 adaptive outer budget

Current predictive mean을 $m_0(q)$, hypothetical observation 뒤 predictive mean을 $m_y(q)$라고 하면 squared-norm EVSI는 $U(a,q)=E_y\|m_y(q)-m_0(q)\|^2$다. 보고하는 MSE는 두 velocity components의 평균이므로 이 utility는 expected MSE reduction의 2배 scale이다. Action ranking에는 영향이 없다.

Finite inner sampling의 positive variance contribution을 줄이기 위해 independent inner runs 두 개로 $(\hat m_y^{(1)}-m_0)^T(\hat m_y^{(2)}-m_0)$를 계산했다. 두 estimates가 독립이고 unbiased이면 기대값은 squared mean shift와 같다. 실제 finite SMC는 biased일 수 있고 shared factual-bank error도 있으므로 완전한 unbiased estimator라고 주장하지 않는다. 음수 추정치를 clipping하지 않았으며 single-run squared-norm ablation도 함께 계산했다.

Main outcome를 분석하기 전에 다음 refinement 규칙을 고정했다. 각 method/quality에서 common-outer bootstrap 1000회의 winner regret upper 95%가 `max(1e-4, 0.05 × maximum EVSI)` 이하이면 practical confidence를 통과한다. Maximum EVSI <=1e-4 query는 inactive로 따로 집계한다. Bridge 선택의 full-utility regret도 같은 tolerance 이하여야 한다. 하나라도 실패한 world는 128→256→512 outer samples로 순차 확장했다. 최대 budget에서 남는 ambiguity는 unresolved로 남기고 policy 평가에서 제외하지 않았다.

이는 exact top-action certainty와 다르다. Near-tie의 action name이 달라도 utility loss가 충분히 작을 수 있다. 또한 adaptive stopping 뒤의 bootstrap은 엄밀한 sequential confidence guarantee가 아니므로 numerical diagnostic으로만 해석한다.

최종 world별 outer budget 분포: 64: 15 worlds, 128: 11 worlds, 256: 13 worlds, 512: 33 worlds.

아래 agreement/confidence 집계에서 active query는 full SMC의 maximum EVSI >1e-4를 기준으로 한다. Refinement 여부를 정할 때에는 두 methods 각각의 active queries를 따로 검사한다.

| Condition | Budget | Active queries | Bridge/full top-action agreement | Bridge practical confidence | Full practical confidence | Bridge exact-winner decisive |
| --- | --- | --- | --- | --- | --- | --- |
| high | 64 outer | 249 | 96.8% | 72.7% | 71.5% | 32.5% |
| high | Adaptive | 249 | 98.8% | 96.0% | 96.0% | 54.2% |
| pixel_only | 64 outer | 249 | 96.0% | 68.3% | 69.5% | 28.9% |
| pixel_only | Adaptive | 249 | 99.2% | 96.4% | 96.0% | 48.2% |

최종 unresolved worlds는 18개이며 case IDs는 `[24, 26, 28, 29, 30, 31, 32, 33, 36, 37, 38, 40, 43, 44, 46, 58, 64, 71]`다. 이 world들도 아래 policy 평균과 bootstrap에 포함했다. 따라서 전체 ranking이 완전히 수렴했다고 해석하지 않는다.

## 6. ESS collapse가 실제로 줄었는가

| World/hypothesis | 기존 importance ESS | 기존 RMSE | Bridge RMSE | Full RMSE | Bridge run 간 RMSE |
| --- | --- | --- | --- | --- | --- |
| 66/110 | 1.00 | 0.267773 | 0.000964 | 0.001789 | 0.001431 |
| 66/107 | 1.21 | 0.048341 | 0.000675 | 0.001261 | 0.000870 |
| 66/82 | 1.95 | 0.092456 | 0.000590 | 0.001061 | 0.004203 |
| 66/50 | 9.63 | 0.032779 | 0.001680 | 0.001938 | 0.002689 |
| 68/110 | 1.38 | 0.060779 | 0.003792 | 0.002305 | 0.002752 |
| 68/27 | 3.52 | 0.046937 | 0.000452 | 0.001135 | 0.002333 |
| 68/81 | 4.74 | 0.120531 | 0.001887 | 0.001851 | 0.002402 |
| 68/68 | 35.32 | 0.011469 | 0.007413 | 0.004729 | 0.009357 |

모든 RMSE 단위는 m/s다. 기존 importance ESS의 denominator는 2048, 새 audit의 N은 4096이므로 ESS 절대값끼리 단순 비교하지 않는다. 기존 failure subset은 낮은 ESS 때문에 미리 선택된 diagnostic cases이며 일반적인 평균 개선율을 추정하기 위한 random sample이 아니다.

![ESS failure recovery](results/analysis/failure_recovery.png)

World68/hypothesis68처럼 bridge run 간 차이가 남는 case도 있다. Stage ESS constraint는 weight degeneracy를 제한하지만 multimodal mixing, factual-bank support 누락, finite SMC bias를 자동으로 해결하지 않는다.

### 6.1 Proposal 방향 전환과 tempering 효과의 구분

추가 control로 같은 factual bank 4096 particles를 한 번만 reweight하는 **one-step factual-bank SNIS**를 계산했다. Tempering과 MH rejuvenation이 없다. v5의 failure estimator는 image-conditioned proposal에 factual likelihood를 적용한 것이므로, 새 bridge와의 차이를 모두 tempering 효과로 해석하면 proposal 방향 전환이라는 confound가 생긴다.

| World/hypothesis | One-step factual-bank ESS | One-step RMSE | Tempered bridge RMSE |
| --- | --- | --- | --- |
| 66/110 | 811.9 | 0.001538 | 0.000964 |
| 66/107 | 306.1 | 0.001867 | 0.000675 |
| 66/82 | 889.2 | 0.001674 | 0.000590 |
| 66/50 | 557.1 | 0.006715 | 0.001680 |
| 68/110 | 121.0 | 0.020969 | 0.003792 |
| 68/27 | 341.8 | 0.003656 | 0.000452 |
| 68/81 | 387.0 | 0.007696 | 0.001887 |
| 68/68 | 225.8 | 0.007523 | 0.007413 |

| Condition | One-step/full predictive RMSE median/max | One-step/full active rank agreement | Matched-outer SNIS MSE | 512-outer SNIS MSE | Bridge − matched SNIS MSE [95% interval] |
| --- | --- | --- | --- | --- | --- |
| high | 0.011888 / 0.049108 | 96.8% | 0.008790 | 0.008775 | -0.000015 [-0.000047, +0.000006] |
| pixel_only | 0.031373 / 0.084004 | 71.5% | 0.008606 | 0.008575 | -0.000393 [-0.002965, +0.002674] |

SNIS의 hypothetical observations는 SMC와 같은 draws이며, 이 control의 posterior bank와 hypothetical generator가 같은 bank를 공유한다. Self-inclusion 때문에 narrow likelihood에서 optimistic finite-bank effect가 생길 수 있다. 따라서 이 control도 continuous posterior reference는 아니다. Failure subset의 복원은 factual-posterior 방향 전환과 continuous rejuvenation을 포함한 전체 변경의 결과이며, tempering만의 단독 기여로 주장하지 않는다.

Matched-outer SNIS는 world별 outer observation 수와 draws를 맞췄다는 뜻이다. SNIS는 bank 4096개 전체를 사용하고 main/refinement SMC는 1024 particles를 사용하므로 particle 수나 compute가 동일한 비교는 아니다. SNIS의 512-outer 결과도 저렴한 baseline의 별도 budget으로 보고했으며, 이를 기준으로 SMC stopping criterion을 바꾸지 않았다.

72 worlds ×512 outer의 SNIS per-case timer 합은 108.3 seconds였다. Factual bank 구축 비용은 제외한다. 이 timer는 뒤의 SMC-call-only timer와 포함하는 작업이 달라 직접적인 speedup ratio로 쓰지 않지만, 이 control이 실용적으로 저렴한 대안임을 보여 준다.

| Condition | Method | One-step ESS median/min | Stage ESS minimum | Mean stages | Mean acceptance |
| --- | --- | --- | --- | --- | --- |
| high | bridge | 54.7 / 1.00 | 768.0 | 5.90 | 15.5% |
| high | full | 6.3 / 1.00 | 768.0 | 8.87 | 25.6% |
| pixel_only | bridge | 23.5 / 1.00 | 768.0 | 7.73 | 14.5% |
| pixel_only | full | 4.0 / 1.00 | 768.0 | 10.50 | 23.9% |

이 표는 fixed-budget main N=1024의 validation/test 합계 72 worlds에서 집계했다. Bridge의 one-step ESS는 factual-bank particles를 한 번 reweight할 때, full의 one-step ESS는 prior particles에 joint likelihood를 한 번 적용할 때의 값이다. 두 값은 출발 distribution이 다르다.

## 7. Inner posterior와 higher-budget reference 비교

| Condition | Bridge/full predictive RMSE median/max | Bridge repeat RMSE median | Full repeat RMSE median | Any-contact probability MAE |
| --- | --- | --- | --- | --- |
| high | 0.007813 / 0.027630 | 0.010003 | 0.010798 | 0.001898 |
| pixel_only | 0.007713 / 0.026468 | 0.010483 | 0.009999 | 0.001671 |

Predictive RMSE는 같은 outer observations에서 두 inner runs의 평균끼리 비교한 값이다. Repeat RMSE는 각 method 안의 두 runs 사이 차이다. Contact probability는 any-contact indicator에 대한 값이며 contact sequence 전체 distribution의 일치를 검증한 것은 아니다.

따라서 여기서의 bridge/full agreement는 선택한 posterior functionals에 대한 agreement다. 전체 joint posterior의 distributional convergence나 모든 tail events의 정확성을 증명한 것은 아니다.

| Case | Condition | Higher-budget bridge/full RMSE | Bridge inner-budget rank agreement | Full inner-budget rank agreement | Bridge inner-budget utility MAE |
| --- | --- | --- | --- | --- | --- |
| 1 | high | 0.002011 | 6/6 | 6/6 | 0.000028 |
| 1 | pixel_only | 0.001916 | 6/6 | 6/6 | 0.000061 |
| 25 | high | 0.008723 | 6/6 | 6/6 | 0.000789 |
| 25 | pixel_only | 0.008620 | 5/6 | 6/6 | 0.000838 |
| 49 | high | 0.008339 | 6/6 | 6/6 | 0.000422 |
| 49 | pixel_only | 0.007821 | 6/6 | 6/6 | 0.000446 |
| 66 | high | 0.005012 | 6/6 | 6/6 | 0.000199 |
| 66 | pixel_only | 0.004879 | 6/6 | 6/6 | 0.000167 |
| 68 | high | 0.004655 | 6/6 | 5/6 | 0.000189 |
| 68 | pixel_only | 0.002980 | 6/6 | 6/6 | 0.000149 |

Inner-budget rank agreement는 동일한 앞 64 outer observations에서 main(1024/12)과 강화 audit(2048/18)를 비교했다. Audit의 128개 전체를 쓰는 비교와 분리해 inner budget 증가 효과를 본 것이다. 선택된 5 worlds에 대한 결과이므로 전체 posterior convergence의 증명은 아니다.

이 audit에서 bridge와 full은 각각 60개 case×quality×query comparisons 중 59개의 action을 유지했다. Bridge는 case25/pixel_only의 query index 3, full은 case68/high의 query index 3에서 바뀌었다. 따라서 검사한 범위에서 inner-budget 증가에 따른 ranking 변화는 작았지만 완전히 없지는 않았다.

## 8. Observation selection의 실제 이득

CRPS는 각 velocity component의 marginal CRPS를 평균한 값으로 단위는 m/s다. 90% coverage 역시 component별 marginal interval coverage의 평균이며 joint 2D credible-region coverage가 아니다.

### high

| Policy | Velocity MSE | Marginal CRPS | 90% coverage |
| --- | --- | --- | --- |
| bridge | 0.008775 | 0.029455 | 93.1% |
| full | 0.008778 | 0.029493 | 93.2% |
| value_rule | 0.011213 | 0.031603 | 92.9% |
| type_rule | 0.019740 | 0.043564 | 92.7% |
| always_view | 0.019111 | 0.055314 | 93.9% |
| random | 0.022281 | 0.050388 | 93.1% |
| bridge_EIG | 0.018084 | 0.042821 | 93.1% |
| bridge_probe_only | 0.011049 | 0.030977 | 93.1% |
| v5_bank | 0.008788 | 0.029291 | 93.2% |
| v5_conditional | 0.009598 | 0.029538 | 93.2% |
| no_observation | 0.050812 | 0.090587 | 94.8% |

| Paired MSE difference | Mean [95% interval] |
| --- | --- |
| Bridge − value_rule | -0.002438 [-0.004496, -0.000504] |
| Bridge − type_rule | -0.010966 [-0.019101, -0.004340] |
| Bridge − always_view | -0.010336 [-0.016173, -0.005127] |
| Bridge − random | -0.013506 [-0.020408, -0.007679] |
| Bridge − bridge_EIG | -0.009310 [-0.016707, -0.003192] |
| Bridge − bridge_probe_only | -0.002274 [-0.005714, +0.000096] |
| Bridge − v5_conditional | -0.000823 [-0.002882, +0.000348] |
| Bridge − full | -0.000004 [-0.000064, +0.000054] |

Bridge의 action counts `[view,left,right,center]`: `[57, 65, 70, 96]`. Full SMC: `[58, 65, 70, 95]`. Counts는 48 worlds ×6 queries이며 inactive query의 tie-breaking도 포함한다.

### pixel_only

| Policy | Velocity MSE | Marginal CRPS | 90% coverage |
| --- | --- | --- | --- |
| bridge | 0.008213 | 0.028092 | 93.4% |
| full | 0.008213 | 0.028150 | 93.4% |
| value_rule | 0.010358 | 0.030247 | 93.4% |
| type_rule | 0.019740 | 0.043564 | 92.7% |
| always_view | 0.009734 | 0.035292 | 95.0% |
| random | 0.019937 | 0.045382 | 93.4% |
| bridge_EIG | 0.009734 | 0.035292 | 95.0% |
| bridge_probe_only | 0.011049 | 0.030977 | 93.1% |
| v5_bank | 0.008585 | 0.030582 | 93.8% |
| v5_conditional | 0.008085 | 0.027775 | 93.2% |
| no_observation | 0.050812 | 0.090587 | 94.8% |

| Paired MSE difference | Mean [95% interval] |
| --- | --- |
| Bridge − value_rule | -0.002145 [-0.004322, -0.000100] |
| Bridge − type_rule | -0.011528 [-0.019578, -0.004879] |
| Bridge − always_view | -0.001522 [-0.004203, +0.001638] |
| Bridge − random | -0.011724 [-0.018043, -0.006496] |
| Bridge − bridge_EIG | -0.001522 [-0.004203, +0.001638] |
| Bridge − bridge_probe_only | -0.002837 [-0.005930, -0.000286] |
| Bridge − v5_conditional | +0.000128 [-0.000519, +0.000859] |
| Bridge − full | -0.000001 [-0.000005, +0.000002] |

Bridge의 action counts `[view,left,right,center]`: `[125, 41, 55, 67]`. Full SMC: `[126, 40, 55, 67]`. Counts는 48 worlds ×6 queries이며 inactive query의 tie-breaking도 포함한다.

![Policy comparison](results/analysis_adaptive/policy_comparison.png)

위 error bars는 policy별 world bootstrap interval이고, 통계적 비교에는 표의 paired difference interval을 사용한다. 288 queries를 independent samples로 취급하지 않았다. 여러 baselines를 비교한 exploratory analysis이며 multiple-comparison correction을 거친 confirmatory claim이 아니다.

Bootstrap은 고정된 validation rule, factual banks, 실제 observation realization에 conditional하다. Validation worlds를 다시 sampling해 rule을 재학습하는 uncertainty나 새로운 factual-bank seeds의 variation까지 포함한 interval은 아니다.

| Condition | 64-outer bridge MSE | Adaptive bridge MSE | 64-outer full MSE | Adaptive full MSE |
| --- | --- | --- | --- | --- |
| high | 0.010824 | 0.008775 | 0.011758 | 0.008778 |
| pixel_only | 0.008153 | 0.008213 | 0.008104 | 0.008213 |

Main과 adaptive의 차이는 acquisition 계산 budget에 따른 선택 변화다. 실제 observation 뒤 posterior fitting과 outcome score는 동일한 v5 값이다. 실제 측정값을 재수집하거나 새로 잘 맞게 fitting해서 생긴 차이는 아니다.

공유한 actual-observation posterior fitting도 finite SMC이다. 그 fitting 자체의 MC error를 제거한 exact Bayes risk를 평가한 것은 아니며, 이번 higher-budget audit는 hypothetical inference를 대상으로 한다.

각 world/action에는 하나의 realized observation만 있다. 따라서 expected utility가 거의 같은 두 action도 이번에 관측된 noise realization에서는 서로 다른 prediction error를 낼 수 있다. Numerical ranking agreement가 높다는 사실이 48-world realized MSE의 유의한 차이를 자동으로 보장하지 않는다. 별도 observation-noise repetitions와 새로운 worlds를 통한 확인이 필요하다.

### 8.1 Ranking이 안정된 queries에서의 결과

질문의 “ranking 안정화 뒤에도 이득이 남는가”를 직접 확인하기 위해 posthoc conditional diagnostic을 추가했다. Full EVSI가 1e-4보다 큰 active query 중 두 methods의 practical confidence가 모두 통과하고 bridge의 full-utility regret도 tolerance 이하인 queries를 practically stable로 정의했다. Exact-decisive subset은 여기에 두 methods의 exact-winner gap interval이 모두 양수인 조건을 더한다. 이 분류에는 actual outcome을 쓰지 않았고 adaptive stopping이나 policy를 바꾸지 않았다.

| Condition | Subset | Worlds/queries | Bridge MSE | Rule MSE | Bridge − rule [95% interval] | Bridge − SNIS [95% interval] |
| --- | --- | --- | --- | --- | --- | --- |
| high | practically_stable | 48/239 | 0.009452 | 0.010869 | -0.001417 [-0.003304, +0.000324] | +0.000003 [-0.000001, +0.000010] |
| high | exact_decisive | 45/134 | 0.010001 | 0.011215 | -0.001214 [-0.003060, +0.000333] | +0.000000 [+0.000000, +0.000000] |
| high | unresolved_active | 8/10 | 0.005247 | 0.044329 | -0.039082 [-0.059562, -0.018169] | -0.000319 [-0.000833, +0.000000] |
| pixel_only | practically_stable | 48/238 | 0.008722 | 0.011342 | -0.002620 [-0.004976, -0.000381] | -0.000207 [-0.002957, +0.003265] |
| pixel_only | exact_decisive | 43/118 | 0.009204 | 0.013143 | -0.003939 [-0.007693, -0.000368] | -0.001824 [-0.004085, +0.000044] |
| pixel_only | unresolved_active | 9/11 | 0.009972 | 0.007964 | +0.002008 [-0.001368, +0.007459] | -0.006777 [-0.023598, +0.005485] |

Subset 안에서 각 world의 eligible queries MSE를 먼저 평균한 뒤 worlds를 동일 가중치로 평균했다. CI도 eligible worlds 단위 paired bootstrap이다. 따라서 전체 48-world 평균과 estimand가 다르며, 쉬운 inference problems가 더 많이 포함될 수 있다. 이것은 새로운 independent test나 안정성이 성능을 개선한다는 causal estimate가 아니다. Unresolved queries를 포함한 앞의 전체 결과가 primary다.

`high`의 stable subset은 239 queries이며 bridge−rule 차이는 −0.001417, 95% interval은 [−0.003304,+0.000324]다. `pixel_only`의 stable subset은 238 queries이며 차이는 −0.002620, interval은 [−0.004976,−0.000381]다. Exact-decisive subset에서도 같은 정성적 결론이 나온다. 그러나 두 조건 모두 stable subset에서 SNIS 대비 interval은 0을 포함한다.

전체 288-query 평균에 대한 차이로 다시 분해하면 `high`의 bridge−rule MSE 차이 −0.002438 중 −0.001074가 unresolved 10 active queries에서 나온다. 약 44%이므로 전체 평균의 개선을 안정된 queries 전체의 확실한 우위로 일반화하면 안 된다. `pixel_only`에서는 stable queries가 −0.002286, unresolved queries가 +0.000141을 기여해 전체 −0.002145가 된다. 이 분해는 앞 표의 eligible-world 평균과 달리 모든 query에 동일한 1/288 weight를 준 값이다.

### 8.2 Family별 test MSE

| Condition | Family | Test worlds | Bridge | Full | Rule | Always-view | Bridge − rule |
| --- | --- | --- | --- | --- | --- | --- | --- |
| high | corner | 16 | 0.000718 | 0.000718 | 0.000726 | 0.007028 | -0.000007 |
| high | gate | 16 | 0.019540 | 0.019603 | 0.026051 | 0.030242 | -0.006511 |
| high | deflection | 16 | 0.006066 | 0.006014 | 0.006863 | 0.020062 | -0.000797 |
| pixel_only | corner | 16 | 0.000620 | 0.000620 | 0.000726 | 0.003262 | -0.000106 |
| pixel_only | gate | 16 | 0.021321 | 0.021319 | 0.026051 | 0.022264 | -0.004730 |
| pixel_only | deflection | 16 | 0.002696 | 0.002701 | 0.004296 | 0.003676 | -0.001600 |

각 family는 16 test worlds뿐이다. 이 표는 평균 차이가 어느 family에서 발생하는지 보는 descriptive breakdown이며, family별 추가 significance claim을 위한 분석이 아니다.

## 9. View correction만의 효과인가

| Condition | v5 conditional | v6 view + v5 probes | v5 view + v6 probes | All v6 bridge | Single-inner-run v6 |
| --- | --- | --- | --- | --- | --- |
| high | 0.009598 | 0.009723 | 0.009952 | 0.008775 | 0.008756 |
| pixel_only | 0.008085 | 0.007077 | 0.008049 | 0.008213 | 0.008212 |

Mixed-estimator ablation은 저장된 utility columns만 교체한 diagnostic이다. v5와 v6는 outer draws와 MC budget이 다르므로 완벽한 factorial causal decomposition은 아니다. 다만 전체 개선을 “view ESS collapse 해결만의 효과”로 단정하는 것을 막는다. 이번 v6의 primary comparison은 동일한 observations/budget을 사용하는 bridge 대 full SMC다.

### 9.1 같은 factual evidence인데 선택이 달라지는가

이번 데이터의 unique factual x-velocity는 corner 19개, gate 10개, deflection 18개다. 같은 family와 같은 factual observation이면 Bayesian target은 같아야 한다. 그런 worlds 사이의 action 차이를 곧바로 instance-specific intelligence라고 해석할 수 없다. 일부는 numerical variation일 수 있다.

| Condition | Budget | Same-evidence active pair disagreements | Unpooled MSE | Pooled MSE | Pooled − unpooled [95% interval] |
| --- | --- | --- | --- | --- | --- |
| high | main | 66/284 | 0.010824 | 0.008891 | -0.001933 [-0.005620, +0.000236] |
| pixel_only | main | 53/284 | 0.008153 | 0.008111 | -0.000041 [-0.000694, +0.000518] |
| high | adaptive | 14/284 | 0.008775 | 0.008849 | +0.000075 [-0.000002, +0.000213] |
| pixel_only | adaptive | 46/284 | 0.008213 | 0.008165 | -0.000048 [-0.000403, +0.000199] |

이 posthoc diagnostic은 동일한 inference problem의 independent Monte Carlo calculations를 pool하고 pooled current mean으로 EVSI를 다시 계산한다. Actual outcomes는 pooling에 사용하지 않는다. 서로 다른 worlds의 likelihood를 곱하거나, 같은 물체를 반복 관측했다고 가정한 것이 아니다. Primary adaptive policy와 stopping criterion에는 이 결과를 사용하지 않았다.

## 10. 계산 비용과 실행 증거

| Condition | Budget | Bridge seconds | Full seconds | Full/bridge time ratio | Full/bridge rollout ratio |
| --- | --- | --- | --- | --- | --- |
| high | Main64 | 7175.3 | 8436.0 | 1.18× | 1.50× |
| high | Final selected budget | 5674.0 | 6603.2 | 1.16× | 1.35× |
| pixel_only | Main64 | 7790.6 | 9015.8 | 1.16× | 1.41× |
| pixel_only | Final selected budget | 6362.0 | 7278.3 | 1.14× | 1.29× |

Seconds는 각 batched SMC call의 합이며 wall-clock completion time과 다르다. Final selected budget은 각 world에서 최종 사용한 level의 비용이며 이전 refinement level까지 더한 총 비용은 아니다. 두 quality는 dynamic-probe 계산을 공유하므로 표의 두 행을 합하면 중복 계산된다. Factual posterior bank 구축, final query prediction, validation, bootstrap, I/O는 이 SMC timer에 포함하지 않는다. Full/bridge rollout ratio도 inference 내부 forward trajectory만 집계한다. Equal particle/MH budget 비교이며 equal wall-time 최적화 비교는 아니다.

Cost table은 validation/test 합계 72 worlds의 inference 비용을 집계한다. 앞의 policy 성능 표는 48 test worlds를 사용한다. 여기서 timer에서 제외한 validation은 Gaussian/API correctness 검사를 뜻하며, 24 validation worlds의 acquisition 계산은 cost table에 포함된다.

Outer128 일부부터 같은 family의 최대 4 worlds를 함께 batch 처리했다. Population별 target과 budget은 동일하고 shared model parameter를 fitting한 것이 아니다. Grouped output의 per-world time/rollouts는 batch total의 equal-share attribution이다. 실제 batch total과 group/seed를 별도 기록했으며 primary efficiency comparison은 execution 변경 전의 Main64다. 동일 input/RNG의 API equivalence 및 mixed-factual likelihood 검사는 세 family에서 bitwise equality를 통과했다. 실제 grouped runs의 RNG stream은 group 구성에 따라 달라지므로 기존 run과 같은 random realization이라고 주장하지 않는다.

Server directory: `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/tempered_bridge_v6`. 사용 allocation은 **2676323(node56)** 및 **2676324(node55)**이며, 시작 전 GPU process/memory를 확인했다. 당시 사용 중이던 2676322/2676325는 이 작업에 사용하지 않았다. 두 worker가 shared queue를 처리했다. 세부 commands, protocol, population diagnostics, seeds, host/job metadata는 코드와 결과에 남겼다.

2026-11-10 23:50 KST에는 기존 training이 종료된 **2676322/2676325(node55)**도 두 차례 확인에서 memory 0 MiB, compute processes 없음이었다. Idle GPU 사용 허용 범위 안에서 guard를 적용한 worker를 추가해 남은 refinement queue를 총 네 worker가 처리했다. Main64 timing은 이 추가 이전의 두-GPU 실행 결과다.

실행 중 네 GPU에 external training이 나타났을 때 우리 inference steps만 중지하고 완료 결과를 보존했다. 이후 해당 processes가 종료되고 반복 확인에서 idle임을 확인한 두 GPU에서 재개했다. 재개 worker는 15초마다 compute process를 검사하며 다른 process가 나타나면 자기 child만 중지한다. Busy-GPU 시작 거부도 검증했다. Interrupted calls는 완료 결과의 timer에 포함하지 않았으므로 표는 전체 소비 resource의 accounting이 아니다. Resource contention과 재개 이력이 있는 adaptive timing을 엄밀한 hardware throughput benchmark로 해석하지 않는다.

2026-11-11 01:49 KST 최종 resource 확인에서는 네 worker가 모두 exit code 0으로 종료됐고, 네 GPU의 compute processes가 비어 있으며 memory usage는 0 MiB였다. 사용자의 Slurm allocation은 취소하지 않았다. 확인 기록은 `results/validation/final_resource_check.json`에 있다.

재사용한 v5 입력 295개는 sealed manifest의 SHA-256과 모두 일치했다. 여기에는 factual banks, case data, calibration, evaluation scores, failure references, engine/belief/base sources가 포함된다. 실행 환경은 Python 3.9.18, PyTorch 2.7.1+cu128, NumPy 1.26.4, CUDA 12.8이며 RTX PRO 6000 Blackwell에서 float64로 계산했다.

Simulation과 SMC는 server GPU에서 수행했다. 결과의 bootstrap 집계, integrity 검사, figure 및 보고서 생성은 local analysis environment에서 수행하고 server에 함께 보존했다. Local analysis dependencies는 `requirements-analysis.txt`에 기록했다.

Bounded correlated Gaussian target을 dense quadrature와 비교한 validation도 통과했다. Bridge/full 2 runs씩에서 normalized posterior mean의 최대 absolute error는 0.001184였다. Physics failure audit는 별도 v5 implementation의 full SMC reference와 비교했다.

최종 integrity 검사에서는 4,260개 저장 batch, 824,320개 conditional SMC populations를 검사했다. Summary/log-evidence의 finite values, strictly increasing temperature와 최종 β=1, pre-resampling ESS threshold, history/stage count가 모두 통과했다. 705개 observation-prefix 비교도 exact equality를 통과했다. 이는 저장 결과와 실행 invariant 검사이며 posterior convergence의 별도 증명은 아니다.

재현 entry points: `bridge.py`, `run.py`, `pipeline.py`, `adaptive_plan.py`, `analyze.py`, `analyze_audits.py`. 실행 설정은 `PROTOCOL.md`, 입력과 resume 방식은 `README.md`를 참고한다. `server_run_manifest.json`은 결과 파일의 size/hash와 실행 metadata를 기록한다.

코드와 실행 증거: [bridge implementation](bridge.py), [protocol](PROTOCOL.md), [reproduction guide](README.md), [stable-query analysis](analyze_stable.py), [source/result manifest](server_run_manifest.json).

## 11. 연구적 해석과 남은 검증

이번 결과의 핵심은 correlated factual posterior를 재사용하면서 hypothetical-observation inference의 numerical failure와 task-aware selection의 이득을 분리해 검증했다는 점이다. Tempering만 구현한 것은 독립적인 논문 기여로 충분하지 않다. 논문 기여 후보는 contact-boundary ambiguity가 acquisition을 왜곡하는 실패 분석, 같은 physical-query 조건에서도 달라지는 observation value, 그리고 numerical uncertainty에 따라 inference budget을 배분하는 검증 가능한 시스템이다.

이번 결과에서는 특히 **정확한 hypothetical prediction과 더 나은 observation selection이 같은 성과가 아니었다.** Pixel-only에서 SNIS/full predictive RMSE median은 약 0.0314 m/s이고 active rank agreement는 약 71.5%로 bridge보다 나빴지만, realized selection MSE의 paired interval은 여전히 bridge의 우위를 뒷받침하지 못했다. High에서는 저렴한 SNIS의 선택 성능이 bridge와 거의 같았다. 따라서 다음 논문 주장에 필요한 것은 sampler를 더 복잡하게 만드는 것보다, 이런 posterior error가 실제 decision loss로 연결되는 조건을 새로운 worlds·반복 observation noise·cost-aware decisions에서 재현하는 것이다.

다만 이번 실험만으로 real-world active perception의 효과, 일반적인 3D geometry inference, learned dynamics의 우수성을 주장할 수 없다. Physics는 bounded 2D rigid-contact model이고 camera/family/segmentation assumptions가 강하다. Shared factual bank가 놓친 mode를 bridge가 반드시 복원하는 것도 아니다. Gaussian observation likelihood 및 fixed noise covariance의 model mismatch도 이번 비교에서는 바꾸지 않았다.

다음 검증은 독립적으로 생성한 새로운 worlds와 factual-bank seeds, observation cost가 다른 설정, contact sequence 및 rare-event prediction, simulator mismatch, 실제 collision videos 순서로 진행하는 것이 타당하다. Algorithm 측면에서는 contact-aware proposal이나 conditional transport가 더 적은 forward budget으로 full-reference 수준에 도달하는지를 equal-compute frontier로 비교해야 한다. 이번 결과는 그런 개선을 평가할 수 있는 reference와 failure diagnostic을 만든 단계다.

Real-video 실험은 이번 v6에서 새로 수행하지 않았다. v5의 contour audit를 이번 physics posterior 검증으로 대체해 해석하지 않는다.

작성 시각 (KST): 2026-11-11T01:53:28.369203+09:00.
