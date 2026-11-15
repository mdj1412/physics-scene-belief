# Expanded-prior control과 contact transitions — v15 진행 보고서

작성일: 2026-11-12 KST

**상태: 구현 및 server correctness checks 완료, 본 비교 실험은 GPU 충돌로 중단. Decision loss와 compute 비교 결과는 아직 없다.** 이 문서는 완료된 실험 결과 보고서가 아니다. 아래의 experimental protocol과 실제 검증된 내용을 구분한다.

## 2026-11-13 재개 확인

사용자의 GPU 해제 알림 후 allocation `2680491` / `node56`를 재확인했다. 처음에는 Blender PID `896170`이 약 5.9 GB를 사용 중이었다. 이후 process가 종료되고 30초 간격으로 idle 상태를 확인했지만, 실행 직전에 새 Blender PID `897182`가 약 5.9 GB를 점유해 guard가 worker 시작을 거부했다. 이번 재개 시도에서는 inference를 시작하지 않았다.

저장된 source 15개의 hashes는 모두 일치하며 생성 데이터도 보존되어 있다. Full policy freeze, evaluation, timing은 여전히 미완료다. 반복되는 Blender 작업의 종료 확인 또는 다른 idle allocation이 필요하며 추가 자동 재시도는 중단했다.

## 1. 현재 무엇까지 진행했는가

사용자 지정 project 아래에 `transition_control_v15`를 만들었다. `datacenter3`를 통해 기존 allocation `2680491`, compute node `node56`의 RTX PRO 6000 GPU를 확인하고 실행했다. Python 3.9/PyTorch environment는 이전 실험의 `continuous_v2/.venv`를 사용했다. Login node에서 simulation이나 inference를 실행하지 않았다.

매번 실행 전에는 GPU process가 없고 memory가 0 MiB였지만, 실행 도중 별도의 Blender process가 나타났다. Guard가 자신의 worker만 SIGTERM으로 중단한 횟수는 세 번이다. 다른 사용 중인 process와 allocation은 변경하지 않았다. 마지막 중단 기록에는 Blender PID `788931`, 약 4,490 MiB 사용이 남아 있다. 이는 중단 당시 기록이지 현재에도 그 process가 실행 중이라는 뜻은 아니다.

두 번째 시도에서는 matched condition의 320 worlds까지 진행 로그가 남았다. 그러나 cell 전체의 policy freeze 및 evaluation 단계에 도달하지 못했으므로 이를 완성된 test 결과로 취급하지 않는다. 이후 operational checkpoint 기능을 추가했다. Acquisition equations, seeds, sample sizes와 statistical criteria는 바꾸지 않았다. 세 번째 시도는 첫 checkpoint가 만들어지기 전에 중단됐다.

| 항목 | 상태 |
|---|---|
| Expanded-prior generator와 nominal/shifted contact models | 구현 완료 |
| Contact/no-contact gate와 slip/stick query | 구현 완료, branch checks 통과 |
| LMMSE, disagreement, nested expected-loss acquisition | 구현 완료, pilot 계산 검증 |
| QMC16k/64k, common decoder, correct-law diagnostic | 비교 pipeline 구현 완료 |
| Fresh 512 worlds의 public/private data | 서버 생성 완료 |
| 본 실험 policy freeze | 미완료 |
| Held-out decision loss, paired CIs | 미실행 |
| GPU/CPU online timing | 미실행 |
| 최종 superiority 또는 paper contribution 판정 | 보류 |

앞서 사용자가 요청한 “모두 사용하고 있다면 모델 학습이나 inference는 잠시 멈춰” 조건에 따라, 반복 충돌 이후 GPU 재시도를 중단했다. CPU로 전환하거나 안정적으로 사용할 GPU가 확보되면 이어서 수행할 수 있다.

## 2. Model mismatch와 acquisition approximation을 어떻게 분리하는가

V14의 `matched` condition은 base contact laws에서만 truth를 생성하고 inference는 flexible laws에도 prior mass를 두었다. 따라서 population prior까지 일치하는 control은 아니었다. V15에서는 truth와 inference가 같은 expanded prior를 사용한다.

- Geometry family 2개 × base frictionless/Coulomb 2개: 각 prior mass 1/8.
- Geometry family 2개 × flexible contact law: 각 prior mass 1/4.
- `s∼U[−0.4,0.4]`, `a∼U[−0.1,0.1]`, `e∼U[0.35,0.9]`, `b₀∼U[−4,2]`, `b₁∼U[−2,2]`.
- Unknown boundary offset `d∼U[−4,4] mm`를 truth와 inference에 동일하게 추가한다.
- Worlds는 iid draws, inference bank는 별도의 scrambled Sobol QMC samples다. True world를 bank에서 뽑지 않는다.

Nominal flexible contact law는 다음과 같다.

\[
j_t=v_t\,\sigma\left(b_0+b_1\log\frac{|v_t|+0.02}{0.5}\right).
\]

Shifted data는 같은 latent parameters와 observation noise를 사용하면서 flexible component의 tangential fraction만 다음으로 바꾼다.

\[
j_t=v_t\,\sigma(b_0)\left[0.2+\frac{0.8}{1+(|v_t|/s_0)^2}\right],
\qquad s_0=0.1+0.7\sigma(b_1).
\]

Base frictionless/Coulomb worlds는 unchanged negative controls다. 다음 세 cells를 비교한다.

| Cell | 실제 data-generating law | Inference law | 목적 |
|---|---|---|---|
| Matched / nominal | Expanded nominal prior와 law | Nominal | Population/model mismatch 제거 |
| Shifted / nominal | Flexible law만 변경 | Nominal | Misspecified inference/design |
| Shifted / correct | 같은 shifted data | 올바른 shifted law | Privileged correct-law diagnostic |

마지막 cell은 실제 배포 가능한 방법이나 새 알고리즘의 성능이 아니다. 어떤 contact law를 써야 하는지 알고 있다는 privileged assumption으로 model mismatch의 영향을 분리한다.

같은 shifted worlds에서 nominal selector의 action을 correct-law decoder로 재평가한다. 그러면 전체 차이를 다음처럼 분해할 수 있다.

\[
L_{\mathrm{nom},a_{\mathrm{nom}}}-L_{\mathrm{correct},a_{\mathrm{correct}}}
=\left(L_{\mathrm{nom},a_{\mathrm{nom}}}-L_{\mathrm{correct},a_{\mathrm{nom}}}\right)
+\left(L_{\mathrm{correct},a_{\mathrm{nom}}}-L_{\mathrm{correct},a_{\mathrm{correct}}}\right).
\]

첫 항은 같은 action에서 decoder 차이, 둘째 항은 correct decoder 아래의 selection 차이다. 두 항은 이 비교 경로에 따른 decomposition이며 보편적으로 독립적인 원인이라는 뜻은 아니다. Finite QMC integration error도 남는다. Matched/shifted 평균 차이만으로 pure mismatch effect를 추정하지 않는다.

## 3. Transition task를 어떻게 만드는가

Factual evidence는 9개 boundary profile samples와 factual outgoing velocity다. Profile noise SD는 1.2 mm, velocity noise SD는 0.07 m/s다. Query position은 `y=25 mm`다.

### Contact/no-contact

Factual profile에 straight-line least-squares fit을 하고 query position까지 extrapolate한 값을 public `x_end`로 사용한다. Candidate boundary가 `x(y)≤x_end`이면 contact, 그렇지 않으면 no-contact로 정의한다. No-contact일 때 outgoing velocity는 incoming velocity `(1,0.9)`를 유지한다.

이는 finite-reach contact gate가 붙은 local impulse surrogate다. 전체 trajectory를 적분하여 continuous collision detection을 수행한 simulator는 아니다. True geometry로 transition 근처 worlds만 사후 선택하지 않으며, public evidence에서 정한 query를 모든 hypotheses가 공유한다.

### Slip/stick

Public profile에서 추정한 slope `ŝ`를 `[-0.6,0.6]`으로 clip하고,

\[
r=0.22(1+0.625),\qquad v_y=\frac{r-\hat s}{1+r\hat s}
\]

로 query incoming velocity를 설정한다. Median restitution의 Coulomb transition을 public estimate로 겨냥하는 방식이다. Exact stick label은 residual tangential velocity의 절댓값이 `1e−8` 이하일 때 1이다. Flexible soft-friction law에는 일반적으로 exact stick이 없으므로, 이 task에는 contact-model identity에 대한 uncertainty도 포함된다.

| Task | Prediction | Loss |
|---|---|---|
| Smooth velocity | Always-contact outgoing velocity | Half squared Euclidean error |
| Contact velocity | Contact gate 이후 velocity | Half squared Euclidean error |
| Contact event | Contact probability | Brier loss |
| Slip velocity | Transition을 겨냥한 query velocity | Half squared Euclidean error |
| Stick event | Exact-stick probability | Brier loss |

Event tasks의 Bayes decision은 probability forecast다. Robot의 discrete control action 성공률을 측정한 것으로 표현하지 않는다. 구현에서는 event target에 `√2`를 곱해 공통 half-squared-error 계산이 Brier loss와 일치하도록 했다.

Observation actions는 한 번의 `query_view`, `factual_view`, `probe`이며, 각각 query 부근 boundary profile, factual 부근 정밀 profile, 다른 incoming velocity의 contact response를 제공한다. 모든 actions의 sensing cost는 같게 두었다. `none`은 별도 no-observation diagnostic이다. 실제 sensing/reset 비용이나 raw RGB processing은 이번 실험에 포함하지 않는다.

## 4. 무엇과 비교하는가

Primary comparison은 같은 expanded QMC16384 posterior/decoder에서 acquisition만 바꾼다.

1. **LMMSE:** Query target과 candidate observation의 covariance를 이용한 linear prediction gain.
2. **Disagreement:** V14의 K256 committee, factual log-likelihood cutoffs 8/2, query difference와 Gaussian observation discriminability의 pairwise score.
3. **Cross-fitted nested128/512:** Independent outer bank가 hypothetical world/observation을 생성하고, inner bank decoder의 squared error를 outer query target에 대해 직접 평균한다.
4. **Fixed actions:** Always-probe, always-query-view, always-factual-view, none.
5. **QMC65536:** LMMSE와 disagreement, fixed actions의 더 큰 bank 비교.
6. **Ablations:** Factual-fit filtering 제거, query weighting 제거, nested posterior-variance proxy.

LMMSE는 최종 predictor가 아니다. 모든 방법의 actual observation 이후 prediction은 같은 posterior mean이다. 따라서 final nonlinear inference를 일부러 약하게 만든 baseline과 비교하지 않는다.

Nested512는 V14의 expected posterior variance만 계산하는 방식과 달리, independent outer hypotheses의 targets를 사용하여 finite inner decoder의 expected loss를 직접 평가한다. 그렇더라도 finite inner/outer banks와 hypothetical samples를 사용하는 approximation이다. Exact oracle로 부르지 않는다.

첫 32 worlds에서는 nested2048의 두 hypothetical seeds, QMC64k inner/outer nested512, pair K512를 audit한다. Numerical audit를 본 뒤 primary K나 sampling settings를 바꾸지 않는다.

## 5. LMMSE failure를 무엇으로 판정하는가

LMMSE acquisition은 linear predictor의 설명력을 계산한다. Scalar target에서 observation과 target의 covariance가 0이면 gain은 0이지만, nonlinear dependence까지 없다는 뜻은 아니다. 예를 들어 symmetric hypotheses `h∈{−2,−0.5,0.5,2}`가 같은 확률이고 target이 `1{|h|<1}`이면 `Cov(h,target)=0`이다. 반면 `h`를 정확히 관측하면 target을 알 수 있다.

이 abstract mechanism check는 서버에서 통과했다.

- Covariance: **0**.
- No-observation Brier loss: **0.25**.
- Gaussian observation noise SD 0.1, 고정된 finite validation samples에서 posterior-mean Brier loss: 약 **1.66×10⁻³⁴**.

마지막 수치는 해당 finite synthetic check의 측정값이며, 모든 Gaussian noise realizations에서 error가 0이라는 뜻은 아니다. 이 four-hypothesis example은 expanded-prior physics population의 결과도 아니고 disagreement selector가 baseline보다 우수하다는 evidence도 아니다. Nonlinearity 때문에 LMMSE gain이 유용한 observation을 놓칠 수 있다는 원리만 검증한다.

실제 physics test에서는 다음을 사전에 고정했다.

- 512 independent worlds, two paired data-generating conditions, five query tasks, actual observation noise 16 repeats.
- Primary family: matched contact-event/stick-event에서 pair 대 LMMSE/always-probe/always-query/nested512, nested512 대 LMMSE. 총 10 contrasts.
- Independent world 단위 30,000 paired bootstrap, Bonferroni-adjusted 99.5% CI.
- LMMSE failure claim: nested512의 mean loss가 최소 10% 낮고 adjusted difference CI의 upper bound가 0보다 작아야 한다.
- Disagreement superiority claim: 해당 event task에서 네 baseline 모두에 adjusted upper bound가 0보다 작아야 한다.

Public factual posterior에서 `p∈[0.2,0.8]`인 ambiguous subgroup, posterior event variance가 남아 있지만 relative LMMSE gain이 작은 subgroup도 secondary diagnostics로 분석하도록 구현했다. Hidden target이나 realized loss로 subgroup을 정하지 않는다. 현재 이 subgroup의 결과는 계산되지 않았다.

## 6. Compute 비교와 numerical reliability

GPU와 CPU 2 threads에서 float64, batch1, fixed 20 worlds × 5 tasks × 3 repetitions의 online timing을 준비했다. Factual fit, public query에 따른 target generation, acquisition, 선택한 observation 하나를 반영한 update/prediction을 포함한다. Shared bank setup/cache는 별도로 기록한다.

Population evaluation은 여러 query targets의 계산을 함께 처리해 효율을 높이지만, timing에서는 한 task를 개별 처리한다. Unselected actual observations나 offline prediction cache로 online inference를 대체하지 않는다. 실제 선택과 prediction이 저장된 evaluation과 일치하는지도 검사한다.

QMC16k와 QMC64k의 posterior ESS 및 common-action decoder swaps를 함께 분석한다. Contact discontinuity 때문에 finite-bank approximation이 커졌다면, 이를 acquisition의 실패 또는 개선으로 오인하지 않아야 한다. 현재 timing 수치와 ESS 결과는 아직 없다.

## 7. 논문 기여 가능성: 지금은 판정 보류

Observation selection을 downstream decision에 맞춘다는 문제의식 자체는 새롭지 않다. EC²는 hypothesis pairs와 equivalence classes를 이용한 information gathering을 다뤘다. 이번 continuous pair score에 그 논문의 adaptive-submodularity guarantees를 그대로 적용할 수 없다. [Golovin, Krause, Ray, 2010](https://proceedings.neurips.cc/paper/2010/file/1e6e0a04d20f50967c64dac2d639a577-Paper.pdf)

최근 GoBOED는 amortized posterior와 differentiable convex decision layer를 결합해 goal-driven design을 수행한다. 현재의 contact-event forecasting 및 finite candidate observation selection과는 다르지만, decision relevance를 objective에 반영했다는 이유만으로 novelty를 주장하기 어렵다. 해당 논문은 convex downstream formulation을 중심으로 하며, 이번 실험에서는 그 learned method를 직접 구현해 비교하지 않았다. [Go, Qian, Yoon, 2026 preprint](https://arxiv.org/html/2605.26093v1)

Model mismatch가 design과 inference 양쪽을 잘못 이끌 수 있다는 문제도 이미 연구되고 있다. GBOED는 Gibbs inference와 Gibbs EIG를 통해 outliers/noise misspecification에 대응한다. 이번 contact-law structural mismatch와 같은 실험은 아니지만, observation selection만 고치면 mismatch를 해결할 수 있다는 단순한 claim에는 반례가 되는 연구 맥락이다. [Barlas, Sloman, Kaski, 2025 preprint](https://arxiv.org/html/2511.07671v1)

Maximin robust Bayesian experimental design은 adversarial perturbation을 고려한 robust objective와 tilted posterior update를 함께 유도한다. 이번 disagreement heuristic에는 이러한 robustness guarantee가 없다. 직접 재현한 baseline이 아니므로 우열을 주장하지 않는다. [Abdulsamad et al., 2026 preprint](https://arxiv.org/html/2603.14094v1)

실험을 마친 뒤의 판단 기준은 다음과 같다.

- **Matched control에서도 LMMSE가 충분하면:** 단순한 transition 추가로 새 acquisition method가 필요해졌다고 볼 수 없다.
- **Nested는 개선하지만 pair는 개선하지 못하면:** Nonlinear acquisition의 필요성은 생기지만 현재 pair surrogate의 contribution은 성립하지 않는다.
- **Pair가 강한 baseline 대비 좋은 loss–compute trade-off를 보이면:** Independent numerical audit, new-world replication, component ablations까지 유지되는지 확인해야 한다.
- **Correct-law decoder 효과가 selection 효과보다 크면:** Acquisition보다 model adequacy와 inference를 먼저 다루는 편이 타당하다.
- **QMC resolution에 결과가 크게 흔들리면:** Numerical integration을 먼저 안정화해야 하며 paper claim을 보류해야 한다.

V14의 negative result를 뒤집을 evidence는 아직 얻지 못했다. 이번 작업에서 검증된 것은 implementation checks와 명확한 control design이며, primary hypothesis의 성공 여부는 미정이다.

## 8. 파일과 재개 상태

Server directory:

```text
/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/transition_control_v15
```

Local mirror:

```text
/Users/mindongjun/Documents/Codex/2026-10-30/new-chat/work/physics-scene-belief/transition_control_v15
```

`PROTOCOL.md`에 primary design, `DIAGNOSTICS_SPEC.md`에 secondary subgroup criteria를 저장했다. `model.py`, `acquisition.py`, `run_policy.py`, `evaluate.py`는 simulation/inference/evaluation을 분리한다. `benchmark.py`, `analyze.py`, `diagnostics.py`는 timing과 statistics를 계산한다. `run_resume.py`가 operational restart를 담당한다.

`results/validation.json`과 `results/worker_exit.json`은 실제 server execution evidence다. 첫 두 중단 기록은 server의 `interrupted/first_gpu_attempt`, `interrupted/second_gpu_attempt`에 보존했다. `seal.py`는 완성된 실험에만 적용하도록 만들었으며, 지금은 최종 integrity seal을 실행하지 않았다. 이전 v14 results는 이번 구현에서 수정하지 않았다.

재개 시에는 새 GPU의 allocation/process/memory를 확인하거나 사용자의 CPU 실행 허용을 받은 뒤 진행한다. 현재는 worker가 종료되어 있고 자동 재시도나 예약 작업을 등록하지 않았다.
