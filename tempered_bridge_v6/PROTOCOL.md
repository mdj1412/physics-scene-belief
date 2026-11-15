# Tempered bridge v6: 실행 전 고정한 비교

작성: 2026-11-10. v5의 결과를 이미 알고 진행하는 후속 numerical audit이며 새로운 independent test set이라고 부르지 않는다.

## 검증 가설

1. Factual posterior에서 hypothetical observation likelihood만 tempering하면 one-step importance weighting의 ESS collapse를 줄이면서 nested full SMC의 predictive distribution에 접근한다.
2. 같은 hypothetical observation과 같은 particle/MH budget에서 bridge가 full SMC보다 candidate evaluation을 적게 사용한다. Factual bank 구축 비용은 공유되며 별도임을 밝힌다.
3. Inner inference와 outer integration의 안정성을 확인한 뒤에도 task-aware observation selection이 validation에서 고정한 rule보다 counterfactual prediction을 개선하는지 확인한다. 유의하지 않거나 ranking이 불안정하면 그대로 보고한다.

## 데이터와 예산

- v5의 동일한 72 worlds, corner/gate/deflection 각각 24. 각 family 앞 8 validation, 뒤 16 test. World당 6 intervention queries. Main test 48 worlds를 bootstrap 단위로 사용한다.
- Factual bank: v5 continuous SMC 4096 particles. Sensor: finite-frame 50 ms, 동일 calibration, high 및 pixel_only boundary representation. Low 및 bbox는 이번 inference 비교에서 제외한다.
- Main: 64 hypothetical observations, 1024 particles, 12 MH moves/stage, independent inner runs 2회, ESS threshold 0.75N. Bridge와 prior-start full 모두 실행한다.
- Actions: view/left/right/center. Dynamic actions는 두 view quality에 공통이므로 한 번만 계산한다. 총 5 conditions × 2 methods × 2 runs × 72 worlds.
- Outer samples: NumPy seed 85000+case; fixed 1024-element draw/noise pool의 prefix. Methods와 actions는 latent draws 및 Gaussian noise를 공유한다. Main inner seeds는 run.py에 고정한다.
- 두 GPU에서 한 worker씩 실행하고 case를 atomic claim한다. Existing user jobs는 수정하지 않는다.

## 알고리즘과 평가

Bridge target: p(theta) L_f(theta) L_a(theta)^beta. Full target: p(theta)[L_f(theta)L_a(theta)]^beta. Adaptive bisection, systematic resampling, covariance/local/global MH mixture를 사용한다. Factual likelihood의 exponent를 bridge에서는 항상 1로 유지한다.

Acquisition은 conditional predictive mean 변화의 squared norm EVSI. 두 independent inner estimate의 cross-product를 사용해 finite-inner variance의 leading positive contribution을 줄인다. 음수 추정치를 clipping하지 않는다. Cross-product는 finite SMC bias와 shared factual-bank error를 없애지 않는다. Single-run squared-norm 값도 함께 분석한다.

동일 outer draws에서 bridge/full predictive mean RMSE, contact probability 차이, action rank agreement, utility regret를 비교한다. 16/32/64 outer prefix와 half-sample ranking을 비교해 outer MC uncertainty를 별도로 평가한다. 높은 agreement만으로 모든 query가 converged라고 하지 않는다.

Policy evaluation은 선택된 action에 해당하는 v5 actual-observation full SMC posterior score를 재사용한다. 선택 알고리즘에는 factual observation, family, query, frozen calibration만 제공한다. Actual observation, hidden parameters, query truth는 평가 모듈에서만 읽는다.

Baselines: validation-only family rule 및 family×query rule, always-view, expected uniform random, probe-only EVSI, v5 bank EVSI, v5 refined-conditional-view EVSI, nested full SMC EVSI. Main metric은 velocity MSE; CRPS와 coverage를 보조로 보고한다. Paired world bootstrap 10000회, 95% interval. 여러 비교에 대한 confirmatory significance를 주장하지 않는다.

## 강화 audit

- v5에서 미리 발견한 case66/68의 lowest-three 및 median-ESS hypothetical views: 4096 particles,18 MH,2 seeds의 bridge/full을 기존 independent full references와 비교한다. Diagnostic subset이며 population-wide estimate가 아니다.
- Main 이후 inner/outer budget 증가 audit: validation case1/25/49와 failure case66/68, 128 outer,2048 particles,18 MH,2 seeds. 선정은 기존 failure와 family coverage에 근거하며 test outcome 개선 여부에 근거하지 않는다.
- 필요하면 ranking ambiguity에 따른 추가 outer samples를 실행하되 변경 이유와 시점을 기록한다. Test MSE를 보고 budget 또는 policy를 선택하지 않는다.

## Outer stability 확장 규칙 (main outcome 분석 전, validation case1 diagnostic 뒤 고정)

Exact winner는 near-tie에서 작은 numerical noise에도 바뀔 수 있다. 두 methods 각각 common-outer bootstrap 1000회에서 현재 winner의 regret upper 95%가 `max(1e-4, 0.05 * maximum EVSI)` 이하인지를 practical confidence 기준으로 사용한다. EVSI는 squared velocity norm 단위이고 maximum EVSI <=1e-4 query는 inactive로 따로 보고한다. Bridge/full 간 full-utility regret도 같은 tolerance 이하여야 한다.

한 quality/query라도 이를 통과하지 못한 world는 128→256→512 outer samples로 순차 확장한다. Inner 설정은 1024 particles,12 MH,2 repeats를 유지하고 methods/qualities 모두 동일한 새 outer prefix에서 계산한다. 최대 budget에서도 실패하면 unresolved로 표시하며 제외하지 않는다. 이 규칙은 posterior summary만 사용하고 actual outcome score는 읽지 않는다. Adaptive stopping의 bootstrap confidence는 엄밀한 sequential confidence guarantee가 아니며 numerical diagnostic으로만 해석한다.

## 실행 방식 변경 기록 — 2026-11-10 22:10 KST

Main 72-world comparison과 outer128의 12 worlds를 완료한 뒤, CPU/kernel-launch overhead를 줄이기 위해 같은 family의 최대 4 worlds를 함께 처리하는 cross-world batching으로 남은 실행을 전환했다. Population마다 factual likelihood, hypothetical observation, factual bank, temperature, covariance, resampling이 독립적이다. Target, particle count, moves, outer draws, refinement criterion은 바꾸지 않았다. `validate_grouped.py`는 세 family의 bridge/full에서 동일한 input/RNG의 scalar-versus-population API가 bitwise-identical한 particles/log-evidence를 내고, 서로 다른 factual observations의 batched likelihood가 개별 계산과 bitwise-identical함을 확인했다.

실제 grouped runs의 RNG stream은 group 구성에 따라 달라지므로 기존 per-world runs와 같은 random realization이라고 주장하지 않는다. Batch identity, case IDs, seed를 각 결과에 저장한다. `seconds`와 `trajectory_rollouts`의 per-world 값은 batch total의 equal-share attribution이며, actual batch totals도 별도 저장한다. Primary cost comparison은 실행 방식 변경 전의 matched main64 결과다. 완료된 결과를 재계산하거나 outcome에 따라 고르지 않았다. 중단한 우리 step의 미완료 claims만 보관 이동했고 기존 allocation 및 다른 사용자 작업은 변경하지 않았다.

`safe_invalid` proposal-evaluation 변형은 별도 bitwise validation을 수행했으나 main/refinement에는 적용하지 않았다(default False).

### 추가 해석용 controls

이후 동일한 family/factual observation의 numerical-invariance를 보는 pooled-estimator diagnostic과, 같은 hypothetical observations에 대한 one-step factual-bank SNIS control을 추가한다. 이 둘은 posthoc diagnostic이며 이미 고정된 adaptive policy를 선택하거나 stopping rule을 바꾸는 데 사용하지 않는다. 특히 기존 image-conditioned importance proposal의 실패가 factual-posterior 방향 전환만으로 줄어드는지와 tempering/rejuvenation의 추가 효과를 구분한다.

### Stable-query conditional diagnostic — 2026-11-10 23:20 KST 추가

사용자가 요청한 ranking 안정화 뒤의 이득을 직접 확인하기 위해 `analyze_stable.py`를 추가했다. Full EVSI >1e-4인 active query 중 두 methods의 practical confidence를 모두 통과하고 bridge의 full-utility regret도 tolerance 이내인 subset을 평가한다. Exact-decisive subset은 양쪽 exact-winner gap interval이 양수인 조건을 더한다. 같은 query mask를 모든 policies에 적용하고 eligible query의 world별 MSE를 먼저 평균한 뒤 worlds를 평균/paired bootstrap한다. Inference-only posthoc 분류이며 actual outcomes를 사용해 subset을 고르지 않는다. Primary 전체-world 결과와 estimand가 다르고 어려운 inference cases가 빠질 수 있으므로 conditional diagnostic으로만 해석한다. Policy, stopping rule, compute budget에는 영향을 주지 않는다.

### Idle GPU 추가 — 2026-11-10 23:50 KST

Main64와 outer128을 완료하고 outer256을 실행하던 중, 기존 external training이 종료된 allocation 2676322/2676325의 GPU가 두 차례 확인에서 memory 0 MiB 및 compute processes 없음으로 확인됐다. 사용자의 idle GPU 사용 허용 범위 안에서 두 allocation에도 `resume_worker.py`를 실행해 shared queue worker를 두 개에서 네 개로 늘렸다. 각 GPU당 process 하나와 15초 runtime guard를 유지한다. 다른 작업은 중지하거나 수정하지 않는다. Statistical targets/budgets/criterion은 동일하고 main timing은 추가 이전의 결과다.

## 한계

Calibrated low-dimensional image features와 bounded 2D rigid-contact simulator이다. 이번 실행은 새로운 raw image baseline이나 real-video validation이 아니다. Full SMC도 approximate reference이며 bridge는 factual bank의 missed mode를 자동 복구한다고 보장하지 않는다. Tempering 자체는 established SMC methodology이며 새로운 algorithmic contribution으로 주장하지 않는다.
