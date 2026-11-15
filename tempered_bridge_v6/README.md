# Tempered SMC bridge v6

Factual posterior에서 hypothetical observation posterior로 이동하는 adaptive tempered SMC와 prior-start nested full SMC를 비교한다. v5의 simulator, frozen image-feature calibration, factual observations, actual-outcome posterior scores는 그대로 사용한다.

실험 설계와 사전 고정한 stopping criterion은 [PROTOCOL.md](PROTOCOL.md)에 있다. `bridge.py`가 inference, `run.py`가 hypothetical observation 생성 및 acquisition population 계산을 담당한다. `adaptive_plan.py`에는 hidden truth 또는 actual-outcome score를 읽는 코드가 없다. `analyze.py`에서만 policy outcome을 평가한다.

## 실행 환경

Server project: `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/tempered_bridge_v6`

Python: `../continuous_v2/.venv/bin/python`. Python 3.9.18, PyTorch 2.7.1+cu128, NumPy 1.26.4, float64 CUDA. 각 allocated GPU에 worker 하나를 사용한다. `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`.

아래 명령은 allocated compute node에서 실행한다. Login node에서는 실행하지 않는다. Job ID는 실행 시점의 allocation과 GPU process 상태를 확인해 지정한다.

```bash
../continuous_v2/.venv/bin/python validate.py
../continuous_v2/.venv/bin/python failure_audit.py
../continuous_v2/.venv/bin/python run.py
../continuous_v2/.venv/bin/python pipeline.py
../continuous_v2/.venv/bin/python analyze.py
../continuous_v2/.venv/bin/python verify_results.py
../continuous_v2/.venv/bin/python analyze.py --adaptive
../continuous_v2/.venv/bin/python analyze_audits.py
../continuous_v2/.venv/bin/python one_step.py
../continuous_v2/.venv/bin/python analyze_one_step.py
../continuous_v2/.venv/bin/python same_evidence.py
../continuous_v2/.venv/bin/python analyze_stable.py
../continuous_v2/.venv/bin/python make_report.py
```

`run.py`의 default는 72 worlds,64 outer,1024 particles,12 MH moves,2 independent inner repeats다. 두 worker가 같은 directory를 사용해 case를 atomic claim한다. `pipeline.py`는 main 72 worlds의 completion barrier를 기다린 뒤 uncertainty 기반 128/256/512 outer refinement와 5-world increased-inner-budget audit를 수행한다. Maximum budget에서 unresolved인 case 목록은 마지막 adaptive plan에 남는다.

현재 `pipeline.py`는 `grouped.py`를 사용해 같은 family의 최대 4 worlds를 함께 처리한다. 각 population의 Bayesian target과 budget은 독립적이다. Grouped statistics의 per-world time/rollouts는 batch total의 equal-share attribution이며 실제 batch totals, group IDs, seed도 저장한다. `validate_grouped.py`에서 API/likelihood의 bitwise equivalence를 확인했다. Main64와 먼저 완료한 outer128 결과는 기존 실행 방식으로 보존했다.

GPU 실험을 다시 시작할 때에는 먼저 기존 worker가 종료됐는지 확인한다. 정상 완료된 case에는 `complete.json`, 각 population batch에는 `.npz`와 `.json`이 있다. 중단된 claim은 자동 탈취하지 않는다. 해당 process의 종료를 확인한 뒤 담당 claim만 이동하고 재시작한다.

## 입력과 출력

- 입력: `../boundary_belief_v5/results/main/{case_###.json,factual_###.npz}` 및 v5 frozen calibration.
- `results/main`: matched-budget bridge/full comparison.
- `results/outer128`, `outer256`, `outer512`: numerical uncertainty에 따른 추가 계산.
- `results/adaptive_plans`: refinement 결정과 unresolved diagnostics.
- `results/failure_audit`: v5의 ESS-collapse hypothetical views 재계산.
- `results/budget_audit`: 128 outer,2048 particles,18 moves,2 repeats.
- `results/analysis`, `analysis_adaptive`: world-clustered bootstrap, ranking diagnostics, policy scores, figures.
- `results/one_step`: 같은 hypothetical observations에 대한 factual-bank SNIS control. Factual bank와 outer generator를 공유하므로 self-inclusion에 주의한다.
- `results/analysis/same_evidence.json`: 동일 family/factual observation의 independent numerical calculations를 pool한 posthoc diagnostic. Repeated likelihood를 곱하지 않는다.

`one_step_ess`는 tempering을 하지 않았을 때의 초기-bank importance ESS다. `min_stage_ess`는 각 tempering stage의 resampling **이전** ESS이며 두 값은 서로 다른 diagnostic이다. 높은 stage ESS만으로 mixing이나 posterior correctness를 보장하지 않는다.

Acquisition은 independent inner estimates 두 개의 cross-product를 사용한다. Finite SMC bias와 factual-bank approximation은 남는다. Full SMC 역시 exact oracle이 아니다. `full_EIG`는 shared factual log-evidence constant를 생략하므로 action ranking만 해석한다.

Local `.venv-analysis`는 결과 검토와 figure 생성만 위한 별도 환경이며 remote inference 환경에 설치하거나 업로드하지 않는다.

재개는 allocation 및 idle 상태를 확인한 compute step 안에서 `resume_worker.py`를 실행한다. 이 wrapper는 시작 시 occupied GPU이면 거부하고 실행 중 15초마다 foreign compute process를 검사한다. 다른 process가 나타나면 자기 child만 중지하고 자기 미완료 claims만 보관 이동한다. `status.py --gpu`는 inference 없이 completion/resource 상태를 확인한다.
