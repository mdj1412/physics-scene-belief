# v6 실행 완료 및 과거 resume 기록

**2026-11-11 완료 상태:** main 72, outer128 57, outer256 46, outer512 33, higher-budget audit 5 worlds가 모두 완료됐다. 네 guarded workers는 exit code 0으로 종료됐고 사용자 allocations는 유지했다. `results/resource_state.json`의 상태는 `inference_complete`다. 최종 보고서는 [REPORT_KO.md](REPORT_KO.md), 결과 검사는 `results/validation/final_integrity.json` 및 `final_resource_check.json`에 있다.

최대 512 outer에서 18 worlds가 numerical uncertainty criterion을 완전히 통과하지 못했다. `outer1024.json`은 이 unresolved 목록을 저장한 diagnostic이며, 1024-outer 실행이 남아 있다는 뜻이 아니다. 이번 protocol의 inference는 완료됐다.

아래는 **2026-11-10 중단 시점의 historical checkpoint와 resume 방법**이다. 현재 진행 상황으로 읽지 않는다. 새로운 실험을 할 때에는 당시의 idle GPU와 유효한 allocation을 다시 확인하고 별도 output directory를 사용한다.

## 2026-11-10 22:45 KST checkpoint

현재 inference는 중단했다. 2026-11-10 22:45 KST에 네 할당 GPU 모두 외부 `doom2-vagen` training process를 사용하고 있음을 확인했다. 기존 allocation과 그 training은 변경하지 않았다.

완료: main 72 worlds, failure audit 8 hypothetical views, Gaussian/bitwise validations, one-step SNIS 72 worlds. Outer128 refinement는 선정된 57 worlds 중 20 worlds가 완전히 완료됐다. 나머지 world의 완료된 population files도 보존했다. Outer256/512 및 5-world increased-inner-budget audit는 아직 완료되지 않았다.

재개는 이전 외부 training process의 종료와 반복된 idle 상태 확인 후, 사용자가 허용한 idle GPU 범위 안에서 수행한다. 현재 살아 있는 allocation, 종료 시각, 실제 GPU memory/process를 먼저 확인한다. 과거 job ID가 계속 유효하다고 가정하지 않는다. 빈 GPU에 대해서만 다음 명령을 실행한다.

```bash
cd /scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/tempered_bridge_v6
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 srun --jobid=IDLE_JOB_ID --overlap --exact -N1 -n1 -c1 --gres=gpu:1 --time=06:00:00 ../continuous_v2/.venv/bin/python -u resume_worker.py
```

`resume_worker.py`는 allocated compute step 밖에서 실행되거나 GPU compute process가 하나라도 있으면 inference를 시작하지 않는다. 성공하면 새 log와 worker PID를 `results/resource_state.json`에 기록하고 `pipeline.py`를 실행한다. 15초마다 GPU process를 확인해 외부 process가 나타나면 우리 child worker만 중단하고, 그 worker의 미완료 claim만 해제한다. 두 GPU가 비어 있으면 각 allocation에서 worker 하나씩 실행할 수 있다. 이미 같은 GPU에서 worker가 실행 중이면 추가로 시작하지 않는다.

중단된 우리 worker의 미완료 claims는 `results/interrupted_claims`로 옮겼다. 완료된 `.json`/`.npz`는 재계산하지 않는다. `.npz`만 있고 matching `.json` marker가 없으면 미완료 결과로 보고 다시 계산한다. Factual data, hypothetical draws, particle/MH budget, stopping criterion은 `PROTOCOL.md`대로 유지한다.

진행 확인: `python3 status.py --gpu`. 외부 process가 해당 GPU에 새로 올라오면 우리 worker step만 중단한다. 다른 process나 allocation은 취소하지 않는다.

완료 뒤 analysis 순서:

```bash
../continuous_v2/.venv/bin/python analyze.py
../continuous_v2/.venv/bin/python analyze.py --adaptive
../continuous_v2/.venv/bin/python analyze_audits.py
../continuous_v2/.venv/bin/python analyze_one_step.py
../continuous_v2/.venv/bin/python same_evidence.py
```

`analyze.py --adaptive`는 refinement 또는 higher-budget audit가 미완료이면 중단한다. 지금의 partial results를 final stabilized-policy 결과로 표시하지 않는다. Local mirror 동기화 후 `make_report.py`가 최종 보고서를 생성한다. 현재까지의 결과는 별도의 `physics_scene_belief_tempered_bridge_v6_interim_report.md`로 보존한다.

2026-11-10 23:02–23:09 KST 갱신: 외부 training PIDs가 모두 종료되고 idle 상태가 반복 확인돼 같은 두 allocation에서 guard를 켜고 재개했다. 위의 22:45 중단 상태와 20/57 완료 수는 그 시점의 checkpoint 기록이다. 현재 상태는 `status.py --gpu`와 `resource_state.json`을 기준으로 확인한다.
