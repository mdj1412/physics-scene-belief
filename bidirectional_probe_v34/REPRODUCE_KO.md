# v34 재현 순서
동일한 프로젝트의 이전 v25–v33 모듈·모델을 사용한다. GPU를 쓰기 전 Slurm allocation, device memory, compute PID를 확인한다. 실행 중인 사용자의 다른 GPU·프로세스는 건드리지 않는다. 기존 archive의 덮어쓰기를 피하려면 새 사본에서 실행한다. CPU 작업은 login node가 아닌 Slurm compute node에서 BLAS thread 1, CUDA 비활성화로 실행했다.

1. [Locked schedules·seeds](policy_design34.json), [기본 비교](PROTOCOL_POLICY_KO.md), [방향 판독 조건](DIAG_PLAN_KO.md), [downstream 조건](DECISION_PLAN_KO.md)을 먼저 확인한다. Coulomb source estimator는 주어진 incoming motion의 부호는 알지만 target orientation은 알 수 없다.
2. 빈 GPU 하나에서 `run_policy34.sh`로 기본 정책을 실행한다. 모든 단계에서 동결한 count·action과 sensor prediction을 대조한 뒤 `results_policy/completion.json`이 생성된다. [GPU 실행 코드](run34.py) · [signed likelihood](detector34.py) · [물리식 audit](verify_interventions34.py).
3. 방향 판독은 독립 CPU corpus에서 실행한다. [계획](DIAG_PLAN_KO.md)과 [진단 결과](DIAG_RESULTS_KO.md)에 나열한 source splits, seeds, models를 사용한다. [실행 가능한 NumPy reader](diag34_model.py), [model lock](diag_models.json). Fit 시에만 simulator의 orientation label이 필요하다.
4. 기본 정책 완료 후 compute node CPU에서 `decisions34.py`를 실행해 `direct_models34.json`과 `decision_stats34.json`을 생성한다. 마지막으로 빈 GPU에서 `run_head_time34.sh`로 estimated head 및 두 baseline을 무작위 interleaving해 timing한다. 같은 hardware에서 직접 실행한 p90만 보고한다. `stats_policy34.py`, `stats_head_timing34.py`는 두 timing을 요약한다.
5. `terminal34.py`, `selector34.py`, `external34.py`, `balance34.py`로 선택 단계, 확률 단계, 별도 물체 집단 및 상대 방향 진단을 재현한다. [서버에서 검증한 선택 코드](direct_head34.py)와 `check_head34.py`가 실제 GPU 결정과 CPU replay의 원확률을 비교한다. `fig34.py`와 `write34.py`는 보고서용 그림과 표를 생성한다. 원자료까지 같은 사본을 사용한 경우 GPU가 없어도 Brier와 통계는 재생할 수 있다.

탐색적 원 decoder 대조를 재현하려면 동일한 CPU 환경에서 [anchored34.py](anchored34.py)를 실행한다.
