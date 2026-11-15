## 프로토콜과 자료를 재검산하는 방법

서버 경로는 `datacenter3:/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/probe_transfer_v39`다. 예측표와 정의·코드 8개의 SHA-256을 남기기 직전 서버 결과 디렉터리가 없음을 검사했다. SHA-256 목록 자체의 digest는 `be0a1fbbcec10f8117ef2488be64229b74addda7ec855f3ec8694b2d47b91b52`이고 동결 시각은 **2026-11-15 08:56:40 UTC (17:56:40 KST)**다. Worker는 해당 목록과 기존 v38 dependency hash까지 검사한 후 실행됐다. [계획](PLAN_KO.md) · [동결 기록](freeze39.sha256) · [시각](frozen_utc.txt) · [예측 계수](forecast39.json)

주 실험에는 NVIDIA RTX PRO 6000 Blackwell Server Edition 한 장을 사용했다. Slurm allocation 2744468의 node55에서 process와 GPU memory·utilization을 확인한 후 추론을 실행했다. 이미 다른 작업이 돌아가던 allocation은 그대로 유지했다. 독립 통계·물리 검산은 node56의 allocation 2744471에서 CUDA를 끈 CPU로 수행했다. 각 block은 별도 NPZ와 hash를 남긴다. [실행 metadata](results/run.json) · [96개 block 완료 기록](results/completion.json) · [주 worker 종료](monitor/results/worker_exit.json) · [보조 geometry worker 종료](monitor_geometry/results/worker_exit.json)

같은 부모 프로젝트의 빈 할당 GPU에서 실행하는 명령은 아래와 같다. 원본 자료는 보존하고, 새로 실행할 때는 생성물이 없는 별도 작업 사본을 사용한다. 기존 완료 block의 경우 hash가 일치할 때만 재사용한다.

```bash
sha256sum -c freeze39.sha256
srun --overlap --jobid="$GPU_ALLOCATION" -n1 bash run39.sh
```

서버의 기존 torch 환경을 이용해 CPU에서 주 표와 그림을 다시 만들 수 있다.

```bash
export CUDA_VISIBLE_DEVICES='' BELIEF_DEVICE=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
py=../continuous_v2/.venv/bin/python
"$py" stats39.py
"$py" tables39.py
"$py" figure39.py
```

[추론과 타이밍](run39.py) · [Target 생성 방정식](PHYSICS_KO.md) · [정책과 정보 범위](POLICY_KO.md) · [독립 Brier 검산 코드](audit39.py) · [물리 및 표본 검산 코드](audit_physics39.py) · [보조 geometry 대조](geometry_control39.py)
