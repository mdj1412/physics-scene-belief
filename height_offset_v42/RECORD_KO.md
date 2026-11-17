## 서버 실행과 검산

2026년 11월 16일 `datacenter3`의 Slurm allocation 2744468에서 사용 직전 compute PID 0개, GPU memory 0 MiB를 확인한 NVIDIA RTX PRO 6000 Blackwell Server Edition 한 장으로 수행했다. PyTorch 2.7.1+cu128, CUDA 12.8를 사용했다. 주 계획·실행 코드·source selector 계수는 10:21:29 KST에 서버에서 SHA-256으로 동결했다. [동결 목록](freeze42.sha256) · [UTC 시각](frozen_utc.txt) · [실행 기록](results/completion.json)

세 agent가 catalog와 독립 물리 구현, source selector 학습·검산, GPU 실행·timing 검산을 분담했다. 288개 원자료와 12개 timing archive의 checksum을 확인했다. 서로 다른 물체 추출은 12,288회이며 복원 추출을 감안한 고유한 deployment prototype은 11,924개, calibration에서 실제 probe를 받은 고유 prototype은 384개였다. 방향 추정에는 12 panel×2 shape×2 height×32개 별도 물체의 paired profile, 총 3,072회 readout이 필요했다.

독립 물리 식으로 15,925,248개의 저장 stick label을 재생했다. 1,492,992개 환경별 Brier의 최대 재계산 오차는 8.33e-16, GPU timing 경로와 archive의 확률 차이는 최대 4.44e-16였다. 모든 contour의 oracle projection 확률 간 최대 차이는 1.05e-13, probe 가지에서 decoder 간 최대 차이는 0.00e+00였다. Brier 변화의 두 event 전환 항을 더한 값과 실제 차이의 최대 오차는 1.53e-16였다. [물리 원자료 검산](physics_audit42.json) · [수치 검산](audit_runner42.json)

서버 폴더는 `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/height_offset_v42`다. 새 사본에서 기존 v18–v41 dependencies를 사용할 수 있게 한 뒤 GPU에서는 `run42.sh`를 실행한다. Raw archive는 `results/<shape>_<height>_<variant>_<law>/catalog_NN.npz`에, GPU 측정 원값은 `results/timing/`에 있다. [재현 안내](REPRODUCE_KO.md) · [GPU runner](runner42.py) · [물리 생성식](world42.py)

표와 그림을 다시 만들 때는 완료된 GPU 원자료에서 서버의 기존 Python 환경을 사용한다. 아래 CPU 단계는 과거 학습 모델이나 GPU 결과를 바꾸지 않는다.

```bash
export BELIEF_DEVICE=cpu CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
py=../continuous_v2/.venv/bin/python
"$py" stats42.py
"$py" reveal42.py
"$py" soft42.py
"$py" dense42.py
"$py" dense_table42.py
"$py" mix42.py
"$py" display42.py
"$py" record42.py
"$py" report42.py
```
