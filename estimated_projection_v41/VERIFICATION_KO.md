## 실행과 독립 검산

서버 `datacenter3`의 Slurm allocation 2744468에서 시작 직전 compute process 0개, memory 0 MiB인 NVIDIA RTX PRO 6000 Blackwell Server Edition 한 장을 사용했다. 주 실험 576개 block의 dependency fingerprint와 파일별 SHA-256을 검산했다. GPU 환경은 PyTorch 2.7.1+cu128, CUDA 12.8였다. 세 agent가 세계 생성·물리 검산, selector 학습·선택 검산, GPU 실행·결과 검산을 분담했다. [서버 기록](results/completion.json) · [재현 절차](REPRODUCE_KO.md)

Calibration과 배포에 독립 catalog를 사용했다. 12×2×8×128=24,576개 deployment draw에 실제로 서로 다른 prototype은 23,153개였다. 실제 calibration 접촉에 쓰인 서로 다른 prototype은 765개다. 12개 독립 panel 안에서 contact law·형상·높이·시작 방향·noise의 짝 구조를 유지했다. [물리 검산](physics_audit41.json)

별도의 NumPy collision 식으로 3,538,944개 저장 label을 재구성했다. 불일치는 0이었다. 248,832개 환경별 loss 재계산의 최대 절대 오차는 1.67e-16였고, 동일 물체의 세 contour에서 oracle 확률의 최대 차이는 1.66e-13였다. 16,384개 가설의 단일 bank를 점검했을 때, 기존 25 mm event를 그대로 사용하면 실제 19 mm·31 mm에서 각각 295개·180개 가설이 다른 event를 예측했다. 새 높이의 likelihood는 직접 계산한 squared distance와 2.89e-15 이내로 일치했다. [Brier 검산](audit41.json) · [Decoder 검산](decoder_audit41.json)

이미 메모리에 있는 64개 profile에서 estimated direction을 다시 구하는 host 구간은 median 0.153 ms, p90 0.187 ms였다. Estimated template과 event를 새로 만드는 전체 setup의 p90은 7.00 ms였다. 두 값은 12개 panel × 3회 × 3개 정책에 걸친 108회 setup에서 계산했다. 실제 64개 profile의 획득·대응·형상 교체를 포함한 총시간은 별도로 측정해야 한다.

서버 작업 폴더는 `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/estimated_projection_v41`다. 주 결과는 [통계 JSON](stats41.json), [GPU runner](runner41.py), [label-free 추정](world41.py), [frozen selector](selector41.py)에서 다시 확인할 수 있다.
