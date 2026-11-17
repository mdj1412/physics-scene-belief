# v43 GPU 실행 및 재현

서버의 기존 `physics-scene-belief` 환경에서 `blind_decoder_v43`를 작업 폴더로 사용한다. `continuous_v2/.venv`와 v38/v41 source 자료·동결 정책·Sobol bank의 코드가 같은 프로젝트에 필요하다.

1. `source_data43.py`와 `source_fit43.py`로 source 자료와 모델을 만들거나 동결된 `frozen_model43.json`을 사용한다. `sha256sum -c freeze43.sha256`으로 target 실행 코드를 검사한다. 서버의 `frozen43_utc.txt`는 첫 target 실행보다 앞서야 한다.
2. 같은 사양의 GPU 두 장에서 Slurm allocation을 확인하고 각각 `bash run43.sh even`, `bash run43.sh odd`로 서로 겹치지 않는 catalog를 평가한다. GPU가 한 장이면 같은 명령을 순서대로 실행한다. Worker는 다른 compute PID가 감지되면 자신의 프로세스 그룹만 중단한다. 이미 완료한 archive는 fingerprint와 checksum이 모두 일치할 때 재사용한다.
3. 두 분할이 모두 끝나면 한 GPU에서 `bash run43.sh timing`으로 12개 catalog에서 12개 정책을 각 1회 warm-up하고 3회씩 순서를 섞어 측정한다. 한 batch는 4개 환경 × 128개 물체다. 측정값 단위는 ms다. 각 열은 고정된 정책 순서, `order`에는 시간순으로 실행한 열 번호가 들어간다. 최종 p90은 정책별 36회 값의 percentile이다.
4. CPU에서 `../continuous_v2/.venv/bin/python stats43.py`와 `../continuous_v2/.venv/bin/python audit_gpu43.py`로 주결과와 checksum·blind sensor 검산을 재현한다. Raw는 `results/panel_NN/` 및 `results/timing/`에 있다. 한 archive의 `pred` 축은 decoder·correction·환경·물체·sensor noise, `labels` 축은 signed height offset·환경·물체다. `xx`는 factual 관측이며 `query_paid`/`probe_paid`의 미구입 위치는 NaN이다. Simulator 정보는 scoring/replay용 metadata에만 있다.
5. Warm p90은 neutral K256 acquisition, paid sensor routing, 현재 physics posterior, 필요한 robust residual 또는 projection, 정해진 correction, CPU–GPU 전송 및 CUDA 동기화를 포함한다. Cold에는 메모리에 있는 bank를 height·direction에 맞춰 template을 구성하는 시간까지 더한다. Offline bank·robust model 생성, 장치의 물리적 촬영·접촉·reset, GPU process 부팅은 제외한다. Robust residual과 calibration의 CPU 계산도 측정 시간에 포함된다.
