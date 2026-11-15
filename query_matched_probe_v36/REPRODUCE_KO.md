# v36 실행과 raw archive 검증

서버 프로젝트는 `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/query_matched_probe_v36`이다. `../continuous_v2/.venv`와 v25–v35 모듈을 사용했다. 환경별 `fixed 4`는 서로 다른 네 calibration objects에서 각각 한 번씩 접촉한다. 새 deployment objects 수는 한 환경당 128개다.

## 실행

```bash
cd /scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/query_matched_probe_v36
srun --overlap --jobid="$ALLOCATION" -n1 bash ./run36.sh
```

`ALLOCATION`에 할당된 빈 GPU 한 장의 Slurm 번호를 넣는다. `run36.sh`는 GPU 수, memory, utilization, active PID를 확인한다. `guard.py`는 다른 GPU 작업이 나타나면 이번 worker만 중단한다. 재시작 시 `meta.json`에 기록한 block의 파일 checksum과 실행 파일 fingerprint를 확인하고 이어간다. 최초 실행에 필요한 source gate는 `gate36_frozen.npz`이며 `results/run.json`에 file hash가 있다.

## Target와 실행 경로

- `target36.py`는 v35의 두 contour와 두 속도별 friction response를 쓴다. Localized 조건에서 height가 0인 far contact는 모두 같은 directional response를 갖는다. Height가 0.023 m 이상이면 strong/weak/null의 amplitude multiplier는 각각 1, 0.3, 0이다. 두 지점 사이에서는 smoothstep으로 연결했다. Global-null은 모든 위치에서 multiplier=0이다. 다른 latent geometry와 noise는 같은 block에서 공유한다.
- Conventional schedule은 세 번까지 alternating이며 네 번째 conventional action은 해당 schedule을 이어간다. Matched schedule에서는 동일한 네 번째 물체에 0.023 m, +0.50으로 접촉한다. Deployment query는 0.025 m, +0.425다. 네 번째 접촉에서 관측하는 양은 noisy 2D outgoing velocity다. Calibration과 deployment 단계는 물체와 noise를 별도로 뽑는다.
- `detector36.py`는 v34 detector의 동일한 8,192 source particles에 matched action의 Coulomb response를 계산해 likelihood에 반영한다. 다른 세 개의 likelihood는 v34 signed detector를 사용한다. 각 environment에서 deployment probe는 128개 중 13개이며 나머지는 view를 얻는다. 두 schedule은 이 13개의 물체를 서로 다르게 고를 수 있다. 같은 schedule 내 method는 동일한 action과 observation을 받는다. Source decoder는 공통 population coupling을 알고 있다.
- `run36.py`는 paid prefix를 완전하게 처리한 뒤 original physics prediction을 계산한다. Far-v35는 4개 conventional inputs에만 v34 frozen four-probe reader와 β를 적용한다. Matched methods는 source-derived first-three proposal z와 기존 β를 쓴다. Near-always는 모든 환경에 적용하고 near-gate3/4는 expected gain이 양수인 환경에 적용한다. Near-gate3는 네 번째 near reading을 acquisition에는 사용하고 terminal gate에서는 가린다.
- Raw archive마다 `p_far, p_near`는 각 schedule의 original probability [8 environments, 128 deployment objects, 32 sensor draws], `labels`는 stick event [8,128]이다. `gate36.py`에 들어가는 공개 입력은 `far_x, near_y, dirs`에서 slice한다. `far_x`는 두 schedule의 네 factual row이며 near reading은 `near_y[:,3]`이다. `dirs` 네 번째 열은 conventional schedule의 부호로, matched action의 속도는 별도 protocol에서 +0.50으로 지정한다. `theta4`, `matched_clean`, `matched_stick`, `theta`는 평가 전용이다. `risk`는 6 methods × 8 environments이며 deployment object 및 noise draws 평균이다.
- Timing은 8개 사전 지정 family/seed clusters에서 각 방법 warmup 1회, 3회 timed repeats로 측정한다. Strong/weak/local-null/global-null이 각각 2개씩 포함된다. `full_ms`는 3 × 6 methods이고 `timing_order`는 각 반복의 무작위 실행 순서다. 모든 method에서 매번 factual likelihood, 네 번의 posterior update, acquisition, 관측 decoding, 해당 gate/보정을 포함해 1,024개 객체의 확률을 계산한다. 양끝에서 CUDA를 동기화하고 host 결과 반환까지 측정했다. Detector·bank와 gate artifact 초기화, simulator의 sensor 생성, 실물 probe의 작동·sensor/reset 대기는 제외했다. 동일 GPU session의 p90이다.
- Timing 시 선택한 action이 batch 평가와 같은지, 첫 sensor realization의 returned probability가 raw 데이터와 absolute tolerance 1e-8 이내인지 호출마다 확인한다. Far-Y/XX/deployment observations가 세 local regimes에서 일치하는 것도 실행 중 검사한다. 아카이브는 4 object×surface cells × 8 seed blocks × 7 signed scenarios이고 파일별 `sha256`을 각 `meta.json`에 기록한다.

## Raw 결과 검산

GPU worker가 정상 종료하고 32개 block의 `meta.json` 및 `results/completion.json`이 생성된 뒤 다음 명령을 실행한다. NumPy와 SciPy가 설치된 동일한 project environment를 사용한다.

```bash
python stats36.py
python increment36.py
python supplement36.py
python figure36.py
python findings36.py
python report36.py
```

`stats36.py`는 224개 archive의 checksum, Brier, contact labels, matched response, 같은 far evidence를 재계산한다. `increment36.py`와 `supplement36.py`는 frozen source artifact로 factual-only 및 conventional-fourth 대조를 복원한다. `stats36.json`에는 regime별 raw loss, simultaneous interval, GPU timing, source prior shrinkage, privileged sensor accuracy, 배포 regime의 빈도를 바꾼 계산을 저장한다. `figure36.py`의 입력은 이 JSON 하나다.
