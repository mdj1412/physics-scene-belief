## 서버 실행과 재현

작업 경로는 `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/probe_speed_v37`이다. 이전 simulator·GPU 환경과 v36 frozen source gate를 사용한다. 다음 명령의 `ALLOCATION`은 본인에게 할당된 빈 GPU 한 장의 Slurm job ID다.

```bash
srun --overlap --jobid="$ALLOCATION" -n1 bash run37.sh
```

서버 runner는 시작 직전 GPU process, memory, utilization을 확인한다. Guard는 실행 중 다른 GPU process가 나타나면 해당 v37 worker를 중단한다. `monitor/results/worker_exit.json`에 종료 코드와 관찰한 외부 process가 기록된다. 완료된 block을 재사용할 때 그 block의 SHA-256과 source fingerprint를 검사한다. 

독립적인 32개 seed block을 4개 object–surface 조합에 균등 배정했다. 여덟 환경마다 네 개의 서로 다른 calibration object를 쓰고, 별도 deployment object 128개에 대해 sensor noise 32회를 평가한다. Local strong/weak/null의 두 orientation과 global-null(방향 없음)을 같은 block에 둔다. 각 세 local regime에서 첫 세 관측과 fourth probe의 latent geometry, random noise를 공유한다. 조건당 deployment probe는 환경별 128개 중 13개이고, 각 speed에서 획득하는 observation은 speed에 맞게 갱신한 Coulomb likelihood와 기존 K256 acquisition이 결정한다. 동일 speed의 두 정책은 보정 적용 방식만 달리한다.

세 speed의 raw archive는 한 파일에 저장된다. `speeds=[0.50,0.75,1.10]`와 `methods=[original,always,shrink,gate]`의 순서를 쓴다. `p_base[speed,environment,world,noise]`는 보정 전 probability, `risk[speed,method,environment]`는 held-out Brier, `fourth_y`는 noisy response, `matched_clean`, `matched_stick`, `theta4`는 평가용 simulator state다. Decision은 공통 prefix의 `z`와 해당 speed의 `accept4`, `score4`, `near_strength4`에 기록한다. `dirs`는 첫 세 conventional actions의 부호다. 마지막 action은 23 mm에서 실행한 양의 명령 speed다.

각 timing call은 여덟 환경의 factual likelihood와 네 번의 calibration posterior update, 1,024개 deployment objects의 acquisition, 선택된 actual sensor의 physics decoding, CPU correction과 host 반환을 전부 포함한다. 원래 source bank와 speed별 model을 캐시한 뒤 CUDA synchronize로 양 끝을 둘러싼 wall time을 측정했다. 고정 shrinkage는 첫 세 측정으로 signed proposal을 추정하고, gate는 여기에 네 번째 paid response의 source likelihood와 expected gain을 더 계산한다. 시뮬레이션에 쓰는 data generation, 센서 대기와 실제 장치 운동, one-time bank 생성 비용은 timing에서 제외한다. 각 방법·속도는 8개의 paired timing clusters에서 각 1회 warmup 후 3회씩 무작위 순서로 측정한다. 모든 timed call의 선택 action과 첫 sensor realization에 대한 예측을 원래 archive와 검사한다.

`audit37.py`는 source hash, seed, paid objects, near-noise pairing, local regimes 간 공통 정보, deployment quota 및 timing 순서를 검증한다. 실행 시 완성된 `results/completion.json`과 224개 archives가 필요하다. `results/<cell>/block_NNN/meta.json`의 SHA-256으로 각 파일을 검증할 수 있다.

## 통계 재계산

GPU worker의 완료 기록이 생긴 뒤 같은 project environment에서 분석을 실행한다. `fixed_validation37.py`는 v37 입력을 읽지 않고 v36 자료의 고정 보정량을 계산한다. `stats37.py`는 source hash와 모든 raw checksum을 대조하고 각 속도의 물리 상태와 Brier를 재생한다. `routing37.py`, `binary37.py`, `slip37.py`, `physics_factor37.py`는 각각 표면별 선택, 두 후보 판별과 세 행동 대조, slip 집단별 Brier 및 물리 요인 분해를 계산한다. `figure37.py`는 `stats37.json`에서 보고서 그림을 생성한다.

```bash
python fixed_validation37.py
python stats37.py
python routing37.py
python binary37.py
python slip37.py
python physics_factor37.py
python figure37.py
```
