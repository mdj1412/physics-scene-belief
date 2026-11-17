# Physics Scene Belief

관측과 접촉 실험으로 새로운 물체의 충돌 결과를 예측하는 연구다. v40에서는 같은 local tangent를 가진 contour와 두 unseen contact law에서 v38 optimizer와 고정 32.5% shrinkage의 held-out Brier loss 및 GPU p90을 비교했다. [v40 보고서](tangent_transfer_v40/REPORT_KO.md) · [실행 전 예측표](tangent_transfer_v40/PREDICTIONS_KO.md) · [재현 방법](tangent_transfer_v40/REPRODUCE_KO.md)

v74에서는 빠른 anti-spin 충돌에서 elastic impulse가 이탈 방향을 잘못 예측하는 이유와 4회·8회 calibration의 효과를 분석했다. [v74 보고서](racket_release_v74/REPORT_KO.md) · [원인 분해](racket_release_v74/mechanism/MECHANISM_KO.md) · [v73 비교 실험](racket_video_v73/REPORT_KO.md)

Git에는 v74까지의 연구 코드와 보고서, 선별한 집계 결과와 재현 자료를 보관한다. 개별 시행의 원자료는 실험 서버의 각 버전 폴더에 있다. v40 원자료 구조와 생성 명령은 [재현 방법](tangent_transfer_v40/REPRODUCE_KO.md)에 정리했다.

## 첫 번째 PyBullet 실험

PyBullet 기반의 통제된 physics 실험이다. 같은 candidate world, observation likelihood, noise, acquisition budget에서 `plug-in geometry`와 `joint scene belief`를 비교한다. 설명과 실제 결과는 `REPORT.ko.md`에 정리했다.

원격 작업 경로:

```text
/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief
```

## 실행 환경

- 실행 host: `node55`, 접속 경로: `datacenter3` → `yonsei-ai-login3` → Slurm allocation.
- Python 3.9.18, PyBullet 3.2.7, NumPy 1.26.4. 전체 dependency는 `requirements.lock.txt`에 있다.
- `CUDA_VISIBLE_DEVICES=""`, CPU 1개, BLAS/OpenMP thread 1개.
- GPU 사용, neural model training, learned model inference는 수행하지 않았다.
- 기존 사용자 job을 취소하거나 변경하지 않았다. 기존 allocation 안에서 짧은 CPU job step만 실행했다.

## 재현

이미 활성화된, 사용자 소유의 적절한 Slurm allocation을 확인한 뒤 login node에서 실행한다. 아래 job ID는 실제 실행에 사용한 ID이며 만료 후 재사용할 수 없다.

```bash
cd /scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief
srun --jobid=2642474 --overlap --exact -N1 -n1 -c1 --gres=gpu:1 \
  --time=00:45:00 bash run_cpu.sh
srun --jobid=2642474 --overlap --exact -N1 -n1 -c1 --gres=gpu:1 \
  --time=00:10:00 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python analyze.py
```

`--gres=gpu:1`은 기존 GPU allocation의 step resource 지정이다. 프로그램은 CUDA를 비활성화하고 CPU만 사용한다. 다른 작업과 resource를 공유하는 `--overlap`은 해당 allocation의 현재 workload를 확인한 경우에만 사용한다. 새 환경에서는 프로젝트 전용 `.venv`를 만들고 `requirements.lock.txt`를 설치한다. 기존 Python 환경은 수정하지 않는다.

`run_cpu.sh`를 다시 실행하면 현재 `data/pilot`, `results/pilot`, 주요 log를 갱신한다. 새로운 실험은 각 Python script의 `--out` 옵션을 이용해 다른 directory에 저장한다. `analyze.py`의 input/output 경로는 파일 상단에 있다.

## 파일

| 파일 | 역할 |
|---|---|
| `physics.py` | PyBullet rollout, exact orthographic RGB rendering, RGB centroid extraction, physics 검증 |
| `experiment.py` | finite-grid posterior, plug-in approximation, EVSI/EIG, paired measurement, metric 계산 |
| `test_experiment.py` | query relevance, stop, geometry refitting, proper score, quadrature 검증 |
| `analyze.py` | held-out 집계, scene-family bootstrap, numerical check, figure |
| `data/pilot/worlds.npz` | candidate simulator output와 observation mean |
| `data/pilot/families.json` | 60개 family parameter 및 split |
| `data/pilot/physics_validation.json` | 실제 host, physics runtime, renderer, physics check |
| `results/pilot/config.json` | seed, budget, noise, code hash, data hash |
| `results/pilot/family_metrics.csv` | family·geometry별 평균 metric, 모든 비교 조건 |
| `results/pilot/test_episode_details.csv` | noise 2 px의 주요 세 방법에 대한 test episode 세부 결과 |
| `results/pilot/summary.json` | test aggregate |
| `results/pilot/paired_contrasts.json` | paired bootstrap CI |
| `results/pilot/analysis_checks.json` | negative control 및 quadrature 검증 |
| `results/pilot/runtime.json` | 실제 실행 시간과 method별 시간 |
| `results/pilot/figures` | 검토한 결과 figure |
| `logs` | 실행 log 및 폐기한 perspective camera의 실패 log |

## 해석 범위

정답을 포함하는 네 candidate world와 알려진 Gaussian measurement likelihood를 사용한다. RGB의 단순 color detector 뒤에 centroid noise를 추가한다. Real-video perception, continuous geometry fitting, unknown contact topology, model mismatch, learned acquisition 성능을 검증한 것이 아니다. `plugin_refit`은 관찰 이력 전체에서 geometry MAP을 다시 구하지만 planning과 prediction 시에는 하나의 geometry로 collapse한다. `plugin_fixed`는 초기 geometry를 계속 유지한다. 두 정의를 혼동하지 않는다.

`budget=2`의 EVSI는 greedy myopic policy다. 최적의 non-myopic policy라고 주장하지 않는다. EIG는 정보량이 사실상 0일 때 stop하며, EVSI는 normalized MSE 기준 cost penalty 0.005를 사용한다. EIG의 nats와 MSE penalty를 같은 단위로 취급하지 않는다.
