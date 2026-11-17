# V42 재현과 계산 비용의 범위

서버 프로젝트 내 작업 경로는 `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/height_offset_v42`다. 상위 프로젝트의 기존 v18~v41 코드와 `continuous_v2/.venv` 환경, source model을 이용한다. 2026년 11월 16일 10:21:29 KST에 target 결과 경로가 아직 없는 것을 확인하고 사전 계획·물리식·selector·분석 및 GPU 실행 파일 13개의 SHA-256을 봉인했다. [Manifest](freeze42.sha256) · [서버 UTC](frozen_utc.txt)

## 실행

1개 GPU에 대한 Slurm allocation에서 사용자가 보유한 job ID를 지정한다. Run은 먼저 모든 frozen file을 확인하고 GPU의 사용량과 compute process를 검사한다. Guard는 같은 GPU에 새 compute process가 나타나면 이번 worker만 중단한다. 모든 결과는 worker에서 파일별 SHA-256을 기록하며, 중간에 재시작할 때 이전 archive의 fingerprint와 실제 바이트가 일치하는지 확인한다.

```bash
sha256sum -c freeze42.sha256
srun --overlap --jobid="$GPU_ALLOCATION" -N1 -n1 -c2 bash run42.sh
```

물리적 offset은 query event의 실제 stick/slip에만 작용한다. 관측과 source hypotheses의 예측 높이는 원래 명목 19 또는 31 mm이고, 동일한 prediction을 사전 지정한 9개 offset에 각각 채점한다. 각 panel에 1개 start를 주어 12개 panel에서 부호가 6개씩 균형을 이룬다. [실험 코드](runner42.py) · [Hidden height](world42.py) · [Selector](selector42.py)

## Archive

| 파일 | 내용 |
|---|---|
| `results/<shape>_<height>_<contour>_<law>/catalog_NN.npz` | 모든 offset의 이진 `labels` [offset, sign, regime, environment, object], `p_base` [decoder, sign, environment, object, noise], `choice` [decoder, sign, regime, environment, noise], `risk` [offset, decoder, policy, sign, regime, environment]. 첫 세 probe 측정, 네 번째 측정, direction·signed cue·optimizer weight, 실제 sensor 선택, source bank에 쓰인 두 alignment parameter도 함께 저장했다. |
| `results/<cell>/catalog_NN.json` | 명목 panel·실제 catalog 식별자, calibration·deployment 및 source bank seed, archive SHA-256 및 dependency fingerprint. |
| `results/directions/panel_NN_<shape>_<height>.npz` | 별도 물체 32개에서 취한 64개 profile, PCA로 얻은 estimated direction 및 simulator의 analytic oracle direction. |
| `results/timing/panel_NN.npz` | Original + zero 및 estimated + zero/fixed/optimizer/selector 5개 조합의 warm·setup·cold 시간과 무작위 실행 순서. 배치별 warm-up 1회 후 4회 측정했다. |
| `results/run.json`, `results/completion.json` | GPU·software·사용된 dependency의 SHA-256 목록과 전체 완료 상태. |

## GPU p90

Warm p90은 각 방법에서 동기화된 실제 8개 환경 × 128개 물체의 single-noise inference 시간이다. Host에 놓인 factual array에서 K256으로 sensor를 고르고, 구입한 view 또는 probe를 GPU로 복사해 16,384개 물리 가설의 posterior를 update하고, 해당 방법이 사용하는 calibration 및 correction을 마친 뒤 CUDA를 동기화했다. Selector는 네 번째 측정을 이용한 v38 optimizer의 추정과 세 action 중 최종 선택을 모두 실행했다. Static zero에는 두 계산을 모두 생략했다. 같은 환경에서 observation이 128개 모여야 하는 시간은 제외했다.

Cold p90은 매번 명목 높이에 맞춰 8개 source bank의 query template과 사건을 다시 계산하는 setup도 포함한다. Estimated는 이때 메모리에 올라온 실제 32쌍을 대상으로 CPU SVD를 수행한다. Source bank 초기 구축, 센서의 실물 접근·촬영·접촉·reset, 64개 profile 확보는 두 timing 모두의 범위 밖이다. Offline Brier를 위한 8개 noise 반복을 1회 inference의 GPU p90에 포함하지 않았다. 모든 방법에 4회 calibration을 동일 예산으로 제공했다. 이번 구현에서 항상 zero와 fixed는 처음 세 probe만 읽어도 동일한 출력을 낸다.

## 독립 검산

각 cell에서 128개 held-out 물체 및 noise별 예측을 보정량과 choice로 다시 구성하고, binary Brier를 `mean(p²) − 2 mean(pY) + mean(Y)`로 독립 계산한다. 같은 물체의 변형을 matched pair로 잡아 nominal height의 label 일치, oracle projection의 contour invariance, probe branch의 decoder invariance를 검사한다. 매 timing의 random permutation과 `cold = warm + setup`, 실측 output과 저장된 prediction의 수치 일치도 확인한다. [Archive 검산기](audit_runner42.py) · [물리식 검산](physics_audit42.py)

```bash
export CUDA_VISIBLE_DEVICES='' BELIEF_DEVICE=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
../continuous_v2/.venv/bin/python audit_runner42.py
```
