# v55 · 새 surface·session의 구성과 독립성 검증

## 실험 단위

24개 새 catalog 각각에 독립적인 surface 두 개를 만들고 두 개의 기존 rate law와 교차해 네 environment를 구성했다. `braid`는 서로 다른 주파수의 공간적 진동이 겹친 surface이고, `seam`은 국소적인 material seam 양측의 마찰 변화가 비대칭인 surface다. Surface마다 amplitude, 중심, 폭, 방향 계수와 phase를 새로 뽑았다. 실제 접촉점의 material field가 factual probe 및 다른 solver의 query impact에 들어간다. 높이 0–40 mm의 diagnostic grid에서 48개 field의 friction scale 범위는 0.910–1.070, orientation multiplier 범위는 −0.907–0.828이었다. 방향의 부호는 forward physics에서만 쓰인다. 네 contour family(saddleback, terrace, swale, scallop)와 두 rate law(rational, gaussian_rate)는 v50과 공통이다. 새 물체와 surface를 시험했으며 contour·law family 자체를 바꿔 재시험한 결과는 아니다.

| 대상 | 독립성을 만드는 단위 | 실제 수 |
|---|---|---:|
| Held-out catalog | 독립 material·physical population·instrument seed | 24 |
| Surface | 두 field family 각각 catalog마다 새 parameter | 48 |
| Deployment object | catalog × 네 environment × 128개 | 12,288 |
| 교체 전 sacrificial calibration object | deployment·교체 후와 분리한 support | 12,288 |
| 교체 후 sacrificial calibration object | 나머지 두 집합과 분리한 support | 12,288 |
| Base calibration realization | catalog × environment × 256회 | 24,576 |
| k별 acquisition 입력 | realization마다 다섯 방향 유지율을 공통 noise로 구성 | 122,880 |

Seed namespace 551117–551121로 material, support, object index, sensor noise, deployment 배정을 분리했다. Support ID는 deployment 5500–5523, 교체 후 5600–5623, 교체 전 5700–5723이다. 시뮬레이션에서 (support ID, environment, object index)가 물체를 식별한다. 같은 6개 physical parameter 벡터가 이 세 집합에 겹치는 경우가 0개였고, 기존 v50–54의 12,288개 deployment·calibration 벡터와도 0개였다. Realization의 네 calibration 자리에는 서로 다른 물체를 배정했다. 여러 realization은 같은 surface별 128개 물체 풀에서 다시 뽑히므로, 122,880개 전체를 독립 surface의 수로 해석할 수 없다. [Manifest](independence55.json)

## 같은 물체에서 무엇을 비교했는가

각 environment의 128개 deployment 물체에는 네 contour와 두 접촉 높이의 조합을 16개씩, 상태는 비활성 64개·양 방향 각 32개씩 배정했다. Tip radius, contact offset, surface-alignment perturbation, sensor noise realization도 물체에 배정했다. 각 물체의 factual·구입한 observation으로 frozen v50 GPU decoder의 signed endpoint 두 개를 계산했다. Calibration acquisition을 만드는 모듈은 gate의 선택이나 결과 loss를 읽지 않았다. GPU에서는 512개 factual object에서 기존 13-of-128 physical probe allocation을 수행한 뒤, 높이별로 각 environment의 64개 물체를 모아 해당 template으로 계산했다. 원시 observation 생성은 timing 입력 준비에 속한다.

같은 접촉점과 물체에서 변형 방향만 보존하거나 뒤집어 PyBullet로 endpoint 두 개를 직접 계산했다. Ground-truth positive는 residual tangential speed ≤0.020 m/s인 stick event다. 물체의 binary label과 예측을 동시에 보수로 바꾸면 slip에 대한 Brier와 동일하다. 12,288개 물체 중 2,234개에서 방향을 뒤집으면 binary outcome이 바뀌었다(활성 6,144개의 36.36%). 비활성 6,144개에서는 변화가 없었다. Fine solver(dt 0.0125 ms, mesh 0.025 mm)와 ultra solver(dt 0.00625 ms, mesh 0.0125 mm)는 24,576개 endpoint 중 794개(3.23%)에서 binary label이 달랐다. 이는 같은 engine의 refinement에 대한 sensitivity다. [Physics 진단](physics_audit55.json) · [생성 코드](generate55.py)

Weak/reference displacement는 접촉 위치에서 ±1.5 mm 떨어진 두 gauge 점의 변형, 물체별 gain·bias, instrument noise로 합성했다. 교체 전·후의 sacrificial support와 meter noise를 분리했다. 교체 후 weak/reference의 방향 유지율 k와 reference→impact의 반전 확률 e는 기존 평가와 같이 독립적인 조건으로 조작했다. 구입한 impact의 세 값은 방향 음·양과 inactive에 해당한다. Binary deployment outcome, 네 번째 후보 측정 및 true e는 acquisition 입력에 들어가지 않는다. 이 설계는 새로운 field에서 보정이 transfer되는지 검증한다. 실제 surface의 마찰이 witness에서 나타나는 k·e를 어떻게 유발하는지는 별도 실물 측정이 필요하다.

## 실행 및 재현

원격 node55의 Slurm allocation 2744468에서 CPU로 world·calibration·Bullet 두 해상도를 생성하고, GPU를 별도로 확인해 label-blind prediction을 수행했다. 기존 GPU worker guard가 다른 PID를 감시했고 exit code 0, foreign process 0이었다. [Freeze SHA-256](generation_freeze55.sha256) · [Guard 로그](generation_guard/results/worker_exit.json)

서버 디렉터리 `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/fresh_surface_v55`에서 CPU-only Slurm worker의 `generate55.py prepare`, `generate55.py solver`, guarded GPU의 `bash run_generate55.sh` 다음에 CPU worker의 `generate55.py assemble`, `audit_holdout55.py`를 순서대로 실행할 수 있다. 각 panel archive는 첫 세 쌍을 포함한 raw calibration, 양쪽 잠재 측정, signed forecast와 두 해상도의 Bullet label을 보관한다. 결과 scorer는 이 archive와 고정된 policy를 읽으며, 같은 session의 비교에서는 deployment sample과 noise를 공유한다. 24개 panel의 SHA-256을 연결해 계산한 manifest digest는 `f335d5943956190d8ec3af9dc976c27dd41cfa2bea13018d65292a8f8efbc73d`이다.
