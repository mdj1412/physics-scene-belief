# v57 · 독립 surface, pilot group, contact label 생성

## Surface와 deployment

24개 catalog마다 재질 배합과 공간적 변화가 다른 2개 surface instance(braid, seam)를 구성했다. 이후 기존 두 rate law를 각각 적용해 96개 environment를 만들었다. 각 environment의 deployment object 128개, 교체 전 calibration support 128개, 교체 후 calibration support 128개는 공통 지지 집합에서 재사용하지 않았다. 36,864개의 6차원 physical parameter vector를 byte 단위로 대조했고, v50·v55·v56에 쓰인 86,016개 및 세 v57 집합 사이의 중복은 0이었다. 이전 합성 family에서 새 물체와 표면을 추출하였다. 새로운 contact solver 종류나 실제 물리 장비의 검증은 수행하지 않았다. [생성 계획](WORLD_PLAN_KO.md) · [표본과 SHA-256](world_audit57.json)

Catalog key 2000–2023을 v55의 frozen surface field, geometry, instrument 및 forecast generator에 전달했다. 각 catalog에서는 네 contour family와 두 높이의 균형 배정을 유지했다. 같은 12,288개 물체에서 deformation 방향을 보존하거나 반전해 PyBullet fine과 ultra로 residual tangential speed를 얻었다. 양성 label은 0.020 m/s 이하의 stick event다. Fine에서 active 물체 6,144개 중 2,180개(35.48%)의 label이 반전 intervention에 따라 바뀌었고 inactive에서 바뀐 것은 없었다. Fine/ultra의 24,576개 binary endpoint 중 778개(3.17%)는 값이 달랐다. 해상도 변경은 동일 solver를 사용한다. Deployment draw는 seed 57041971, 57045813으로 각 environment의 새 128-object empirical pool에서 128개씩 복원추출했다. 동일 물체의 event·measurement를 정책에 짝지었으므로 반복 수를 독립 표본 수로 세지 않는다. [코드](generate57.py) · [감사 기록](world_audit57.json)

## 어떤 정보가 같은 session을 표시하는가

각 material·regime·protocol에서 4,096개 pilot episode를 생성했다. 해당 material의 경우마다 8개 서로 다른 sacrificial specimen을 새로 생성하고 design·regime 사이에 paired counterfactual로 재사용했다. Specimen마다 shape family, height, deformation magnitude, direction 및 두 sensor sample의 noise coin을 별도 생성한다. 이 pilot specimen은 analytically simulated instrument draw로, 위 36,864개 physics object와 다르다. Reference magnitude에는 bias·noise 및 0.12 mm dead zone이 있고 impact는 inactive에서 0, active에서 해당 변형 부호를 돌려주는 이상적인 방향 계측이다. 첫 reference 이후 reset이 물체의 변형 특성을 바꾸지 않는다는 가정이 필요하다.

8×1 설계는 각 slot에서 e를 독립적으로 뽑는다. 4×2 설계는 (0,1), (2,3), (4,5), (6,7)로 묶인 두 slot에서 첫 번째 specimen과 같은 e를 사용한다. 따라서 같은 session에서 공유되는 것은 e이고, 두 번째 specimen의 active 여부, 방향, flip coin은 독립이다. 두 protocol에서 각 pair의 첫 slot의 e, 여덟 specimen의 방향과 계측 noise 및 mismatch coin을 공유해 paired comparison의 분산을 줄였다. Pilot과 deployment는 동일한 regime의 조성만 공유한다. 실제 regime ID와 e는 acquisition에 제공하지 않았고 첫 policy 결정의 입력은 signed reference/impact, group 위치뿐이었다. 4096개 반복은 같은 material의 시뮬레이션 정밀도이며 material 수를 늘리지 않는다. [Pilot 생성](generate57.py)

## 관측 직접 검산

각 mode·regime별 분모는 196,608개 episode, 1,572,864 paid attempts다. 판독 가능한 pair는 785,860개(49.96%)였다. 인접 두 물체 모두 판독된 경우는 195,969개였고, 62,196개 episode(31.63%)에는 두 물체 모두 판독된 group이 없었다. 유효 reference의 부호를 latent sign과 독립 감사한 결과 오류가 0개였다. 아래의 double mismatch는 두 물체가 모두 판독된 group에서 두 물체 모두 reference와 impact 부호가 어긋난 비율이다. 8×1 행의 인접 두 slot은 서로 다른 session이다.

| Pilot 설계 | 환경 | Single mismatch | Double mismatch | Double mismatch 이론값 |
|---|---|---:|---:|---:|
| 8×1 | A | 28.93% | 8.30% | 8.41% |
| 8×1 | B | 28.94% | 8.40% | 8.41% |
| 4×2 | A | 28.92% | 14.34% | 14.50% |
| 4×2 | B | 28.92% | 9.22% | 9.25% |

A의 e=0/.25/.5 구성비는 42/0/58%, B는 0/84/16%다. 두 경우 E[e]=.29이므로 8×1에서 double mismatch의 이론값은 .29²이다. 4×2에서는 E[e²]이고, 이 값이 0.145 및 0.0925로 차이 난다. Surface가 주어졌다고 해서 모든 pilot에서 이 차이를 관찰할 수 있는 것은 아니다. [원시 빈도 및 검산](world_audit57.json)

## 독립 readout error 민감도

Active 물체마다 impact 부호가 독립적으로 뒤집힐 확률을 r로 주고, error-free data에서 정한 4×2 MAP 규칙은 동결했다. 완전 판독 그룹에서 조건부 불일치 확률이 e에서 r+(1−2r)e로 바뀐다. 이 확률의 첫째·둘째 moment로 group 관측들의 분포를 계산하고 네 group의 모든 outcome을 정확히 합산했다. 표의 mean은 이 계산된 route 확률과 새 24개 catalog의 paired Brier를 조합한 expectation이다. Primary에서 실제로 추출한 finite pilot sample과 수치가 약간 다를 수 있다.

| Independent impact sign error | A에서 fixed I | B에서 fixed I | Balanced accuracy | Mean Brier | Gate 대비 차이 (×10⁻⁶) |
|---:|---:|---:|---:|---:|---:|
| 0% | 44.98% | 36.51% | 54.23% | 0.158351472 | +4.36 |
| 5% | 42.16% | 35.40% | 53.38% | 0.158351662 | +4.55 |
| 10% | 39.76% | 34.51% | 52.63% | 0.158351839 | +4.72 |

같은 센서에서 sign error가 커질수록 두 regime의 2개 물체 패턴에 대한 차이가 줄어든다는 model sensitivity다. 실제 impact sign을 관측할 수 있는 hardware, 측정 시 두 물체의 신호 오차 상관, reset 이후 surface state는 아직 검증 대상이다. [민감도 코드](noise57.py) · [CI 및 집계](pilot_noise57.json)

## 재현

서버 경로는 `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/mismatch_grouping_v57`다. 각 Slurm worker가 해당 디렉터리에서 CPU-only `generate57.py prepare`, `bash run_refs57.sh solver`를 실행하고, idle GPU를 확인한 뒤 `bash run_generate57.sh`로 inference를 실행했다. Solver 및 forecast 종료 후 CPU-only `bash run_refs57.sh panel`로 outcome 결합과 전체 감사를 수행한다. Node55에서 Slurm allocation 2744468의 GPU memory·utilization 0, compute PID 없음 확인 후 guard를 실행했다. Forecast worker 종료 코드는 0이고 foreign GPU process는 0이었다. 최초 source freeze를 남긴 채 solver 및 forecast 실행 전 checksum helper의 오타를 수정했다. Generator의 physical·pilot 샘플링 식에는 수정이 없었다. [최종 freeze](generation_freeze57.sha256) · [보호 로그](generation_guard/results/worker_exit.json)
