# v56 · 독립 surface, pilot, deployment의 생성과 검산

## 표본과 정보 경계

v55와 같은 braid·seam surface family에서 spatial friction scale과 orientation parameter를 새로 뽑았다. 각 surface는 두 rate law에 걸쳐 같은 physical material과 같은 pilot 결과를 공유한다. 24개 새 catalog × 두 surface가 독립 material realization 48개를 이룬다. 하나의 catalog에서 두 surface의 material을 그리므로 uncertainty의 cluster 단위는 catalog다. 각 material에서 high-mismatch session의 비율 w를 사전 명시한 일곱 값으로 각각 설정했다. 이 값은 synthetic transfer mechanism의 개입 조건이다. 실제 material parameter에서 관측한 불일치율로 해석할 수 없다.

| 샘플 | 표본 또는 대조 | 독립성 점검 |
|---|---:|---|
| Deployment object | 24 × 4 environments × 128 = 12,288개 | 서로 다른 physical parameter vector |
| 교체 전/후 terminal calibration support | 각 12,288개 | Deployment, 두 시점, 이전 v55·v50과 중복 0 |
| Deployment calibration realization | 24 × 4 × 256 = 24,576개 | 다섯 방향 유지 조건에서 같은 noise로 짝지음 |
| Stationary surface의 pilot | w당 48 × 512 = 24,576개 group, group당 16번까지 구매 | Pilot과 deployment는 별도 RNG, 두 law 간에는 같은 pilot을 재사용 |
| Label의 물리적 점검 | 12,288개 물체 × 2개 방향 = 24,576개 endpoint | Fine·ultra PyBullet, 변형 방향을 제외한 동일한 물체·충돌 |

고정된 v55 generator에 physical catalog key 1000–1023을 공급했다. 이 방법은 domain과 instrument code를 바꾸지 않고 seed 및 object support를 분리한다. 36,864개 새 parameter vector 중 과거 v55·v50의 49,152개 vector와 같은 사례는 없었다. 새 catalog의 변형 방향만 뒤집었을 때 활성 deployment 물체 6,144개 중 **2,280개(37.11%)**에서 물리 예측 label이 바뀌었다. 비활성 물체 6,144개의 label은 동일했다. 두 solver resolution에서 같은 endpoint가 다른 stick label을 보인 횟수는 **889/24,576(3.62%)**였다. Positive label은 impact 이후 residual tangential speed ≤0.020 m/s이다. 각 비교는 128개 물체별 두 potential label과 같은 센서 noise를 공유하고, session score는 material·law별 128개 finite pool에서 새 seed로 128개를 복원추출해 계산했다. [생성기](generate56.py) · [Manifest·Physics audit](world_audit56.json)

## 한 쌍을 구입하면 무엇을 알게 되는가

하나의 surface 안에서 pilot 1회는 새 session에서 희생 물체 1개를 뽑아 두 번 접촉하는 것이다. 2.0 N reference 하중 뒤의 signed deformation과 impact 하중 뒤의 방향을 같은 물체·같은 위치에서 읽는다. Reference와 impact 사이에 reset해도 반응이 보존된다는 가정이 있다. 한 번 시도하면 결과가 비활성이라도 2 contacts, 2 resets를 사용하며 대체 물체를 더 뽑지 않는다. Weak preload는 이 두 부호를 직접 비교하는 데 필요하지 않다. 정작 deployment의 네 쌍은 기존 weak/reference 및 weak/impact protocol을 유지한다. Pilot은 서로 다른 calibration session에서 1쌍씩 뽑으므로 e의 분산보다 평균에 민감하다.

각 시도에서 e=.5인 session을 확률 w로, e=0·.25인 session을 각각 (1−w)/2로 정한다. 대안 session의 물체 상태는 비활성 50%, 활성 양 방향 각 25%다. 활성 reference의 부호를 확률 e로 뒤집으면 impact 방향이다. Reference는 geometry와 접촉 높이가 정한 full-load amplitude, session instrument gain·bias, specimen variation 및 σ 0.060 mm read noise를 포함한 displacement에서 부호를 읽는다. 절댓값 0.12 mm 미만은 읽을 수 없는 값 0으로 기록한다. Impact는 active direction일 때 정확한 부호, inactive일 때 0을 반환하는 가상 센서다. Pilot의 가상 물체는 별도 instrument draw로 생성했다. 36,864개 physics vector 집계에 넣지 않았고 pilot마다 독립적인 Bullet collision이나 실물 collision을 수행한 것도 아니다.

각 material·w에 대한 512개 pilot group은 서로 독립이다. 동일 group의 K별 비교에서는 최초 K쌍을 잘라 쓰고, w별 비교에서는 동일 fixture와 noise를 사용해 regime 배당과 mismatch만 바꾼다. 두 law에 대해 같은 pilot을 다시 돈 주고 사지 않는다. K=16에서 w당 393,216회 시도 중 197,095회(50.12%)에 두 부호를 모두 읽을 수 있었다. 이 실현 표본에서 활성 물체의 reference 부호가 잘못된 경우는 0회였다. K=8에서는 24,576개 group 중 100개(0.41%)가 전부 unreadable이어서 기본 gate로 돌아간다. [표본 계획](WORLD_PLAN_KO.md) · [Pilot counts](world_audit56.json)

## Pilot impact 부호의 오독에 대한 추가 실험

별도 sensitivity에서 읽을 수 있는 pilot impact의 부호만 서로 독립적으로 틀리는 상황을 시험했다. 8회 시도의 두 이항 분포를 convolution해 오독 이후의 모든 count를 적분했다. Acquisition 식과 threshold는 그대로 두고, 사전 지정 population(.35/.52 균등)을 평가했다. 이는 hardware에서 측정한 오류율이 아니다. 실물의 false negative나 correlated readout error는 다루지 않는다.

| 활성 pilot impact 오독률 | Fixed impact로 전환 | Switch mean | Switch−항상 gate (×10⁻⁶) | Switch−항상 fixed I (×10⁻⁶) |
|---:|---:|---:|---:|---:|
| 0% | 46.60% | 0.158239094 | −0.328 | −1.479 |
| 5% | 50.22% | 0.158239240 | −0.182 | −1.333 |
| 10% | 53.79% | 0.158239383 | −0.039 | −1.190 |

Primary 범위에서는 mismatch 빈도가 0.5보다 낮아 대칭적인 sign error가 관측 불일치 비율을 높인다. 그에 따라 fixed impact로 전환하는 session이 늘고, 원래 gate 대비 작은 mean 개선의 point estimate가 거의 사라졌다. 각 오독률에서 두 비교의 pointwise 95% interval은 모두 0을 포함했다. [Sensitivity 코드](noise56.py) · [Catalog bootstrap 결과](pilot_noise56.json)

## 서버 실행과 재현

서버의 v56 폴더에서 CPU worker에는 BELIEF_DEVICE=cpu, CUDA_VISIBLE_DEVICES 빈 값을 설정했다. CPU에서 generate56.py prepare, 이어서 bash run_refs56.sh solver를 사용한다. GPU allocation 2744468(node55)의 idle 여부를 검사한 다음 bash run_generate56.sh를 실행했다. GPU worker의 exit code는 0, 다른 GPU process는 0개였다. 두 solver 결과와 GPU forecast가 모두 준비된 뒤 CPU에서 bash run_refs56.sh panel과 noise56.py를 실행했다. Fresh deployment outcome seed는 56041971, 56045813이다. 정량 scorer와 timing은 별도 원시 파일을 읽는다. 24개 catalog 각각의 pilot·panel·scored archive 전체를 서버와 로컬에서 SHA-256 대조했다. [Freeze](generation_freeze56.sha256) · [GPU worker](generation_guard/results/worker_exit.json) · [재실행 명령](run_refs56.sh)
