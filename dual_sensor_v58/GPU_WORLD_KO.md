# v58 · fresh physics, GPU p90 및 sensing·reset 비용

## Deployment의 생성과 독립성

24개 catalog의 새 key 3000–3023을 동결했다. 각 catalog에서 서로 다른 material instance 두 개에 rate law 두 개를 교차해 4개 환경을 만들었다. 128개 deployment object씩 총 12,288개에서 deformation 방향을 유지하거나 뒤집는 두 물리 intervention을 비교했다. 이전 연구와 같은 PyBullet solver·contour family·두 rate law를 사용한다. 시간 해상도를 fine/ultra로 나누고, residual tangential speed가 0.020 m/s 이하이면 stick으로 기록했다. 6,144개 active object 중 2,307개(37.55%)의 binary outcome은 방향을 바꾸면 달라졌다. Inactive object에서는 바뀐 값이 없었다. Fine과 ultra 사이에 24,576개 binary endpoint 중 897개(3.65%)의 값이 달랐다. 이 비율은 같은 simulation 안에서의 조건부 결과다. [Aggregate counts](world58.json) · [사전 계획](WORLD_PLAN_KO.md)

교체 전 support, 교체 후 support, 배포 물체를 모두 포함한 36,864개의 6차원 parameter vector가 서로 유일했다. v50·55·56·57에서 확인한 122,880개와 byte 기준 중복이 0개였다. 두 방향의 GPU probability를 미리 생성하고, 각 pilot이 선택한 gate 또는 fixed 3R+I가 구입한 배포 관측만 사용해 frozen joint terminal posterior를 계산했다. Fine/ultra, conditional mean, seeds 58041729와 58084591의 realized Bernoulli deployment replay를 저장했다. [Generator](generate58.py) · [24 catalog × 7종 array checksum](deployment58.sha256)

## GPU에서 무엇을 재었는가

사용자 allocation 중 두 GPU에 기존 workload가 있고 두 GPU에서 메모리 0 MiB, utilization 0%, compute PID 0개를 확인했다. Fresh forecast는 node55의 allocation 2744468에서 guarded evaluation으로 실행했다. Node56에서 timing을 시작하려던 요청은 자동 review에서 점유 가능성을 제기해 실행되지 않았다. Forecast가 code 0으로 끝난 node55의 UUID를 다시 확인해 0 MiB·0%·PID 없음일 때 같은 GPU를 timing에 재사용했다. 두 worker 모두 자기 process tree 밖의 새 GPU process를 감시했다. [Forecast 종료 기록](generation_guard58/results/worker_exit.json) · [Timing 종료 기록](timing_guard58/results/worker_exit.json)

Online 측정의 입력은 두 sensor가 같은 여덟 물체에서 읽은 signed reference와 signed impact다. Pilot당 32개의 잠재적인 sign 판독이 있다. 4×2 dual selector의 timed call에는 이 판독을 paired outcomes로 변환하고, 네 session의 likelihood로 환경의 누적 evidence를 계산하고, outer policy와 inner v54 acquisition을 실행한 뒤 실제 fresh geometry를 frozen GPU model에 넣는 전체 과정이 포함된다. Single-sensor rule에서는 첫 instrument만 읽고, constant gate·fixed는 pilot 계산을 생략한다. Archive decoding과 device·model initialization은 warmup 전에 끝냈다. 센서 계측·기계 reset·CPU physics solver는 online GPU p90에 포함되지 않는다.

각 catalog마다 두 material의 pilot이 각자 두 law의 배포 session을 담당한다. 각 정책을 한 차례 warmup한 뒤 6-treatment Williams crossover의 세 order로 실행해 512-object batch당 정책별로 72회 측정했다. CUDA를 host wall clock 양끝에서 synchronize했다. 24개 catalog를 paired bootstrap으로 16,000회 재추출하고, primary MAP 4×2 대 8×1에 대한 pointwise CI와 다섯 대조의 Bonferroni 동시 CI를 계산했다. [Frozen protocol](TIMING_PLAN_KO.md) · [Benchmark](bench58.py)

| 정책 | GPU p90 (ms/512개) | 4×2 dual MAP 대비 차이¹와 동시 95% CI (ms) | Outer fixed 선택 (48개 material pilot) |
|---|---:|---:|---:|
| 4×2 dual MAP | 521.580 | 기준 | 19/48 |
| 8×1 dual MAP | 523.374 | -1.793 [-6.244, +1.900] | 0/48 |
| 4×2 dual loss-weighted | 521.238 | +0.343 [-7.123, +4.049] | 0/48 |
| 4×2 single naive MAP | 519.949 | +1.631 [-3.482, +4.877] | 16/48 |
| 항상 v54 gate | 522.299 | -0.719 [-8.224, +2.975] | 0/48 |
| 항상 fixed 3R+I | 519.942 | +1.638 [-4.990, +5.005] | 48/48 |

¹ 차이는 4×2 dual MAP의 p90에서 해당 정책의 p90을 뺀 값이다. Primary pointwise 95% CI는 -1.793 [-4.326, +1.261] ms다. 각 arm에서 실제 선택된 readout만 사용한 결과를 별도의 2-branch serial posterior와 저장한 forecast의 조합으로 검산했으며 최대 probability 오차는 0이었다. Risk-weighted arm의 timing fixture 48개에서 fixed로의 전환은 0회였다. 드문 전환 branch 단독 latency는 이 측정에서 추정하기 어렵다. 개별 catalog에서 여섯 정책의 timing이 함께 변동했고, 모든 사전 등록 대조의 동시 CI에 0이 포함됐다. [GPU 수치](timing58.json) · [측정 코드 및 raw checksum](timing_results58.sha256)

## 물리 예산

Pilot에서는 서로 다른 8개 specimen 각각에 1 reference 및 1 impact, 이후 각각 1 reset을 지불한다. 설계당 16 contacts, 16 resets, 32 signed sensor outputs(결측 시에도 slot 사용)다. 두 sensor가 동일 접촉에서 동시에 읽기 때문에 물리적 impact는 둘 다 8회다. v54 policy를 그다음 1회 deployment하는 경우 8 contacts·8 resets가 추가되어 24 contacts·24 resets다. 교체 전 공통 calibration과 deployment object당 공통 weak readout은 양쪽에 같아 증분 원가에서 제외했다. 장치 간 synchronization와 동시 촬영이 pilot당 sensing 시간을 늘릴 수 있다.

Pilot을 H개 deployment session과 공유할 때 physical contacts와 resets는 각각 16+8H개다. 8×1의 session setup은 8회, 4×2는 4회이며, 표본 교체는 양쪽 모두 8회다. 독립 session 재설정에 S초가 추가되면 같은 contact/reset 비용에서도 4×2가 4S/H초 줄인다. 각 정책이 마지막 impact를 사는 비율 r에 따라 unpadded deploy 시간은 달라진다. V57의 모델 시간(weak+reset 0.44초, reference+reset 1.04초)에서 impact+reset T의 양쪽 공통 padded budget은 deployment당 (8·(1.04+T)+8S)/H + 4.88 + max(1.04,T)초다. 센서 준비, 개별 물체 준비 등 공통 비용은 이 예시에서 0으로 놓았다. H=1, S=0이면 T=1.04초에 모두 22.56초다. T=2.08초이면 공통 cap 31.92초 아래에서 dual MAP 4×2는 31.49초(r=58.67%), dual MAP 8×1은 31.24초(r=35.08%), risk 4×2는 31.24초(r=35.08%), fixed는 31.92초다. 앞 두 방식의 차이는 약 0.245초다. 기계 실측은 아직 없으며, pilot을 새로 사는 실제 상황에서는 constant gate나 fixed가 pilot을 생략할 수도 있다. [Cost script](cost58.py) · [Cost table](cost58.json)
