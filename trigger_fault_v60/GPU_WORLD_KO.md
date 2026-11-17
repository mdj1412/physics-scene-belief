# v60 · Fresh deployment, raw frame부터 측정한 GPU p90, sensing 예산

## 새 물체와 공통 outcome

Catalog key 5000–5023에서 material instance 두 개씩을 새로 만들고 각각 기존 두 rate law의 별도 128개 deployment object를 평가했다. 총 12,288개다. 교체 전·후 support를 포함한 36,864개의 6차원 physical parameter vector는 서로 고유했고 v50·55·56·57·58·59에서 확인한 이전 196,608개와 같은 vector는 없었다. 동일 물체에서 방향만 뒤집은 두 PyBullet trajectory의 residual tangential speed가 0.020 m/s 이하이면 stick으로 두었다. 변형이 활성인 6,144개 중 2,301개(37.45%)에서 intervention에 따라 binary stick label이 바뀌었다. 비활성 물체에서는 바뀐 label이 없었다. Fine/ultra의 24,576개 endpoint 중 811개(3.30%)에서 판정이 달랐다. 이 연구의 fresh 물체는 기존 합성 family에서 생성했다. [생성 계획](WORLD_PLAN_KO.md) · [Generator](generate60.py) · [집계](world60.json)

두 변형 방향의 conditional probability를 frozen GPU forecast로 보관했다. 기존 v54 acquisition, 구입한 네 번째 sensor, frozen continuous joint terminal과 결합해 각 surface의 future event별 Brier 및 두 개의 realized replay(seed 60041729·60084591)를 구성했다. 168개 원시 archive의 SHA-256을 local 및 remote copy에서 검산했다. [Deployment manifest](deployment60.sha256)

## GPU 계측 단위

4개 allocation 중 2개에서는 다른 workload가 각각 40 GB가량 사용 중이었고, 두 개는 0 MiB·0%·compute PID 없음이었다. Node55/2744468의 NVIDIA RTX PRO 6000 Blackwell Server Edition 한 장을 골랐다. Fresh forecast와 timing 직전에 각각 memory·utilization·compute process를 다시 확인했고, guard로 외부 GPU process가 생기는지 감시했다. CPU solver와 vision 준비는 CUDA를 끄고 Slurm compute node에서 수행했다. [Forecast 종료 기록](generation_guard60/results/worker_exit.json) · [Timing 종료 기록](timing_guard60/results/worker_exit.json)

한 호출은 2개 surface의 32개 8-frame clip, 같은 여덟 물체의 paid electronic sign, 그리고 이들이 적용될 4개 deployment session의 센서를 사용해 총 512개 object의 probability를 계산한다. Frame은 CPU 메모리의 실제 uint8 pixel 배열이다. 동기화된 arm은 LED contrast로 event 위치를 정렬한다. Uncorrected arm은 nominal window를 사용한다. 검출·부호 추출·likelihood·outer 정책 선택·필요한 session의 inner v54 gate·terminal mixture·GPU model의 forward를 모두 host wall clock에 포함하고 양끝에서 CUDA를 synchronize했다. 항상 v54 gate는 같은 물리 예산을 할당받아도 결정을 바꿀 수 없는 pilot을 online에서 읽지 않는다. 광학 획득·기계 접촉·전자장치 reset 대기, 디스크 압축 해제와 모델 기동 시간은 제외했다. [Benchmark](bench60.py) · [계측 계획](TIMING_PLAN_KO.md)

세 가지 impact trigger 지연 비율 0/20/100%와 세 가지 surface fault 관계 matched/independent/reversed를 교차했다. 각 fresh catalog 내부에서는 세 measured block에 서로 다른 delay와 fault 관계를 배정했다. 각 block에서 6개 정책 모두를 동일한 픽셀·전자 readout·deployment 물체에 대응시켰고, Williams crossover로 실행 순서와 직전 정책의 빈도를 균형화했다. pooled 72개 block에는 9개 cell이 각각 8개씩 있다. 주 Brier 평가와 동일한 independent-hazard 집단은 24개 block이며 catalog마다 1개씩이다. Independent 셀에서 두 hazard의 균등 배정을 material·surface regime별로 층화했다. 모든 관측 코드가 독립 저장 readout과 일치하는지 검사하고 terminal과 actual forecast도 별도 replay와 비교했다. [모집단 정렬 사전 명세](TIMING_ALIGNMENT_KO.md)

| 정책 | Independent fault·균등 delay GPU p90 (ms/512 objects) | 세 fault 관계·균등 delay GPU p90 (ms/512 objects) |
|---|---:|---:|
| Corrected loss-weighted reset | 480.029 | 480.649 |
| Corrected loss-weighted sham | 480.451 | 481.030 |
| 항상 v54 gate | 479.928 | 480.407 |
| Uncorrected loss-weighted reset | 480.173 | 480.326 |
| Uncorrected loss-weighted sham | 485.086 | 487.596 |
| Corrected MAP reset | 493.251 | 488.975 |

| 같은 모집단에서 corrected reset − 대조 | Independent fault 차이 [동시 95% CI], ms | 전체 차이 [동시 95% CI], ms |
|---|---:|---:|
| Corrected loss-weighted sham | -0.422 [-18.078, +0.145] | -0.381 [-14.982, +7.828] |
| 항상 v54 gate | +0.100 [-3.842, +0.828] | +0.242 [-15.062, +8.050] |

데이터는 independent 모집단 정책당 24회, 전체 72회다. 24개 catalog를 함께 16,000회 복원추출해 각 모집단에서 두 primary contrast에 대한 동시 95% CI를 계산했다. 실행 순서 및 직전 정책의 각 조합은 전체에서 각각 12회 나타났다. 관측된 최대 forecast replay 오차는 0였다. Cell별 8개 관측의 p90은 작은 표본의 기술통계다. [Raw pixel-to-terminal 결과](timing60.json) · [원자료 예시](timing60/panel_00.npz)

별도의 CPU microbenchmark에서는 동일 32개 영상의 marker 검출·pixel tracking·policy routing만 따로 실행했다(정책당 432회, compute node56). p90은 corrected reset 0.3474 ms, corrected sham 0.3462 ms, uncorrected reset 0.3451 ms, 항상 gate 0.0012 ms였다. 두 reset 조건의 preprocessing 차이는 0.00235 ms였고, 전체 GPU p90의 차이는 이 값에서 단순히 계산할 수 없다. [Pixel-to-routing 비용](reader_cost60.json) · [측정 코드](reader_cost60.py)

## 물리 및 촬영 예산

한 surface에서 네 session마다 두 개의 서로 다른 specimen을 검사한다. 각 specimen에 reference 1회와 impact 1회, contact마다 mechanical reset을 1회 지불하므로 모든 비교가 **8 impacts, 8 references, 16 mechanical resets**다. Reset과 sham을 구분하는 비용은 둘째 물체 전 **4 electronic sensor resets**다. Camera arm은 동일 contact를 동시에 촬영해 16개 clip(128개 frame, 8-bit 15×27 pixel에서 51,840 bytes)을 읽는다. Sync는 buffered pixel의 알려진 LED ROI를 사용한다. Uncorrected와 corrected arm이 별도의 물리 contact나 추가 clip을 지불한 적은 없다.

교체 후 첫 deployment까지 합하면 24 contacts·24 mechanical resets다. 한 pilot을 H개 deployment에 재사용하면 각각 16+8H개다. 실제 장치에서 전자 reset당 겹치지 않는 시간 R, clip당 K, surface당 marker/clock 초기화 비용 L이 추가되면 세션당 해당 arm의 비용에 4R/H, 16K/H, L/H를 더해야 한다. R, K, L은 실물에서 측정하지 않았다. Gate는 운영 시 8회 pilot을 생략할 수 있다. 보고한 같은 예산의 비교에서는 gate도 동일 pilot 접촉을 지불한다. [Cost 산출](cost60.py) · [모집단별 ledger](cost60.json)

Independent 모집단에서 마지막 deployment impact의 구매율은 corrected reset 35.8995%, corrected sham 35.8701%, 항상 v54 gate 34.9080%다. 첫 deployment와 겹치지 않는 추가 setup을 0으로 두고 reference 및 impact 1회에 reset 포함 각각 1.04초를 주면 모든 방법에서 22.56초다. Impact가 2.08초이면 common padded budget은 31.92초이고 padding 전 기대시간은 각각 31.25335, 31.25305, 31.24304초다. 이전 버전과 같은 모형 비용이며 hardware wall time을 실측한 결과는 아니다.
