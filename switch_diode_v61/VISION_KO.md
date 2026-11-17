# v61 · Contact switch와 photodiode로 LED 지연과 camera trigger 지연을 구별할 수 있는가?

두 신호와 camera의 optical pulse를 함께 읽으면, 같은 2 ms 지연이라도 LED 자체가 늦게 켜진 경우와 camera가 늦게 기록한 경우를 구별할 수 있었다. 24개 신규 catalog의 고유 impact clip 196,608개 중 **179,038개(91.06%)**에서 세 신호를 모두 확보했다. 이 clip들의 LED 지연 추정 MAE는 **0.068 ms**, trigger 지연 추정 MAE는 **0.211 ms**였다. 같은 두 지연을 동시에 가하면 LED-only reader가 물체 복귀 방향을 읽었다. 두 reader 모두 판독한 118,671개 paired 관측에서 LED-only는 118,639개를 틀렸고 dual reader의 오독은 없었다. [지연·오독 통계](vision61.json) · [사전 명세](VISION_PLAN_KO.md)

## 신호의 기준과 판독

물리 contact의 미관측 시각을 C, 실제 LED emission 지연을 L, nominal trigger에서의 camera 지연을 T라 하자. 16 kHz ADC에 contact switch와 별도 photodiode의 waveform을 동시에 담았다. 두 검출기는 각각 고유 응답 지연·gain·noise·dropout을 갖는다. Contact switch에는 bounce, photodiode에는 0.4 ms 광학 pulse를 주었다. Acquisition origin과 물리 contact의 차이는 접촉마다 무작위화했다. Camera는 1,000 fps이며 8개 frame의 15×27 pixel을 보존한다. 세 번째 frame(index 2)이 예정된 trigger 시점이다. 같은 camera 내 optical marker는 짧은 광학 pulse가 가장 가까운 exposure에 겨우 한 frame 보이는 global-shutter 근사로 렌더링했다. 실제 rolling shutter와 노출 시점에서의 impulse response는 별도 검증 대상이다. [렌더링 및 계측](vision61.py)

Switch와 photodiode에서 두 연속 sample이 baseline보다 40 counts 높아지는 rising edge를 찾고 선형 보간한다. 시간 차이로 L̂을 구한다. Camera에 관측된 발광 frame이 m일 때 **T̂ = L̂ + (2−m) × 1 ms**다. Camera만 보면 m은 L−T에 의해 결정되어 두 지연이 함께 늘 때 거의 같은 위치에 표식이 남는다. Dual reader는 실제 contact가 있을 것으로 추정한 frame부터 여섯 frame이 들어오면 상단 변형 표식과 고정 표식의 상대 이동에서 방향을 판독한다. L̂과 T̂는 공통의 L̂을 포함해 추정 오차가 서로 독립일 필요가 없다. Primary는 contact switch와 photodiode의 ADC timebase가 같다고 가정한다. 전자센서의 가상 mismatch, 실제 물리 방향, surface 또는 debounce 정답은 이 reader의 입력에 없다. [원시 ADC·camera 결합 코드](vision61.py)

## 독립적인 교차 설계

신규 board 8개의 local latch 빈도는 20.06~33.36%, electronic sign error는 1.37~3.20%였다. Switch 응답 offset은 0.032~0.137 ms, photodiode는 0.019~0.087 ms, camera offset은 −0.167~+0.211 ms, nominal LED driver 지연은 0.099~0.317 ms였다. 개별 접촉에 별도 timing jitter, waveform noise와 dropout을 더했다. Board와 parameter는 seed 61170917로 고정했으며 frozen router는 이 목록에 접근하지 않는다. 각 48개 새 surface에서 board당 64개의 8-impact pilot을 균등 생성했다. 한 pilot의 네 session과 여덟 물체가 같은 board를 사용한다. 24개 timing fixture는 catalog마다 episode 선택을 순환하여 board당 3개씩 배정했다. [Board parameter와 per-board 통계](vision61.json)

아래 0과 2는 **추가한 지연(ms)** 이다. 0에서도 실제 board offset과 jitter가 존재한다. 물리 부호, 표면·board·물체·sensor reset 배정·noise를 고정한 채, impact의 LED와 trigger 지연을 각각 바꿨다. Reference는 각 board의 nominal 조건을 유지했다. 동일 모집단을 네 cell에서 다시 평가했으므로 4배 큰 독립 표본으로 해석할 수 없다. 주어진 0/+2 ms 수준을 판별하는 threshold 1 ms는 timing의 진단에만 사용했다. [생성 코드](vision61.py) · [통계 코드](vision_score61.py)

| LED, trigger의 추가 지연 (ms) | LED / trigger MAE (ms) | LED / trigger bias (ms) | LED / trigger 수준 판별률† |
|---|---:|---:|---:|
| 0, 0 | 0.068 / 0.211 | −0.056 / +0.074 | 100.000% / 100.000% |
| 0, 2 | 0.068 / 0.211 | −0.056 / +0.074 | 100.000% / 99.906% |
| 2, 0 | 0.068 / 0.211 | −0.056 / +0.074 | 100.000% / 100.000% |
| 2, 2 | 0.068 / 0.211 | −0.056 / +0.074 | 100.000% / 99.902% |

† 모두 세 신호를 검출한 91.06%의 impact에서 계산했다. Body sign을 읽기 위한 여섯 frame이나 물체의 물리적 활성까지 요구하지 않은 분모다. Board별 세 신호 가용률은 89.40~93.64%, LED 지연 MAE는 0.034~0.103 ms, trigger 지연 MAE는 0.141~0.357 ms였다. 실물에서 새롭게 뽑은 board 수의 변동에 대한 CI는 여덟 개의 고정 board만으로 추정하지 않았다. [Board별 결과](vision61.json)

동일 clip의 관측 가능 집합에서 LED에만 2 ms를 추가하면 L̂의 평균은 **+1.99999 ms**, T̂는 **−0.000009 ms** 바뀌었다. Trigger에만 2 ms를 추가하면 L̂는 **0 ms**, T̂는 **+2.00000 ms** 바뀌었다. 평균 contrast가 정확히 2 ms에 가까운 것은 paired common noise 및 한 frame의 정수배 이동을 쓴 설계의 결과다. 한 번의 측정에서 0.00001 ms 정밀도를 얻는다는 뜻은 아니다. [Paired factorial 분석과 조건별 CI](vision61.json)

## 실제 관측 가능한 부호와 결측

| LED, trigger 추가 지연 (ms) | Nominal의 판독률 / 오독률 | LED-only의 판독률 / 오독률 | Dual의 판독률 / 오독률 |
|---|---:|---:|---:|
| 0, 0 | 65.41% / 0% | 50.38% / 0% | 45.49% / 0% |
| 0, 2 | 41.96% / 99.78% | 53.71% / 0% | 48.44% / 0% |
| 2, 0 | 65.41% / 0% | 0% / — | 45.49% / 0% |
| 2, 2 | 41.96% / 99.78% | 33.87% / 99.97% | 48.44% / 0% |

판독률은 active 물체 중 reference와 impact 변형 부호를 모두 판독한 비율이다. 오독률은 그 집합에서 두 부호 사이의 mismatch 판정이 틀린 비율이다. 각 cell의 392,688 active 관측에는 같은 noise와 물체를 쓴 P/C × A/B의 네 potential outcome이 포함된다. Dual reader가 요구하는 가용 신호가 늘어 정시에는 판독률이 LED-only보다 4.89 pp 낮았다. (2,0)에서는 늦은 LED 뒤로 여섯 frame이 남지 않아 LED-only가 모두 거부했고 dual은 contact 시점으로 창을 옮겨 178,653건을 판독했다. (2,2)에서는 오독률을 같은 118,671건에서 비교해도 99.973%에서 0%로 줄었다(24-catalog paired bootstrap 차이 95% CI −99.990~−99.954 pp). 모든 fixture에서 switch, diode, 영상 표식을 차례로 가리자 dual reader는 매번 결측으로 처리했다. [공통 관측 집합·음성 대조](vision61.json)

## 재현과 범위

실험은 Slurm compute allocation의 CPU에서 CUDA를 숨긴 채 실행했다. Catalog별 512개 pilot의 code, 모든 부호·원인별 optical outcome, 고유 contact별 unrounded delay truth와 estimate를 저장했다. 각 catalog의 대표 pilot에는 raw grayscale video, 16 kHz 2채널 waveform, electronic readout, 실제 delay와 진단용 truth가 있다. 4개 cell × 3 reader에서 각 fixture의 부호를 raw pixel과 waveform으로 다시 읽어 전체 archive와 정확히 일치하는지 검증했다. [대표 원자료](fixtures/panel_00.npz) · [전체 checksum](vision_bundle61.sha256)

두 지연의 분리는 독립적으로 발생한 waveform, 같은 ADC 시계, 파악한 camera frame cadence, 동기화 표식의 가시성, 2 ms buffer를 전제한다. Timing과 electronic sign error의 범위는 실험자가 정한 synthetic design이며 실제 장치에서 학습한 분포가 아니다. 본 검증은 기존 변형 궤적을 유지한 새 board 및 표면에 대한 것으로 실물 board, surface별 반사, 실제 접점 탄성, sub-frame LED exposure stream에서의 성능은 별도의 측정이 필요하다.
