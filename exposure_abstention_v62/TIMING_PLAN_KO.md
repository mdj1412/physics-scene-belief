# v62 GPU p90 사전 명세

LED pulse 폭 0.20·0.80 ms와 shutter exposure 0.25·0.80 ms의 네 조합을 모두 본다. 24개 신규 catalog 각각의 2개 표면에서 동일 raw clip 32개, ADC waveform 64개, paid sensor readout으로 정책을 실행한 뒤 4 deployment session의 512개 물체에서 공통 frozen forecast를 계산한다. 각각의 surface에서 4 sessions × 2 objects로 reference와 impact를 8회씩 가한다. 각 fixture에서 board는 catalog index modulo 8이며 8개 board가 각 3개 catalog에 배치된다. P/C와 target A/B는 균등, C의 표면별 fault high/low는 실제로 유리한 행동과 독립 배정한다.

주 비교는 selective reset과 v61 reset, v61 sham, always v54 gate다. 보조 비교는 selective sham과 nominal reset이다. Gate는 물리 비용이 맞춰져 있어도 online에서 쓰지 않는 pixel/ADC를 읽지 않는다. Reader가 얻은 code와 action을 전체 pilot의 독립 생성 결과와 검산한다. 모든 arm에서 terminal forecast까지 실행해 physics GPU 결과를 독립 저장한 두 방향의 예측과 대조한다. CUDA synchronization 전후의 host elapsed time에 raw readout 판독, confidence 보류, routing, v54 acquisition, terminal 계산, CUDA inference가 포함된다. 광학 획득·접촉·reset 대기·디스크·모델 초기화는 별도다.

각 catalog의 4개 cell마다 six-arm Williams crossover를 실행하며 정책당 96개 measured block, catalog마다 arm별 warmup 1회다. 주 통계는 96개 block 전체 p90 및 24개 catalog paired bootstrap 16,000회의 세 비교 동시 95% CI다. 추가로 8개 board를 먼저 복원추출한 뒤 각 board에서 세 catalog를 재추출하는 nested bootstrap 8,000회, board별·cell별 및 paired median을 보고한다. Accuracy에 쓰는 완전 교차 sampling과 timing에 쓰는 nested sampling의 단위는 다르다. 같은 알고리즘의 raw 판독 + routing만 별도 CPU에서 정책당 576회 측정한다.
