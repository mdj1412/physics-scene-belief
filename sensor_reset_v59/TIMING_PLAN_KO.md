# v59 GPU p90·비용 사전 계획

동일한 512개 fresh deployment object에 대한 GPU prediction에 paid reference/impact sign의 encoding, 실제 reset instrument ID, 카메라의 사용 또는 masking, 첫 3회 reference와 v54 acquisition, 공통 terminal decoder까지 포함한다. 한 material에 8개 sacrificial object와 4개 group을 사용한다. 2개 material, 2개 law의 4개 deployment session을 한 번의 batch로 처리한다. 24개 catalog의 material·A/B·물리/오류 원인을 균형화하고 pilot 하나를 같은 material의 두 law에 공유한다. Actual sign acquisition과 actuator reset, CPU contact solver는 device latency 밖이며 별도 ledger로 보고한다.

6개 arm은 reset on/off × audit camera on/off 4개의 MAP와 항상 v54 gate, fixed 3R+I다. 각 catalog마다 무작위 순서 warmup 1회, 이후 두 인접 catalog가 6개 Williams crossover order를 절반씩 나눠 정책당 3회 측정한다. 각각 총 72회, paired 24-catalog bootstrap 16,000회로 p90의 reset 효과(cam on/off), camera 효과(reset on/off)의 4개 contrast에 대한 pointwise 및 동시 95% CI를 구한다. 계산은 양끝 CUDA synchronize를 쓴 host wall clock이다. 4쌍 전부 sensor reset을 해도 여덟 impact는 같고, 기계 contact/reset 횟수와 sensor reset 4회를 분리한다. 실제 device setup·readout duration은 관측하지 않는다.
