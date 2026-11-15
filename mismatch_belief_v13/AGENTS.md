# v13

- 설명은 한글, technical terms는 English로 작성한다.
- 실제 experiments와 statistical analysis는 allocated server compute node에서 실행한다.
- 이후 사용자 GPU authorization은 상위 초기 CPU-only pilot 제한을 대체한다. Idle GPU만 사용하고 다른 jobs/allocation을 수정하지 않는다.
- v12 sealed artifacts를 수정하지 않는다. Compact phase는 QMC/posterior teacher에 의존하지 않는다.
- Thresholds와 policy는 test 전에 freeze한다. Detector/selector는 public factual evidence와 실제 선택된 observation만 사용한다.
- Post-observation expansion은 저장한 factual evidence와 selected observation을 replay한다. Target truth는 evaluation에만 사용한다.
