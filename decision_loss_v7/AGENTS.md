# v7 scope

- 기존 user authorization에 따라 현재 idle인 사용자 GPU allocations만 사용한다. 초기 pilot의 CPU-only 조건은 이 GPU extension에 적용하지 않는다.
- 외부 GPU process가 시작되면 우리 worker만 중지한다. 기존 jobs/allocations를 취소하지 않는다.
- Simulation/inference는 allocated server compute node에서만 실행한다. Local은 준비, 결과 분석, 보고서 작성용이다.
- v5/v6 artifacts를 변경하지 않는다. Acquisition에는 public factual evidence와 posterior banks만 전달한다. Truth와 반복 actual observations는 생성/evaluation 단계에서만 읽는다.
- 설명은 한글, technical terms는 English. Exploratory stress condition의 결과를 real-world 또는 새로운 SMC algorithm의 입증으로 표현하지 않는다.
