# Continuous extension

- 사용자가 2026-11-04에 현재 할당된 GPU 전체 사용을 허용했다. 실제 기존 process를 확인한 후 GPU inference를 실행할 수 있다. 사용자의 다른 작업을 중단하거나 allocation을 변경하지 않는다.
- 상위 AGENTS.md의 CPU-only 조건은 이전 four-world pilot에 해당한다. 이 extension에서는 PyBullet은 CPU, posterior/acquisition 계산은 CUDA로 실행한다.
- 기존 pilot source, environment, result를 변경하지 않는다. 모든 새 artifact는 continuous_v2 아래에 저장한다.
- Report는 한국어 설명과 English technical terms를 사용한다. Continuous truth는 inference grid와 독립적으로 sampling한다.
- Approximate Bayes design reference와 clairvoyant oracle을 구별한다. Query-type rule의 선택과 hyperparameter는 validation에서만 정한다.
