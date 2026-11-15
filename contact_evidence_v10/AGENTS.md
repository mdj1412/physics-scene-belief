# v10 실험 제약

- 실제 simulation, inference, analysis는 서버의 할당된 compute node에서 수행한다.
- 사용자가 허용한 idle GPU를 사용한다. 상위 AGENTS.md의 CUDA 비활성화는 최초 CPU pilot에 한정되며, 이후 사용자 GPU 실행 지시를 따른다.
- 기존 allocation과 다른 process를 변경하지 않는다. Foreign GPU process가 나타나면 이 실험 worker만 중단한다.
- Acquisition에는 public factual data만 전달한다. Private truth와 unselected actual observations는 policy freeze 이후 evaluation에서만 읽는다.
- v8 sealed artifacts는 수정하지 않는다. 설명은 한글, technical terms는 English로 작성한다.
