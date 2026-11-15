# v11 execution

- 실제 simulation, inference, fitting, analysis는 서버의 할당 compute node에서 실행한다.
- 사용자가 허용한 idle GPU만 사용한다. 초기 CPU pilot의 CUDA 제한은 이후 GPU 사용 지시로 대체되었다.
- 기존 jobs와 sealed v10 artifacts를 변경하지 않는다. Foreign GPU process가 나타나면 자신의 worker만 중단한다.
- Selector는 public factual evidence만 읽는다. Private truth는 policy freeze 이후 evaluation에서만 읽는다.
- 설명은 한글, technical terms는 English로 작성한다.
