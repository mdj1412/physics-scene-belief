# v12

- 실제 simulation, inference, fitting, timing, analysis는 server allocated compute node에서 수행한다.
- 사용자에게 허용된 idle GPU만 사용한다. 상위 초기 CPU pilot 제한은 이후 GPU 사용 지시로 대체되었다.
- 다른 jobs와 v11 sealed files를 변경하지 않는다. Foreign GPU process가 나타나면 자신의 worker만 중단한다.
- Compact inference는 full posterior weights, moments, labels를 읽지 않는다. Model equations와 prior만 공유한다.
- Test policy는 public factual evidence만 사용하고 private targets/observations를 읽기 전에 freeze한다.
- 설명은 한글, technical terms는 English로 작성한다.
