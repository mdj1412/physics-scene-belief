# v15

- 설명은 한글, technical terms는 English. 실제 computation은 allocated remote node에서만 수행한다.
- 이후 GPU authorization은 초기 CPU-only pilot 제한을 대체한다. Idle GPU만 사용한다.
- 기존 allocation/process 및 v14 sealed files를 변경하지 않는다.
- Policy는 factual evidence와 public query context만 받는다. Truth는 generator/evaluator만 사용한다.
- Correct-law inference는 privileged diagnostic으로 명시한다. 결과를 본 뒤 primary settings를 바꾸지 않는다.
