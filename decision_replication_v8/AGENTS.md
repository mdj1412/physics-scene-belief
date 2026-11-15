# v8 scope
- 사용자에게 할당된 idle GPU에서만 inference를 실행한다. 원래 pilot의 CPU-only 규칙은 이 GPU extension에는 적용하지 않는다.
- 사용자 기존 jobs/allocations를 종료하거나 수정하지 않는다. Foreign GPU process가 나타나면 우리 worker만 종료한다.
- 기존 v7 artifacts는 수정하지 않는다. Local은 준비·분석·보고서 작성용이다.
- 설명은 한글, technical terms는 English. Synthetic precise view를 실제 camera로 표현하지 않는다.
- Policy selection에는 public factual evidence와 independent outer observations만 사용한다. Private truth/actual observations는 policy freeze 이후 evaluation에서만 읽는다.
