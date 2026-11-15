# v14 constraints

- 한글 설명, English technical terms. Actual simulation/inference/statistics는 allocated compute node에서만 실행한다.
- 이후 GPU authorization은 초기 CPU pilot 제한을 대체한다. Idle GPU만 사용하고 타 jobs/allocation을 변경하지 않는다.
- 기존 v12/v13 artifacts를 수정하지 않는다. Compact phase는 bank/teacher posterior를 import하지 않는다.
- New selectors는 public factual evidence와 candidate predictions만 사용한다. Actual unselected observations나 target truth는 선택에 사용할 수 없다.
- Test 전에 protocol/code와 policy를 freeze한다. Failure와 negative evidence를 보존한다.
