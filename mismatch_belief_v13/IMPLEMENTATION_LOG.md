# Pre-test implementation log

- 첫 실행에서 expanded linear-Gaussian validation은 통과했으나 calibration artifact의 `primary` key가 actions와 scores에 중복되어 저장에 실패했다.
- Test data generation/freeze 이전에 발생했으며 test outcomes를 읽지 않았다. Key를 `primary_actions`로 구분했다.
- 실패한 실행의 calibration artifacts/logs는 `pretest_attempt1/`에 보존한다. Final calibration과 test는 깨끗한 `results/`에서 다시 실행한다. Model, thresholds quantile rule, test design은 변경하지 않았다.
