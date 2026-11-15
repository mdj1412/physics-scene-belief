# Scope

- 이 directory는 user-authorized GPU extension이다. Parent의 old CPU pilot 제한은 적용하지 않는다.
- 실제 inference/simulation은 datacenter3의 사용자 Slurm allocation에서만 실행한다.
- 기존 jobs를 변경하지 않는다. GPU process와 memory를 확인한다.
- 설명은 한국어, technical terms는 English. Hidden truth는 generator와 evaluation에만 전달한다.
- 모든 comparison은 observation budget, particle budget, forward-call count를 기록한다.
- Synthetic RGB sensor를 real-world validation 또는 general visual reconstruction으로 표현하지 않는다.
- 결과가 negative여도 보존한다. Test 결과로 hyperparameter를 선택하지 않는다.
