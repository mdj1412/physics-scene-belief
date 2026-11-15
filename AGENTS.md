# Project constraints

- 설명과 보고서는 한국어로 작성하고 technical terms는 English로 유지한다.
- 실제 실험은 이 remote project directory 아래에서 실행한다. Local mirror는 준비·검토·보고서 전달용이다.
- 사용자의 기존 jobs, processes, allocation을 취소하거나 수정하지 않는다.
- 모델 training 또는 inference를 새로 실행하기 전에 사용자에게 할당된 GPU의 allocation 및 실제 process/memory usage를 확인한다. 모든 GPU가 사용 중이면 해당 실행을 중단하고 사용자의 후속 지시를 기다린다. 순간 utilization 0만으로 idle이라고 판정하지 않는다.
- 이 pilot의 CPU simulation과 Bayesian calculation에는 CUDA를 비활성화한다. Login node에서 compute workload를 실행하지 않는다.
- Simulator state는 renderer, candidate forward model, evaluation에만 사용할 수 있다. Policy에 hidden world, unselected observation, target ground truth를 넣지 않는다.
- Known-geometry/restitution 진단은 privileged information baseline으로 명시한다.
- 기존 결과를 변경하는 추가 실험은 새로운 output directory에 저장하고, 완료 여부와 실제 실행 evidence를 보고한다.
