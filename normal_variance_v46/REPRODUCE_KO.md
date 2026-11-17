## 재현 자료 및 서버 기록

서버 경로는 `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/normal_variance_v46`다. 두 idle GPU allocation에서 각 6개 catalog의 prediction을 계산하고 latency는 같은 단일 GPU에서 측정했다. Source calibration·pressure·rigid·paired statistics도 할당된 서버 compute node에서 실행했다. 다른 작업이 쓰던 두 GPU는 사용하지 않았다.

| 재현 단계 | 코드·데이터 |
|---|---|
| 사전 계획, geometry, source/target 경계 | [분석 계획](PLAN_KO.md) · [Geometry](geometry46.py) · [Source router](router46.json) · [Target label 실행 시점](TARGET_GO) |
| Raw clip 재생, source 학습, geometry audit | [영상 생성](capture46.py) · [Variance estimator](camera46.py) · [Source fit](fit_route46.py) · [Source CV](source_cv46.json) · [CPU/GPU route replay](route_replay46.json) |
| Target probability·ground truth | [GPU prediction](runner46.py) · [Pressure와 Bullet](mechanics46.py) · [압력·rigid archive SHA-256](mechanics_outputs46.sha256) |
| Primary 채점·GPU timing | [Paired Brier](score46.py) · [Timing](timing46.py) · [두 번째 randomized crossover](audit_latency46_repeat.json) · [GPU 사용 종료](gpu_release46.log) |
| 원인 분리와 추가 검산 | [Target cross-fit](transfer46.py) · [Target axis 진단](aligned_axis_score46.py) · [Paired variant](paired_variants46.py) · [보수적 위치 baseline](conservative46.py) · [그림 생성](figures46.py) |
