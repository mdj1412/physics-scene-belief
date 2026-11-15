# Boundary Belief v5

Raw-image curvature 정보가 observation selection의 결론을 바꾸는지 평가한다. 설명은 한국어, technical term은 English. 기존 contact_model_v4의 결과는 변경하지 않았다.

Remote: `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/boundary_belief_v5`.

## 구현

- `geometry.py`: RGB rasterization과 image-space noise, connected-component boundary extraction. `[g,d]`와 `[g,d,corner-area-deficit]`를 같은 image에서 얻는다. Rounded rectangle에서는 area deficit=(4−π)b²다.
- `calibrate_geometry.py`: independent image calibration, common covariance, b-grid bias curve, 별도 q-only holdout posterior audit.
- `belief.py`: five-dimensional continuous-proxy SMC, observation-conditioned full refit, task EVSI / parameter EIG.
- `run.py`: fresh 72 worlds, 두 resolution과 두 geometry descriptors. Dynamic outcomes와 posterior를 공유한다.
- `analyze.py`: validation-selected rules, world-clustered CI, geometry 개선과 policy 변경 효과의 분리.
- `audit.py`: predetermined validation cases 1/25/49에서 8192-particle acquisition, 3072-particle view refit, 두 seeds.
- `real_boundary.py`: 기존 Deform360 six videos의 contour/SDF 추출, representation fidelity와 segmentation sensitivity. Physical prediction은 수행하지 않는다.
- `pixel_only.py`: pre-render g/d jitter를 제거한 secondary sensor condition. Factual/dynamic 결과를 그대로 재사용한다.
- `conditional_view.py`: sharp view likelihood에 대한 conditional importance integration; original bank acquisition과 비교한다.
- `validate_conditional.py`: truncation normalization을 독립 Gaussian rectangle quadrature와 비교한다.
- `real_geometry_audit.py`: oriented shape / quadrilateral / convex hull controls와 crop stratification.
- `queue_run.py`: incomplete worlds의 shared-queue resume. 최초 parity main processes를 모두 끝내거나 종료하고 completed-output 검증을 마친 뒤 사용한다. 실행 중인 parity workers와 동시에 사용하지 않는다.
- `conditional_failure_audit.py`: aggregate 확인 후 ESS로 선택한 cases 66/68의 hypothetical views를 full SMC 두 seeds로 재검증한다. Main scores를 바꾸지 않는다.

## 실행 순서

사용자의 활성 allocation 중 빈 GPU에서 실행한다. 이번 사용 allocation은 2676322(node55), 2676323(node56)이다. 기존 jobs와 pending allocations를 변경하지 않는다. GPU inference를 login node에서 실행하지 않는다.

아래 목록은 compute node 안에서 실행할 명령이다. Login node에서 시작할 때는 예를 들어 `srun --jobid=<현재 승인된 allocation> --overlap --exact -N1 -n1 -c1 --gres=gpu:1 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ../continuous_v2/.venv/bin/python -u run.py --worker 0`으로 감싼다. 재현 시 만료된 allocation ID를 그대로 사용하지 않는다.

```bash
cd /scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/boundary_belief_v5
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
../continuous_v2/.venv/bin/python calibrate_geometry.py --quality high
../continuous_v2/.venv/bin/python calibrate_geometry.py --quality low
../continuous_v2/.venv/bin/python calibrate_geometry.py --quality pixel_only
../continuous_v2/.venv/bin/python validate.py
../continuous_v2/.venv/bin/python run.py --worker 0
../continuous_v2/.venv/bin/python run.py --worker 1
../continuous_v2/.venv/bin/python pixel_only.py --worker 0
../continuous_v2/.venv/bin/python pixel_only.py --worker 1
../continuous_v2/.venv/bin/python conditional_view.py --worker 0
../continuous_v2/.venv/bin/python conditional_view.py --worker 1
../continuous_v2/.venv/bin/python conditional_view.py --worker 0 --outer 128 --inner 2048 --out results/conditional_refined
../continuous_v2/.venv/bin/python conditional_view.py --worker 1 --outer 128 --inner 2048 --out results/conditional_refined
../continuous_v2/.venv/bin/python conditional_view.py --cases 1 25 49 --outer 128 --inner 2048 --out results/conditional_convergence
../continuous_v2/.venv/bin/python validate_conditional.py
../continuous_v2/.venv/bin/python audit.py
../continuous_v2/.venv/bin/python real_boundary.py
../continuous_v2/.venv/bin/python real_geometry_audit.py
../continuous_v2/.venv/bin/python analyze.py
../continuous_v2/.venv/bin/python analyze_audit.py
../continuous_v2/.venv/bin/python figures.py
../continuous_v2/.venv/bin/python conditional_failure_audit.py --case 66
../continuous_v2/.venv/bin/python conditional_failure_audit.py --case 68
../continuous_v2/.venv/bin/python write_report.py
```

실제 실행에서는 calibration quality와 main worker를 각각 다른 GPU allocation에 배치했다. Python은 기존 venv, OpenCV는 `../contact_model_v4/vendor`를 읽는다. 새 neural model을 학습하지 않는다.

## 재현과 한계

`PROTOCOL.md`에 comparison을 정의했다. Result case 파일이 있으면 main/audit는 건너뛴다. 설정 변경은 별도의 result directory에서 수행해야 한다. Calibration을 다시 실행하면 파일을 덮어쓰므로 완료된 main run에 다른 calibration을 섞으면 안 된다.

Known camera, color, axis alignment, family와 rectangle width를 가정한다. 이 family 안에서 boundary descriptor가 collision proxy를 복원하지만 arbitrary shape의 raw pixel likelihood는 아니다. Gaussian feature likelihood는 independently calibrated approximation이다. GPU full refit을 사용하며 small-bank importance update를 primary 결과로 사용하지 않는다.

Dynamic sensor는 T=0.75, 0.8의 실제 rendered position으로부터 50 ms average velocity를 얻는다. Likelihood도 같은 값을 예측한다. Counterfactual target인 T=0.8 instantaneous velocity와 구분한다. 수정 이전 partial output은 이름에 `_instantaneous_invalid`가 들어간 directory에 보존하며 analysis에서 읽지 않는다. 상세 amendment는 `PROTOCOL.md`에 남겼다.

`calls`는 likelihood용 forward observation evaluations, `dynamic_calls`는 그중 dynamic forward evaluations다. Query prediction, acquisition의 candidate evaluations, rendering과 feature extraction을 모두 포함한 end-to-end budget이 아니다. Policy에 따라 관측하는 action은 1개지만 view와 real probe의 실제 비용이 같지는 않다. 모든 정책의 likelihood와 parameter prior, observation noise를 일치시켰다.

`trajectory_rollouts`는 해당 timed SMC 또는 conditional integration 구간의 실제 `simulate` trajectory 수이며 late-contact prefix simulation도 포함한다. 이 수 역시 전체 process의 rendering, calibration, predictive evaluations를 모두 합한 비용은 아니다. Conditional integration에는 추가 계산이 들므로 equal-compute comparison으로 표현하지 않는다.

Real video의 canonical segmentation을 reference로 삼은 IoU는 physical GT accuracy가 아니다. 128-point contour는 external silhouette만 압축하며 holes는 full SDF에 남는다. Pixel-space curvature를 material stiffness/restitution으로 해석하지 않는다. Camera calibration이 존재해도 deforming cloth의 visible contour를 바로 3D collision surface로 바꿀 수는 없다.
