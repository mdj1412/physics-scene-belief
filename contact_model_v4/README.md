# Contact Model Belief — v4

2026-11-05에 수행한 user-authorized experiment extension이다. 기존 `continuous_v2`, `correlated_v3` 결과는 변경하지 않았다. 설명은 한국어, technical term은 English로 작성한다.

## 범위

세 rigid-contact layout에서 다음을 분리한다.

- Query-dependent observation selection: EVSI, parameter EIG, validation-selected rules.
- Finite posterior support: 같은 upstream SMC를 systematic / contact-sequence stratification으로 압축.
- Collision-proxy misspecification: sharp / rounded / outside-support truth, predictive check와 model expansion.
- Real data feasibility: Deform360 한 물체의 세 interaction을 두 camera에서 확인. **Real physical prediction 검증은 아니다.**

‘세 family’는 동일한 rigid impact law의 서로 다른 scene layout이다. Rotation, articulated contact, deformable dynamics의 세 가지 물리계를 구현한 것이 아니다. Noisy RGB는 known camera/color 기반 synthetic sensor이며 learned perception이 아니다.

## 실행 환경

Remote root: `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/contact_model_v4`.

기존 `../continuous_v2/.venv/bin/python`: Python 3.9.18, NumPy 1.26.4, SciPy 1.13.1, PyTorch 2.7.1+cu128, PyBullet 3.2.7. GPU는 node56의 RTX PRO 6000 Blackwell 두 개, Slurm allocations 2642475/2642476을 사용했다. 다른 사용 중인 allocation은 변경하지 않았다. Float64 CUDA simulation/inference이며 neural model training은 없다. OpenCV 4.10.0.84는 이 directory의 `vendor/`에 별도로 설치했다.

아래 명령은 **활성화된 자기 allocation 안에서** 실행한다. Historical job ID를 새로운 실행에 그대로 사용하지 않는다. 한 GPU에서는 순서대로 실행하면 된다. 실제 연구에서는 두 worker를 서로 다른 할당 GPU에 배치했다.

```bash
cd /scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/contact_model_v4
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
../continuous_v2/.venv/bin/python validate.py
../continuous_v2/.venv/bin/python run.py --worker 0 --workers 2 --n 768
../continuous_v2/.venv/bin/python run.py --worker 1 --workers 2 --n 768
../continuous_v2/.venv/bin/python calibrate.py
../continuous_v2/.venv/bin/python audit.py
../continuous_v2/.venv/bin/python refine.py --worker 0
../continuous_v2/.venv/bin/python refine.py --worker 1
../continuous_v2/.venv/bin/python continuous_proxy.py --worker 0
../continuous_v2/.venv/bin/python continuous_proxy.py --worker 1
../continuous_v2/.venv/bin/python analyze.py
../continuous_v2/.venv/bin/python analyze_refined.py
../continuous_v2/.venv/bin/python plot_diagnostics.py
```

`run.py`, `refine.py`, `calibrate.py`는 이미 존재하는 case를 건너뛴다. 새로운 configuration은 **별도 directory**에 복사해 실행해야 하며, 기존 결과와 섞지 않는다. `audit.py`는 main case가 모두 존재한 뒤 실행한다. 초기 부분 실행에서 누락된 case를 보완하기 위해 이번에는 `audit.py --posterior-only`를 추가 실행했다. 최종 convergence record는 36개여야 한다.

## Dataset

```bash
../continuous_v2/.venv/bin/python download_subset.py
../continuous_v2/.venv/bin/python dataset_audit.py
```

저장 위치는 사용자가 지정한 `/scratch2/mdj1412/datasets/deform360_contact_belief_audit`다. Default revision은 `5ea8c5d3fc7b4a7b4f9f921f2ceb1de24610f6a4`로 고정했다. 16 files, 22,615,329 bytes를 내려받았다. `download_manifest.json`에 URL, byte count, SHA256이 기록되어 있다. `dataset_audit.py`는 공식 calibration dictionary의 pickle-backed `.npy`를 읽는다.

## 결과와 해석

- `results/main/`: 72 worlds, 각 6 queries; 24 validation / 48 test. 원본 observation, truth, metrics, SMC cost, factual posterior.
- `results/calibration/`: 독립 null calibration 192 worlds, family당 64.
- `results/refined/`: 72 worlds의 각 observation 후 full SMC-1536 refit, moves 12. 추가 three-observation broad/known-proxy 진단.
- `results/continuous_proxy/`: post-hoc strong baseline. Rounding까지 continuous parameter로 추정하는 five-dimensional SMC-1536. ‘Outside’는 원래 discrete support 밖이라는 의미이며 이 baseline의 prior 밖은 아니다.
- `results/audit/cross_engine.json`: 최초 unmatched solver 설정의 결과. 삭제하지 않았다.
- `results/audit/cross_engine_matched.json`: contact ERP/slop 설정을 맞춘 PyBullet comparison 72 rollouts.
- `results/audit/posterior_convergence.json`: predetermined validation 9 cases × 2 particle counts × 2 seeds.
- `results/analysis/summary.json`: equal-budget SMC-768 model comparison과 compression 결과.
- `results/analysis/refined_summary.json`: observation selection의 primary 결과. 동일 SMC-1536 full refit을 사용한 policy 비교.
- `results/real_data_audit/`: decoding, timestamp alignment, calibration 존재 여부와 시각적 preview.

평가 비용의 `calls`는 likelihood 계산용 **forward observation evaluations**이며, `view`도 포함한다. Acquisition, query prediction, rasterization 비용까지 합한 end-to-end simulator-call count가 아니다. Compression은 upstream SMC를 공유하므로 end-to-end efficiency를 입증하지 않는다. 동시 GPU 작업 중 측정한 `seconds`는 speed benchmark로 해석하지 않는다.

`coverage90`는 vx/vy 각각의 marginal central interval을 평균한 수치다. Joint 2D region coverage가 아니다. ESS가 1536이라고 posterior mixing이 완벽한 것은 아니다. Bootstrap은 world 단위이며 같은 world의 six queries를 독립 표본으로 세지 않는다. 모든 interval은 exploratory이고 multiple-comparison correction을 적용하지 않았다.

Model checking의 95% null threshold는 independent calibration의 62nd/64 order statistic이다. 이는 counterfactual coverage나 OOD detection 보장이 아니다. `known_proxy`는 true rounding을 알려주는 privileged diagnostic이며 실제 사용 가능한 baseline이 아니다. `joint_MAP`은 factual-only point diagnostic으로, conditional plug-in geometry baseline과 다르다.

연구 해석과 논문 검토는 상위 workspace의 `outputs/physics_scene_belief_contact_model_v4_*` 보고서에 정리했다. Remote에는 이 directory의 `reports/`로 복사한다.
