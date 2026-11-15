# Raw-image boundary experiment protocol

2026-11-08. 분석 기준을 main test 결과 해석 전에 고정한다. 기존 v4는 변경하지 않는다.

## 비교

- Fresh 72 worlds: family당 24, 첫 8 validation / 나머지 16 test. Seed 2026110800 + case. Continuous b ~ Uniform(0,6 mm), 다른 parameter는 v4의 prior를 유지한다.
- 같은 image의 `[g,d]` bbox descriptor와 `[g,d,corner-area-deficit]` boundary descriptor를 비교한다. 알려진 rounded-rectangle family에서는 deficit = (4−π)b²로 curvature가 식별된다. 임의의 shape에 대한 충분한 representation이라는 주장은 하지 않는다.
- High: 4000 px/m; low: 1000 px/m. 두 resolution은 같은 world와 같은 g/d localization jitter를 사용한다. g/d noise는 1.5 mm로 유지하므로 resolution에 따른 모든 camera error를 모델링한 것은 아니다.
- Independent calibration: family × resolution × 13 b bins × 40 images. 별도의 80 holdout images/family/resolution으로 q-only rounding posterior를 검증한다.
- Full continuous-proxy SMC: factual 4096 particles / 12 MH moves, observation 후 1536 / 12. Factual posterior를 공유하고 dynamic observation posterior도 공유한다. 추가 view는 각각의 descriptor likelihood로 refit한다.
- Acquisition: 512 hypothetical observations, 같은 dynamic-action random numbers. 추가 observation budget 1, stop 제외. View/probe의 물리적 시간·에너지가 같다는 뜻은 아니다.

## Primary / secondary comparisons

1. High-resolution boundary EVSI vs boundary query-value rule. Rule은 validation outcome으로 family×query offset별 결정한다.
2. Boundary always-view vs bbox always-view: policy change 없이 추가 curvature feature의 inference 효과를 확인한다.
3. Boundary EVSI vs bbox EVSI: representation과 policy의 합산 효과.
4. Boundary EVSI vs 같은 boundary posterior를 쓰되 bbox EVSI의 action을 고정한 policy: selection 변화 자체의 효과.
5. Parameter EIG, family별 query-type rule, always-view, uniform random 및 probe-only EVSI를 비교한다.
6. Low-resolution 반복은 secondary robustness analysis다. High vs low를 서로 다른 random worlds로 비교하지 않는다.

## 지표와 주장 범위

MSE, marginal CRPS, marginal central-90% interval coverage, width, any-contact Brier score. CI는 world bootstrap 10,000회, same-world six queries를 내부 평균한다. Rule-fitting uncertainty와 multiple-comparison correction은 포함하지 않은 exploratory interval이다.

Geometry error는 posterior mean의 rounding MAE와 같은 normal angle에 대응한 collision-envelope point RMSE로 평가한다. 후자는 일반적인 Hausdorff distance가 아니다. Truth는 evaluation에만 사용한다. 모든 output은 scene family와 camera calibration이 알려진 synthetic setting에 조건부다.

## Real-video scope

Deform360 008-pink-cloth의 기존 six videos를 이용한다. 20 frames 간격의 모든 sampled frames에서 bounding box, best rounded rectangle, 128-point contour의 image-space representation fidelity를 비교한다. Reference는 image에서 얻은 segmentation mask이며 GT shape가 아니다. HSV threshold perturbation에 대한 boundary sensitivity를 별도로 기록한다. Occluded cloth silhouette를 physical collision surface로 취급하지 않고, 3D metric reconstruction·material inference·held-out physical prediction을 했다고 주장하지 않는다.

## 실행 중 수정 기록 — 2026-11-08

1. **Sensor correctness:** fresh gate case 36에서 T=0.75 이후 collision이 발견되었다. 마지막 frame의 instantaneous velocity로 두 frame의 이동을 외삽하던 sensor를 실제 T=0.75, 0.8의 position rendering으로 수정했다. Dynamic likelihood도 동일한 finite-frame average로 변경했다. Counterfactual target은 T=0.8의 instantaneous velocity다. 기존 partial main/audit를 `main_instantaneous_invalid`, `audit_instantaneous_invalid`로 분리하고 **72개 전부 동일 seed로 재실행**했다. Scene을 제거하지 않았다.
2. **Sensor sensitivity:** test aggregate를 분석하기 전에, g/d에 1.5 mm localization jitter를 주는 가정 자체를 점검하기 위해 `pixel_only` condition을 추가했다. High resolution을 유지하되 pre-render g/d jitter를 0으로 한다. Pixel phase, blur, RGB noise는 남긴다. Calibration과 holdout을 독립적으로 다시 만들고 factual/dynamic evidence는 그대로 공유한다. 이는 secondary sensitivity experiment다.
3. **Acquisition numerics:** sharp geometry likelihood가 finite factual bank의 적은 atoms에 집중할 수 있어 `conditional_view.py`를 추가했다. Geometry를 hypothetical image feature에 조건부로 proposal하고 factual likelihood로 importance weighting한다. 이는 standard conditional importance integration이며 새 알고리즘 기여가 아니다. Main은 64 outer × 512 inner, predetermined validation cases 1/25/49는 128 × 2048로 비교한다. SNIS mean의 squared-error utility에 first-order Monte Carlo variance correction을 적용한 결과와 raw 결과를 함께 저장한다. Original bank EVSI는 원래 protocol 결과로 보존하고, conditional EVSI는 추가 robustness analysis로 명시한다. Dynamic acquisition은 같은 bank estimate를 유지하므로 모든 acquisition이 수렴했다고 주장하지 않는다.
4. **Real-video controls:** overlay 확인 후 axis alignment의 불이익을 분리하기 위해 oriented box, oriented rounded rectangle, best approximate quadrilateral, convex hull을 추가했다. 114개 전체와 frame boundary에 닿지 않는 39개를 별도로 보고한다. 이 비교도 동일 segmentation에 fitting한 representation capacity audit다.

위 수정은 retrospective amendment이며 사전 등록된 protocol로 표현하지 않는다. Primary original EVSI와 secondary conditional EVSI 중 유리한 것만 선택해서 제시하지 않는다.

5. **Validation-driven refinement (12:05 KST):** validation case 25의 pixel-only condition에서 64×512 → 128×2048 변경 시 6개 query 중 2개 action이 바뀌었다. 이에 test aggregate를 확인하기 전에 72 worlds 모두에 128×2048 conditional integration을 추가하기로 결정했다. `conditional_view`의 coarse 결과와 `conditional_refined`의 높은 budget 결과를 둘 다 분석한다. 두 계산은 seed를 공유하지만 tensor dimension이 달라 nested sample은 아니다. 높은 budget도 exact integration으로 부르지 않는다.

6. **Execution scheduling (12:46 KST):** parity workers의 소요 시간 차이로 남은 world가 한 GPU에 몰렸다. 이 project의 두 main process만 중단하고, 완료된 JSON 및 factual NPZ를 검사했다. 완료 62개는 모두 유효했고 incomplete write는 없었다. 남은 case 57/59/61/63/65/67/68/69/70/71은 `queue_run.py`에서 동일한 `run_case`, seeds, calibration, physics, SMC budget으로 재개했다. Atomic directory claim으로 중복 실행을 막고 factual NPZ를 먼저 저장한 뒤 case JSON을 atomic rename한다. 진행 중 중단한 두 world의 partial computation은 성능 결과로 사용하지 않는다. Initial parity runtime files와 resume 기록을 모두 보존한다. Wall time을 method efficiency로 해석하지 않는다.

7. **Post-hoc numerical failure audit (13:00 KST):** 전체 집계 후 high/boundary conditional integration에서 median inner ESS < 50인 test cases 66/68을 확인했다. Test target error가 아닌 ESS로 선택했다. 각 case의 128 hypothetical images 중 ESS가 가장 낮은 3개와 median 위치 1개를 골라, 동일 factual+hypothetical-view evidence로 SMC-4096/18 moves를 두 seeds에서 실행했다(총 16 fits). Hypothetical view는 원래 128×2048 run과 같은 seed로 재현한다. 이 diagnostic은 acquisition score를 교체하거나 test worlds를 제외하는 데 사용하지 않는다. 낮은 ESS가 실제 posterior prediction 불안정과 연결되는지 확인하는 추가 검증이다. Target truth를 읽어 hypothesis를 선택하지 않는다.
