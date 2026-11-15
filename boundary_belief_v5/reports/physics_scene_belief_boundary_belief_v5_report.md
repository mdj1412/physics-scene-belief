# Raw-image boundary와 continuous scene belief: 서버 실험 보고서

실행일: 2026-11-08. 기존 contact_model_v4를 보존하고 별도 boundary_belief_v5 project에서 실행했다.

## 1. 결과와 판단

**Boundary 정보를 보존하면 geometry와 view-conditioned prediction은 좋아진다. Observation selection의 이득은 남아 있는 비교가 있지만, 모든 sensor condition과 strong baseline에 대해 안정적으로 유지된다고 말할 근거는 부족하다. Curvature 정보를 이용해 action을 바꾸는 효과만 분리하면 세 조건 모두 CI가 zero를 포함한다.**

같은 RGB의 bbox descriptor와 boundary descriptor를 같은 continuous-proxy SMC에 연결했다. High condition의 rounding MAE는 1.313→0.409 mm, collision-envelope RMSE는 1.726→1.431 mm로 감소했다. Pixel-only condition에서는 각각 1.292→0.325 mm, 0.941→0.255 mm다. 이 개선은 hidden truth를 geometry baseline에 주지 않고 얻었다.

Original bank EVSI는 high condition에서 query-value rule보다 MSE가 21.6% 낮았다. 하지만 다른 numerical acquisition estimator에서는 이 비교의 CI가 zero를 포함했다. Pixel-only conditional EVSI는 rule보다 평균 21.9% 낮지만 always-view에 대한 차이는 CI가 zero를 포함했다. 더구나 conditional estimator의 일부 hypothetical posterior에서 ESS collapse와 prediction 불안정을 실제 full-SMC 재검증으로 확인했다. 따라서 큰 sample count를 썼다는 이유만으로 수렴한 reference라고 부르지 않는다.

Synthetic physics와 real-video boundary audit는 별개의 검증이다. Real video에서는 boundary representation을 점검했으며 restitution inference나 physical counterfactual accuracy를 검증하지 않았다.

| Sensor condition | Original bank EVSI MSE | Refined conditional EVSI MSE | Query-value rule MSE | Always-view MSE | Conditional EVSI − rule, 95% CI |
| --- | --- | --- | --- | --- | --- |
| high | 0.008788 | 0.009598 | 0.011213 | 0.019111 | -0.001615 [-0.004447, +0.002007] |
| low | 0.008709 | 0.010285 | 0.011213 | 0.021049 | -0.000929 [-0.004311, +0.003168] |
| pixel_only | 0.008585 | 0.008085 | 0.010358 | 0.009734 | -0.002273 [-0.004419, -0.000187] |

Original bank EVSI는 처음 protocol의 estimator다. Conditional integration은 sharp image likelihood의 수치적 문제를 점검하며 추가한 standard importance integration이다. Refined는 128 outer × 2048 inner samples를 사용한다. 유리한 estimator 하나만 남기지 않고 두 결과를 보존했다. MSE unit은 (m/s)²다.

표의 empirical policy score는 실제 선택한 action과 full-SMC update의 결과로 유효하다. 다만 conditional EVSI를 Bayes-optimal action ranking의 수렴된 근사로 해석하는 데에는 §7의 명확한 반례가 있다.

| Condition | Original bank EVSI − query-value rule MSE, 95% CI |
| --- | --- |
| high | -0.002425 [-0.004484, -0.000488] |
| low | -0.002504 [-0.004495, -0.000588] |
| pixel_only | -0.001773 [-0.005442, +0.002004] |

## 2. Raw image에서 무엇을 보존했는가

기존 view는 `[g,d]`만 남겨 outer extent가 같은 sharp/rounded obstacle을 구분하지 못했다. 이번에는 color segmentation의 connected component에서 bbox area와 foreground area의 차이를 추가한다.

`q = (A_bbox − A_component)/(4 − π)`

Continuum의 rounded rectangle에서는 `q=b²`다. Pixel rasterization, blur, thresholding으로 생기는 bias와 covariance는 별도의 rendered calibration images로 추정한다. Boundary descriptor는 `[g_mm, d_mm, q_mm²]`, bbox descriptor는 앞의 두 값이다. 둘 다 같은 RGB에서 추출하며 test truth로 geometry를 보정하지 않는다.

알려진 axis-aligned rounded-rectangle family 안에서는 g,d,b가 obstacle boundary를 결정한다. Sphere-center collision boundary는 obstacle을 sphere radius R로 dilation한 Minkowski sum이다. Core half extent는 `half−b`, corner arc radius는 `R+b`가 된다. 따라서 b가 달라지면 같은 bbox에서도 contact normal과 grazing outcome이 달라진다.

이 descriptor는 임의의 contour 전체에 대한 raw-pixel likelihood가 아니다. Known camera, object color, scene family, obstacle width, axis alignment를 사용한다. General shape에서는 같은 area deficit을 가진 서로 다른 local boundary가 존재한다. 실제 영상에서는 이러한 family 가정을 강요하지 않고 contour와 full signed-distance field를 별도로 보존했다.

Deflection에서는 첫 obstacle에서 추출한 g,d,b를 known layout을 통해 두 번째 obstacle에 공유한다. Gate에서는 두 obstacle의 area deficit을 평균한다. 이번 curvature는 모든 corner에 공유되는 rounding radius b로 표현한다. 자유로운 multi-object geometry나 arbitrary local curvature field를 reconstruction한 설정은 아니다.

## 3. Sensor calibration과 조건

High는 1024×1024, 4000 px/m이며 low는 256×256, 1000 px/m이다. 두 조건은 g,d에 1.5 mm pre-render localization jitter를 공유한다. Pixel-only는 high resolution에서 이 jitter를 제거한다. 모든 조건에 subpixel phase, 2× supersampling, LANCZOS downsampling, Gaussian blur 0.35–0.8 pixel, RGB noise σ=3이 남는다.

Family × condition마다 13 b values × 40 images로 mean curve와 pooled covariance를 calibration했다. Condition당 1,560 calibration images이며 별도의 240 holdout images에서 q만으로 추정한 b posterior를 평가했다. Bbox의 calibration mean은 b에 대해 pooled constant로 두어 calibration noise가 몰래 curvature cue가 되지 않도록 했다.

| Condition | Holdout b MAE [mm] | Central-90% coverage | Mean interval width [mm] |
| --- | --- | --- | --- |
| high | 0.3612 | 90.83% | 1.4424 |
| low | 0.9770 | 89.17% | 3.6552 |
| pixel_only | 0.3329 | 89.58% | 1.3620 |

이 coverage는 q-only synthetic holdout에 대한 결과다. Full scene belief나 real-video uncertainty의 calibration을 보장하지 않는다. 특히 b≈0에서는 q=b²의 sensitivity가 작아 작은 rounding을 분리하기 어렵다. Binning 결과도 원본 summary에 보존했다.

![Curvature calibration](/Users/mindongjun/Documents/Codex/2026-10-30/new-chat/work/physics-scene-belief/boundary_belief_v5/results/analysis/curvature_calibration.png)

## 4. 비교 조건과 실행 범위

Fresh 72 worlds를 만들었다. Corner, gate, deflection마다 24개이며 첫 8개는 validation, 나머지 16개는 test다. Test는 48 worlds, world당 6 queries다. g∈[−8,8] mm, e∈[0.15,0.85], μ∈[0,0.6], d∈[22,34] mm, b∈[0,6] mm를 continuous uniform prior에서 생성하고 같은 prior로 inference했다. 이전 v4의 discrete truth regimes와 seed가 다르므로 v4의 개선율과 이번 개선율을 직접 차감할 수 없다.

Physics는 radius 20 mm sphere와 static rounded rectangles의 2D continuous collision detection이다. Solid-sphere inertia, instantaneous Coulomb impulse와 spin을 포함한다. Initial speed는 1 m/s, horizon은 0.8 s, gravity는 없고 contact event budget은 32다. 같은 rigid law의 세 layout이며 세 종류의 물리계를 검증한 것은 아니다.

Factual evidence는 source offset 0의 x-motion 하나다. 추가 action은 view 또는 source offset −22/+22/+9 mm의 left/right/center probe 중 하나다. Target query는 −30/−17/−7/+4/+15/+28 mm로 source를 옮긴 counterfactual의 T=0.8 velocity다. Factual, dynamic observations, initial posterior, random seeds와 dynamic posterior를 geometry baseline 사이에 공유했다. 각 policy는 선택한 observation 하나만 사용한다. Offline 평가를 위해 네 outcomes를 미리 계산했지만 selector에는 actual outcome이나 truth를 전달하지 않는다.

Adaptive-tempering SMC는 factual 4096 particles, observation 후 full refit 1536 particles, stage당 12 MH moves를 사용한다. Observation 후 작은 bank를 단순 reweighting한 결과를 prediction으로 사용하지 않는다. Same particle budget이며 exact equal-runtime comparison은 아니다. Query-value rule은 validation family×query offset별 MSE 최소 action, query-type rule은 family별 fixed action이다. EIG, always-view, uniform random, probe-only EVSI도 비교했다.

Dynamic RGB는 centroid 측정을 위한 sphere-only rendering이며 전체 scene의 photorealistic video가 아니다. Factual evidence에 geometry appearance를 함께 제공하지 않고 추가 view에서만 제공한다. Initial factual video 자체에서 geometry cue를 함께 추출하는 setting까지 검증한 것은 아니다. 이 정보 접근 가정은 observation selection의 일반화 범위를 제한한다.

Dynamic likelihood는 coordinate당 Gaussian σ=0.025 m/s를 사용한다. 실제 measurement는 두 rendered frames의 color centroid에서 얻으므로 Gaussian likelihood는 quantization을 포함한 sensor의 근사다. 동일한 one-observation budget을 비교하며 stop action은 넣지 않았다. View와 physical probe의 실제 시간·에너지 비용이 같다는 뜻은 아니다.

## 5. Geometry와 prediction의 개선을 분리한 결과

| Condition | Descriptor | b MAE [mm] | Collision-envelope RMSE [mm] | b coverage | Always-view MSE |
| --- | --- | --- | --- | --- | --- |
| high | bbox | 1.3127 | 1.7265 | 97.92% | 0.021079 |
| high | boundary | 0.4094 | 1.4312 | 87.50% | 0.019111 |
| low | bbox | 1.3172 | 1.7486 | 97.92% | 0.021710 |
| low | boundary | 0.7196 | 1.5954 | 91.67% | 0.021049 |
| pixel_only | bbox | 1.2924 | 0.9405 | 97.92% | 0.012535 |
| pixel_only | boundary | 0.3250 | 0.2552 | 87.50% | 0.009734 |

Collision-envelope RMSE는 각 corner에서 같은 normal angle에 대응한 sphere-center boundary point들의 RMSE다. 일반적인 Hausdorff distance가 아니다. Posterior mean geometry로 계산했다. 같은 view action을 고정한 always-view 비교는 selector의 변경 없이 geometry evidence 자체의 prediction 효과를 보여준다.

| Condition | Boundary − bbox always-view MSE, 95% CI | Boundary − bbox collision-envelope RMSE [mm], 95% CI |
| --- | --- | --- |
| high | -0.001968 [-0.003784, -0.000439] | -0.2953 [-0.4455, -0.1505] |
| low | -0.000662 [-0.001686, +0.000362] | -0.1533 [-0.2868, -0.0263] |
| pixel_only | -0.002801 [-0.004710, -0.001154] | -0.6853 [-0.8599, -0.5095] |

![Raw image and reconstructed collision envelope](/Users/mindongjun/Documents/Codex/2026-10-30/new-chat/work/physics-scene-belief/boundary_belief_v5/results/analysis/raw_image_geometry.png)

Validation case 1/25/49를 사용한 시각화다. Colored line은 object silhouette가 아니라 sphere-center collision envelope다. Deflection panel은 첫 obstacle 주변을 보여준다. Test 성능이 좋은 case를 골라 시각화한 것은 아니다.

## 6. Observation selection의 이득

아래는 high condition의 test 결과다. Boundary라는 표현은 curvature feature를 보존한 descriptor를 의미한다. Conditional EVSI/EIG는 view의 acquisition integral을 refined conditional integration으로 계산하고 dynamic actions는 기존 bank estimate를 유지한다.

| Policy | MSE ↓ | CRPS ↓ | Marginal 90% coverage | Mean width |
| --- | --- | --- | --- | --- |
| bbox_conditional_EVSI | 0.009655 | 0.029519 | 93.23% | 0.1721 |
| boundary_conditional_EVSI | 0.009598 | 0.029538 | 93.23% | 0.1733 |
| boundary_value_rule | 0.011213 | 0.031603 | 92.88% | 0.2053 |
| boundary_type_rule | 0.019740 | 0.043564 | 92.71% | 0.2604 |
| boundary_conditional_EIG | 0.012690 | 0.039654 | 94.27% | 0.2407 |
| boundary_always_view | 0.019111 | 0.055314 | 93.92% | 0.3272 |
| boundary_random | 0.022281 | 0.050388 | 93.10% | 0.2936 |
| boundary_probe_EVSI | 0.010800 | 0.030396 | 93.23% | 0.1699 |
| boundary_conditional_frozen_bbox_policy | 0.009665 | 0.029565 | 93.23% | 0.1724 |

MSE는 vx/vy squared error의 평균, CRPS는 두 marginal의 평균이다. Coverage는 scalar central-90% intervals의 empirical coverage이며 joint region coverage가 아니다. No-observation reference의 MSE는 0.050812다. 추가 observation budget이 다르므로 main ranking에는 넣지 않았다.

| Condition | EVSI − EIG MSE, 95% CI | EVSI − always-view MSE, 95% CI | EVSI − frozen bbox policy MSE, 95% CI |
| --- | --- | --- | --- |
| high | -0.003092 [-0.006603, -0.000109] | -0.009513 [-0.015456, -0.004470] | -0.000067 [-0.000340, +0.000193] |
| low | -0.009304 [-0.015903, -0.003283] | -0.010764 [-0.016818, -0.005525] | -0.000308 [-0.003699, +0.002742] |
| pixel_only | -0.001649 [-0.004307, +0.001477] | -0.001649 [-0.004307, +0.001477] | -0.000485 [-0.001175, +0.000118] |

Frozen bbox policy는 bbox가 선택한 action을 고정하고 posterior update만 boundary descriptor로 수행한다. 이 비교에서의 차이가 추가 curvature feature로 인해 **action을 변경한 효과**다. Boundary EVSI와 bbox EVSI의 단순 차이는 representation과 selection의 효과를 함께 포함하므로 같은 뜻이 아니다.

CI는 world 내부의 여섯 queries를 평균한 뒤 48 worlds를 10,000회 bootstrap한 paired difference다. 음수는 앞 method가 유리함을 뜻한다. Rule fitting uncertainty, multiple-comparison correction, inference Monte Carlo uncertainty를 모두 포함한 확정적 검정으로 해석하지 않는다.

| Condition | Combined effect: boundary EVSI − bbox EVSI | Representation effect: frozen − bbox EVSI | Selection effect: boundary EVSI − frozen |
| --- | --- | --- | --- |
| high | -0.000057 [-0.000340, +0.000236] | +0.000010 [-0.000033, +0.000059] | -0.000067 [-0.000340, +0.000193] |
| low | -0.000331 [-0.003726, +0.002717] | -0.000023 [-0.000076, +0.000020] | -0.000308 [-0.003699, +0.002742] |
| pixel_only | -0.001049 [-0.002100, -0.000143] | -0.000564 [-0.001338, +0.000042] | -0.000485 [-0.001175, +0.000118] |

세 열은 모두 MSE paired difference와 95% CI다. Mean에서는 combined effect = representation effect + selection effect가 정확히 성립한다. 각각의 CI endpoints를 더하는 것은 올바르지 않다.

| Condition | BBox conditional EVSI action counts | Boundary conditional EVSI action counts | Changed actions / 288 |
| --- | --- | --- | --- |
| high | [14, 71, 77, 126] | [19, 69, 76, 124] | 7 |
| low | [16, 72, 76, 124] | [14, 72, 76, 126] | 6 |
| pixel_only | [70, 53, 63, 102] | [89, 42, 62, 95] | 19 |

Action 순서는 view/left/right/center다. 거의 zero-gain query의 tie도 raw count에 포함된다. Action 다양성 자체를 성능 이득으로 세지 않는다. 같은 query offset에서 world별 선택이 달라지는 상세 count는 `summary.json`의 `conditional_by_query`에 보존했다.

![Observation policy comparison](/Users/mindongjun/Documents/Codex/2026-10-30/new-chat/work/physics-scene-belief/boundary_belief_v5/results/analysis/policy_comparison.png)

Figure의 full은 boundary descriptor를 뜻하며 full raw-pixel likelihood라는 뜻이 아니다. Error bar는 policy별 mean의 world-bootstrap CI다. Paired comparison은 위 difference table로 판단한다.

| Condition | Family (16 test worlds) | Conditional EVSI MSE | Query-value rule MSE | Always-view MSE |
| --- | --- | --- | --- | --- |
| high | corner | 0.000657 | 0.000726 | 0.007028 |
| high | gate | 0.022221 | 0.026051 | 0.030242 |
| high | deflection | 0.005917 | 0.006863 | 0.020062 |
| low | corner | 0.000657 | 0.000726 | 0.007596 |
| low | gate | 0.021367 | 0.026051 | 0.030702 |
| low | deflection | 0.008830 | 0.006863 | 0.024848 |
| pixel_only | corner | 0.000573 | 0.000726 | 0.003262 |
| pixel_only | gate | 0.020932 | 0.026051 | 0.022264 |
| pixel_only | deflection | 0.002749 | 0.004296 | 0.003676 |

## 7. Numerical audit와 실행 중 발견한 오류

Fresh gate case 36에서 T=0.75 이후 collision이 발견되었다. 기존 sensor는 마지막 frame의 instantaneous velocity를 뒤로 외삽해 이전 frame을 만들고 late collision을 거부했다. 이를 실제 T=0.75/0.8 position을 rendering하도록 고쳤다. Likelihood도 `(p(0.8)−p(0.75))/0.05`를 예측하도록 수정했다. Prediction target인 T=0.8 instantaneous velocity와 observation의 finite-frame average를 구분한다.

오류가 있던 partial run은 `main_instantaneous_invalid`, `audit_instantaneous_invalid`로 분리했다. Case 36을 삭제하지 않고 **72 worlds를 같은 seed로 전부 재실행**했다. Analysis는 `measurement_model=finite_frame_50ms`를 확인한다. Late-contact regression check와 geometry-feature/dynamic-acquisition invariance check를 통과했다. 이전 v4 결과를 이번 finite-frame 모델로 소급 재검증한 것은 아니다.

Sharp image likelihood에서는 factual particles 중 소수만 큰 weight를 받아 acquisition이 부정확할 수 있다. 이를 점검하기 위해 hypothetical image에 조건부로 g,d,b를 제안하고 e,μ는 prior에서 뽑은 뒤 factual likelihood로 importance weighting하는 standard conditional integration을 구현했다. Prior bounds의 truncation normalizer, b quadrature normalizer와 prior volume을 포함한다. Independent Gaussian rectangle quadrature와 비교한 normalization의 최대 relative error는 0.0335%였다. 이 검사는 proposal algebra에 대한 검증이며 physics posterior convergence를 증명하지 않는다.

64×512에서 시작했으나 validation case 25의 pixel-only condition에서 높은 budget으로 action 2/6개가 바뀌어, test aggregate를 보기 전에 전체 72 worlds에 128×2048을 추가했다. SNIS mean의 squared utility에 first-order Monte Carlo variance correction을 적용했고 raw utility 결과도 저장했다. 표준 수치 적분의 보강이며 새로운 inference algorithm으로 제안하지 않는다.

| Condition | Bank EVSI | Conditional 64×512 | Conditional 128×2048 | Refined − coarse MSE, 95% CI |
| --- | --- | --- | --- | --- |
| high | 0.008788 | 0.009880 | 0.009598 | -0.000281 [-0.003705, +0.002790] |
| low | 0.008709 | 0.009899 | 0.010285 | +0.000385 [-0.003312, +0.003845] |
| pixel_only | 0.008585 | 0.008447 | 0.008085 | -0.000362 [-0.001227, +0.000444] |

Predetermined validation case 1/25/49에서 factual SMC-8192 두 seeds, view SMC-3072/18 moves 두 seeds로 확인했다. 다음 prediction RMSE는 main posterior mean과 audit posterior mean 사이의 차이이며 truth에 대한 error가 아니다.

| Case | Condition | Main b mean [mm] | Audit b means [mm] | Prediction mean disagreement [m/s] |
| --- | --- | --- | --- | --- |
| 1 | high | 2.777 | 2.777/2.792 | 0.0036/0.0017 |
| 1 | low | 1.663 | 1.674/1.679 | 0.0009/0.0053 |
| 1 | pixel_only | 2.612 | 2.586/2.612 | 0.0015/0.0009 |
| 25 | high | 0.561 | 0.553/0.552 | 0.0082/0.0089 |
| 25 | low | 0.800 | 0.777/0.791 | 0.0068/0.0130 |
| 25 | pixel_only | 0.557 | 0.554/0.550 | 0.0044/0.0057 |
| 49 | high | 1.953 | 1.921/1.934 | 0.0065/0.0089 |
| 49 | low | 3.411 | 3.471/3.489 | 0.0025/0.0113 |
| 49 | pixel_only | 2.115 | 2.116/2.096 | 0.0013/0.0003 |

![Acquisition convergence](/Users/mindongjun/Documents/Codex/2026-10-30/new-chat/work/physics-scene-belief/boundary_belief_v5/results/analysis/acquisition_convergence.png)

높은 conditional budget도 exact integration이 아니다. Outer distribution은 factual SMC bank에 의존하고 dynamic acquisition은 bank estimate를 유지한다. 소수 validation cases에서의 안정성을 전체 posterior의 보증으로 일반화하지 않는다. GPU processes가 일부 겹쳐 실행되어 wall time은 엄밀한 효율성 비교에 적합하지 않다. 실제 forward trajectory count를 결과에 기록했다.

**집계 후 확인한 실패:** high/boundary condition에서 test case 66의 median inner ESS는 9.51/2048, case 68은 34.5/2048였다. 최솟값은 약 1이었다. Pixel-only에서도 case 66의 median은 83.0/2048이었다. Validation case 1/25/49가 상대적으로 안정적이었다는 사실로 이 실패를 덮을 수 없다.

High/boundary의 median ESS < 50인 cases 66/68을 골라, 각 case에서 ESS가 가장 낮은 hypothetical views 세 개와 median 위치 한 개를 full SMC-4096/18 moves의 두 seeds로 다시 fit했다. 총 16 fits다. 이 선택에는 test target truth를 사용하지 않았다. Main action이나 test score를 이 diagnostic으로 교체하지 않았다.

| Case | Hypothetical view index | Inner ESS / 2048 | Conditional vs SMC prediction-mean RMSE [m/s] | Between SMC seeds RMSE [m/s] |
| --- | --- | --- | --- | --- |
| 66 | 110 | 1.00 | 0.27049/0.26507 | 0.00641 |
| 66 | 107 | 1.21 | 0.04827/0.04843 | 0.00196 |
| 66 | 82 | 1.95 | 0.09156/0.09336 | 0.00233 |
| 66 | 50 | 9.63 | 0.03359/0.03200 | 0.00247 |
| 68 | 110 | 1.38 | 0.05974/0.06183 | 0.00249 |
| 68 | 27 | 3.52 | 0.04878/0.04512 | 0.00429 |
| 68 | 81 | 4.74 | 0.11786/0.12321 | 0.00586 |
| 68 | 68 | 35.32 | 0.01162/0.01137 | 0.00163 |

Case 66의 최저-ESS view에서 conditional estimate와 full-SMC reference의 prediction mean 차이는 0.265–0.270 m/s였다. 두 SMC seeds 사이의 차이는 0.00641 m/s였다. 즉 단순히 ESS가 작아 보인다는 우려를 넘어, 해당 hypothetical posterior의 predictive integral이 불안정하다는 실증적 증거를 얻었다. 두 SMC fits도 exact posterior proof는 아니지만 같은 근사를 다시 계산한 결과보다 훨씬 일치한다.

해석상 원인은 image-conditioned geometry proposal 뒤에 material parameters를 prior에서 뽑는 구조가 factual evidence의 얇은 joint support를 충분히 덮지 못하는 데 있다. Proposal normalization 검사는 통과했으므로, 이 결과를 normalization formula의 오류와 혼동하면 안 된다. 추가 computation을 단순히 inner sample count로 늘리는 것만으로 joint geometry/material ambiguity가 해결된다고 가정할 수 없다.

![Conditional acquisition numerical failure](/Users/mindongjun/Documents/Codex/2026-10-30/new-chat/work/physics-scene-belief/boundary_belief_v5/results/analysis/conditional_failure.png)

이 diagnostic은 posterior mean 사이의 disagreement이며 target truth에 대한 prediction error가 아니다. ESS로 고른 어려운 hypothetical views의 결과이므로 전체 views의 평균 오류로 일반화하지 않는다.

## 8. 실제 영상에서 검증한 범위

[Deform360 official dataset](https://huggingface.co/datasets/brownu/deform360/blob/main/README.md)의 기존 local subset을 사용했다. Dataset은 deformable-object의 multi-view visuotactile recordings를 제공한다. 이번에 사용한 파일은 `008-pink-cloth`의 lift-corner/edge/center 세 interactions와 두 cameras, 총 여섯 videos다. Pinned revision은 `5ea8c5d3fc7b4a7b4f9f921f2ceb1de24610f6a4`다. 새로운 dataset을 임의로 추가해 결과를 섞지 않았다.

2,214 frames의 videos에서 20-frame 간격으로 얻은 114 sampled frames를 평가했다. Largest color component의 128-point arc-length contour와 full signed-distance field를 저장했다. Local curvature는 smoothed image-space contour에서 계산하며 pixel unit이다. Bbox, axis-aligned/oriented rounded rectangle, oriented box, approximate quadrilateral, convex hull을 동일 canonical segmentation에 fit했다.

| Representation | All 114 frames IoU | Not-cropped 39 frames IoU |
| --- | --- | --- |
| Axis bbox | 0.6459 | 0.6826 |
| Axis rounded rectangle | 0.6908 | 0.7400 |
| Oriented box | 0.6704 | 0.7005 |
| Oriented rounded rectangle | 0.7044 | 0.7525 |
| Approximate quadrilateral | 0.8108 | 0.7974 |
| Convex hull | 0.8922 | 0.8851 |
| 128-point contour | 0.9887 | 0.9862 |

Contour의 symmetric boundary-distance p95를 frame별로 구한 뒤 평균하면 3.52 px이며 bbox는 115.49 px다. 하지만 reference가 같은 segmentation이므로 이는 representation fidelity다. 128-point contour의 높은 IoU는 true object boundary의 accuracy나 novel algorithm의 우월성을 뜻하지 않는다. 더 많은 shape degrees of freedom을 썼다는 효과도 포함한다.

75/114 frames는 mask가 frame edge에 닿는다. HSV threshold perturbation의 mean IoU는 0.9878이지만, variant별 boundary-p95 거리의 95th percentile은 25.19 px다. 평균 IoU가 높아도 local boundary가 크게 움직일 수 있다. Overlay에서는 일부 hand/gripper contamination도 관찰된다. Occlusion boundary, physical contact boundary, material surface를 동일시할 수 없다.

![Real-video boundary representations](/Users/mindongjun/Documents/Codex/2026-10-30/new-chat/work/physics-scene-belief/boundary_belief_v5/results/real_boundary/boundary_overlays.jpg)

완료한 것은 real RGB에서 boundary 정보를 추출·보존하고 단순 geometry compression의 손실과 segmentation sensitivity를 측정하는 단계다. 이 cloth에 synthetic rigid collision law를 적용하지 않았다. 3D metric contact surface, restitution, material stiffness, real intervention의 counterfactual accuracy는 이번 결과로 검증하지 못했다.

## 9. 연구 방향에 대한 판단

현재 가장 강한 결과는 **raw image의 boundary information을 버리지 않는 baseline의 필요성**이다. High와 pixel-only의 always-view 비교에서 prediction 개선도 확인했다. 반면 geometry가 좋아졌다는 사실을 곧바로 새로운 observation selector의 기여로 옮길 수는 없다. High condition에서 boundary conditional EVSI와 bbox conditional EVSI의 MSE는 0.009598/0.009655로 거의 같고, action 변경 자체의 차이는 −0.000067이다. Pixel-only의 combined improvement는 −0.001049, CI [−0.002100, −0.000143]이지만, 이를 representation과 selection으로 나누면 각 CI는 zero를 포함한다.

Sensor 조건에 따른 결론도 다르다. High/low에서 conditional EVSI는 always-view보다 평균 약 50% 낮은 MSE를 보이지만 rule에 대한 CI는 zero를 포함한다. Pixel-only에서 conditional EIG는 모든 test world에서 view를 선택해 always-view와 같아졌다. 이때 EVSI의 평균 MSE는 더 낮아도 EIG/always-view 대비 CI는 zero를 포함한다. Low-resolution deflection에서는 conditional EVSI가 query-value rule보다 오히려 나빴다. 이 차이를 전체 평균만으로 가리지 않겠다.

따라서 현재 결과를 그대로 “새로운 active physical reasoning algorithm의 우월성”으로 묶어 논문을 쓰는 것은 이르다. Area-deficit geometry, continuous SMC, EVSI, conditional importance integration 각각을 새로운 알고리즘이라고 주장할 수 없다. 실제 영상에서는 아직 physical prediction이 연결되지 않았다. Query-value rule도 family당 validation worlds 8개로 정한 rule이므로 모든 learned policy보다 강한 baseline이라고 부를 수 없다.

다음 우선순위는 **joint posterior를 보존하면서 acquisition ranking의 신뢰성을 확인하는 것**이다. Factual posterior에서 hypothetical view likelihood를 점진적으로 추가하는 tempered SMC bridge와 nested full SMC를 reference로 두고, 현재 conditional proposal의 실패를 먼저 해결해야 한다. 두 방식 자체도 standard inference baseline이다. 그 뒤에 contact boundary 부근의 local geometry uncertainty를 어떤 해상도로 유지해야 query risk와 action ranking이 보존되는지 연구하는 편이 현재 증거와 맞는다.

이 방향을 실제 기여로 발전시키려면 같은 total forward-evaluation budget에서 full boundary representation, continuous SMC, 더 큰 validation set으로 학습한 policy와 비교해야 한다. 또한 initial factual video의 geometry cue를 모든 방법에 공평하게 제공하고, 여러 corner가 독립적으로 변하는 held-out shapes와 camera/occlusion uncertainty를 넣어 known rounded-rectangle family를 넘어가야 한다. Real-video 단계에서는 contour IoU를 physical validation으로 대신하지 말고, initial state와 interventions가 명확한 rigid-object recordings에서 held-out trajectories를 검증해야 한다. 현재 cloth videos는 그 목적의 reference가 아니다.

계속 진행할 판단 기준은 구체적이다. Acquisition convergence를 개선한 뒤에도 frozen-policy comparison에서 action 변경의 이득이 확인되는지 먼저 본다. 이득이 유지되지 않으면 observation selection의 새로운 우월성 주장보다 geometry uncertainty의 측정·calibration 문제로 범위를 좁히는 것이 타당하다. 이번 실험은 이 판단을 할 수 있도록 baseline과 numerical failure를 드러낸 단계다.

## 10. 재현과 산출물

Remote project: `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/boundary_belief_v5`

Dataset subset: `/scratch2/mdj1412/datasets/deform360_contact_belief_audit`

Slurm allocations 2676322(node55), 2676323(node56)의 RTX PRO 6000 Blackwell 두 GPU에서 float64 simulation/SMC를 실행했다. 기존 다른 allocations와 pending jobs는 변경하지 않았다. Neural model training은 하지 않았다. 기존 venv의 Python 3.9.18, NumPy 1.26.4, SciPy 1.13.1, PyTorch 2.7.1+cu128, Pillow 11.3.0, OpenCV 4.10.0을 사용했다.

Main seed는 `2026110800+case`, image seed는 `202611088000+case`, factual SMC는 `150100+case`, refit은 `160100+case`, acquisition은 `170100+case`다. Main 72×8 + pixel-only 72×2 = 720 completed SMC fits, 별도 validation audit 24 fits와 ESS failure audit 16 fits를 분석했다. 중단·재시작한 partial computation은 이 completed-fit 수에 넣지 않았다. Calibration/holdout은 condition마다 독립 seed를 사용했다.

[실행 안내](/Users/mindongjun/Documents/Codex/2026-10-30/new-chat/work/physics-scene-belief/boundary_belief_v5/README.md), [protocol과 amendments](/Users/mindongjun/Documents/Codex/2026-10-30/new-chat/work/physics-scene-belief/boundary_belief_v5/PROTOCOL.md), [geometry 구현](/Users/mindongjun/Documents/Codex/2026-10-30/new-chat/work/physics-scene-belief/boundary_belief_v5/geometry.py), [continuous SMC](/Users/mindongjun/Documents/Codex/2026-10-30/new-chat/work/physics-scene-belief/boundary_belief_v5/belief.py), [conditional integration](/Users/mindongjun/Documents/Codex/2026-10-30/new-chat/work/physics-scene-belief/boundary_belief_v5/conditional_view.py), [전체 수치](/Users/mindongjun/Documents/Codex/2026-10-30/new-chat/work/physics-scene-belief/boundary_belief_v5/results/analysis/summary.json), [convergence audit](/Users/mindongjun/Documents/Codex/2026-10-30/new-chat/work/physics-scene-belief/boundary_belief_v5/results/analysis/convergence.json)를 함께 제공한다. Raw images, factual particle banks, real-video per-frame SDF/contour는 remote project에 보존했다. [Server artifact manifest](/Users/mindongjun/Documents/Codex/2026-10-30/new-chat/work/physics-scene-belief/boundary_belief_v5/server_run_manifest.json)는 source와 결과의 SHA-256을 기록한다.
