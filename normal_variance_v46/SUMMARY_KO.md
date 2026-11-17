# v46 연구 보고서 · 접촉 전 영상의 surface normal variance로 projection을 선택할 수 있는가

**새 contour family에서는 위치와 normal variance를 함께 쓴 규칙이 always-original에 졌다.** Terrace와 dimple의 12개 새 catalog에서 held-out Brier loss는 always-original **0.155170**, v45 위치 전용 **0.162776**, 위치와 분산을 함께 쓴 joint **0.162760**이었다. 분산을 더했을 때 위치만 쓴 경우보다 줄어든 loss는 **0.000016**이고, 사전 지정한 세 비교의 동시 95% 구간은 0을 포함했다. 반면 always-original 대비 손해 **+0.007589**는 두 family에서 모두 확인됐다. 독립 PyBullet에서도 같은 순서였다. [Primary 결과](score46.json) · [독립 solver 결과](score46_bullet.json)

실패의 단서는 **source에서 projection이 유리했던 geometry가 target에서는 손해였다는 것**이다. 예를 들어 31 mm 위치에서 source의 세 번째 variance 사분위는 projection의 Brier가 0.00069 줄 것으로 학습됐지만, 새 물체에서는 같은 구간에서 0.02862 늘었다. 이 구간 하나에 전체 target 영상의 12.90%가 속했다. 영상이 위치를 잘 찾더라도 조건별 손익의 방향이 바뀌면 잘못된 decoder를 고르게 된다. [사전 lookup](router46.json) · [조건별 실측 손익과 빈도](transfer46.json)

**같은 영상 선택을 유지하고 새 형상에 맞는 projection axis만 주자 초과 손실의 약 90.4%가 줄었다.** 그래도 Brier는 0.155897로 always-original보다 높았다. 이 대조는 변형 쌍과 family를 아는 사후 진단이며, 구현한 영상 정책의 배포 성능에는 포함하지 않았다. [축만 교체한 결과](aligned_axis_score46.json)

![새 형상의 손익과 source 사분위별 손익 전이](normal_variance_v46.png)

그림 A의 양수는 joint에서 Brier가 더 높았다는 뜻이다. 막대는 두 family × 세 비교에 대한 동시 95% 구간이다. B의 검은 선은 source에서 shrinkage 후 학습한 손익이고, 점은 같은 구간의 실제 target 손익이다. 점 크기는 해당 위치에서의 target 관측 빈도를 나타낸다.

## 1. 실험의 대상과 정보 예산

- **두 새 family.** Terrace는 완만하게 올라갔다 내려오는 비주기 quintic shelf, dimple은 폭과 깊이가 다른 두 Gaussian depression이다. 세 variant에서 위치·폭 또는 깊이가 달라진다. Source의 fold·ripple과 별도로 식을 정하고 target 충돌 전에 고정했다. 공통 latent slope와 curvature를 더했고, 미분값을 finite difference와 대조했다. [형상과 neutral fixture](PHYSICS_PLAN_KO.md)
- **동일 물체의 비교.** 신규 support 600–611의 catalog 12개에서 환경당 128개씩 6,144회 추출했다. 같은 support 내 중복을 제외한 prototype은 5,976개다. 각 물체에서 2 family × 2 surface × 2 law × 3 variant × 2 nominal height(19, 31 mm)를 교차하고, 세 offset(−0.5, 0, +0.5 mm), 네 sensor noise를 짝지었다. 독립 통계 단위는 catalog 12개이며 두 family는 고정한 평가 대상이다. [GPU archive 검산](audit_archive.json)
- **손실의 의미.** Original은 구매한 7점 query profile의 모든 정보를 쓴다. Projection은 같은 profile에서 v45의 source axis 하나를 제거한다. 두 방법 모두 같은 factual evidence와 이미 선택한 paid sensor를 받고 미래 stick 확률을 예측한다. Deployment probe를 산 물체에서 두 예측은 일치한다. Brier는 예측 확률과 binary stick proxy의 제곱오차다. 위치와 분산은 어느 decoder를 호출할지만 정한다. 새 family 이름, 물체 ID, contact law, pressure radius, target label은 배포 정책의 입력에서 제외했다. [Frozen decoder 경로](runner46.py) · [GPU 검증](GPU_KO.md)
- **Pressure와 rigid 대조.** Primary는 처음 닿는 위치와 총 normal impulse를 맞춘 narrow/broad pressure(peak radius 0.25/2 mm)의 균등 평균이다. Point까지 포함한 2,654,208번의 충돌 조건을 계산했다. 같은 물체의 별도 rigid sphere PyBullet 충돌은 884,736회다. 두 engine 모두 잔류 접선 속도 0.020 m/s 이하를 stick proxy로 쓴다. 시뮬레이션에서 정의한 gross slip이며 실제 pad의 partial slip은 측정하지 않았다. [Pressure·rigid 검산](PHYSICS_RESULTS_KO.md)
- **동결 순서.** 서버에서 분석 계획, physics, 영상 estimator와 source lookup을 동결한 뒤 2026년 11월 16일 19:39:44 KST에 target outcome 생성을 허용했다. Source에서 predictor 간 상대 Brier를 학습할 때는 완료된 v45 물체만 사용했다. 같은 source로 다시 학습한 x-only, 정확한 geometry로 source부터 다시 학습한 oracle, 독립 rigid 검증도 사전 지정했다. [분석 계획](PLAN_KO.md) · [Router 동결](router_freeze46.sha256) · [Target 시작 기록](TARGET_GO)

![두 새 형상의 pre-impact frame과 analytic profile 변형](families_v46.png)

왼쪽은 새 물체를 그린 실제 실험 입력 세 frame이다. 오른쪽은 공통 물체 기울기를 더하기 전 각 family의 형상 함수이며, 음영은 두 명목 위치에서 normal을 측정한 ±2 mm 범위다. 실물 카메라 영상은 이번 데이터에 포함되지 않는다.

## 2. 영상에서 규칙까지

1. **위치를 추정한다.** V45와 같은 96×32 grayscale frame 세 장, 같은 rail fiducial 보정, 같은 tip 추적기를 사용했다. Tip의 접촉 전 경로를 surface 경계까지 외삽해 x를 얻었다. 모든 visual 방법은 동일한 영상 bytes를 받으며 target에서만 221,184개 clip이다.
2. **주변 법선의 변화를 잰다.** 첫 frame에서 명암과 부분 pixel coverage를 이용해 silhouette의 경계를 추정했다. 추정 x 전후 2 mm에서 0.5 mm 간격으로 높이 9개를 읽고, 1 mm 길이의 겹치는 secant 7개로부터 normal의 각도를 얻었다. 통계량은 이 일곱 각도의 population variance(rad²)다. 단일 2D 외곽이 보이는 평면의 법선 변화량이며 3D normal 전체나 측정 오차의 분산으로 해석하지 않는다. [영상 구현과 측정 단위](camera46.py) · [영상 감사](VISION_KO.md)
3. **이 분산에서 projection의 기대 손익을 학습한다.** V45의 같은 영상·같은 물체에서 D = Brier(projected)−Brier(original)을 계산했다. 여섯 x bin을 구성하고, 각 bin의 source 영상만으로 variance를 사분위 네 개로 나눴다. 각 구간의 평균 D를 해당 x 전체 평균 쪽으로 25% shrink했다. 예측 D<0인 구간에서만 projection을 선택한다. Refit-x는 동일한 source·목표·입력 x로 각 x bin의 평균 D만 학습한다. 두 방법의 차이가 variance의 추가 기여다. [학습과 source 검증](fit_route46.py) · [동결한 수치](router46.json)
4. **규칙의 실제 차이를 확인한다.** Source 전체로 학습한 refit-x의 최종 결정은 고정 v45 규칙과 같았다. 30.5·31.0 mm에서만 projection한다. Joint는 30.5 mm의 Q2와 31.0 mm의 Q1·Q2에서 projection을 취소한다. 다른 구간은 그대로다. 이 세 구간에 포함된 target 영상은 단 **0.4078%**였다. V45의 32.938%와 joint의 32.530%라는 projection 사용률 차이가 작았던 이유다. [구간별 빈도와 검산](transfer46.json)

## 3. Held-out Brier loss와 같은 GPU의 p90

아래 Brier는 각 조건과 catalog를 동일 가중한 값이다. Latency는 별도 단일 RTX PRO 6000 Blackwell에서 512개 물체·sensor noise 1회·warm 상태에 측정했다. 각 방법에서 요구하는 계산만 수행했다. Visual에는 raw frame의 host→GPU 전송, 위치 또는 variance 추출, 선택 및 한 개의 posterior 계산이 들어간다. 카메라 촬영, simulator rendering·target 생성, 실물 probe·reset 비용은 별도다. [GPU 측정](GPU_KO.md)

| 정책 | Finite pressure Brier ↓ | Rigid PyBullet Brier ↓ | Primary GPU p90, ms [95% CI] |
|---|---:|---:|---:|
| Always-original | **0.155170** | **0.161305** | **465.64** [465.23, 470.71] |
| V45 위치 전용 | 0.162776 | 0.169248 | 471.35 [470.51, 476.07] |
| 동일 source로 refit한 위치 전용 | 0.162776 | 0.169248 | 471.10 [470.32, 475.33] |
| 위치 + normal variance | 0.162760 | 0.169232 | 473.53 [471.97, 485.91] |

| Joint에서 비교 방법을 뺀 값 | Finite pressure Brier 차이 [동시 95% CI] | Rigid Brier 차이 [동시 95% CI] |
|---|---:|---:|
| Joint − always-original | **+0.007589** [+0.006977, +0.008202] | **+0.007927** [+0.007308, +0.008547] |
| Joint − v45 위치 전용 | −0.0000164 [−0.0000330, +0.0000003] | −0.0000159 [−0.0000329, +0.0000011] |
| Joint − refit 위치 전용 | −0.0000164 [−0.0000330, +0.0000003] | −0.0000159 [−0.0000329, +0.0000011] |

Finite pressure의 **12/12 catalog에서 always-original이 joint보다 유리**했다. Joint는 10/12에서 위치 전용보다 소폭 낮은 loss를 보였지만 사전 기준의 동시 구간은 0을 포함했다. 두 위치 baseline이 동일하게 행동해 Brier도 일치했다. Terrace에서 joint−original은 +0.012403 [여섯 비교 동시 95% CI +0.011545, +0.013261], dimple은 +0.002776 [+0.002044, +0.003508]이었다. Rigid에서도 각각 +0.013199와 +0.002655였다. [Family별 결과와 원자료](score46.json) · [Rigid 재채점](score46_bullet.json)

**비용.** Joint−always-original의 p90 차이는 **+7.89 ms** [paired 95% CI +1.73, +20.56], joint−v45 위치 전용은 **+2.18 ms** [−3.10, +14.49]였다. 전송과 feature 추출·선택만의 p90은 joint 4.60 ms, 위치 전용 3.15 ms였다. 같은 GPU에서 순서를 다시 무작위화한 두 번째 crossover에서도 always-original 470.76, v45 위치 475.67, joint 478.18 ms였다. 위치 전용 위에 더해지는 전체 지연시간의 정확한 크기는 두 측정 모두 오차가 크다. 전체 영상에 대해 GPU와 CPU의 route 불일치는 0건이었고, 선택한 decoder의 확률 replay 최대 오차는 4.45×10⁻¹⁶ 이하였다. [Primary timing](audit_latency46.json) · [실제 pixel의 GPU replay](route_replay46.json)

**더 보수적인 위치 규칙.** 기존 v45의 별도 source-matched 규칙은 30.5 mm에서만 projection한다. 이 규칙의 finite pressure Brier는 0.158414로, joint보다 0.004346 [두 solver 동시 95% CI 0.004062, 0.004630] 낮았다. 같은 x bin에서 joint의 선택률을 보존하고 선택 대상만 무작위로 섞은 경로도 joint보다 평균 Brier가 0.000074 낮았다 [보조 95% CI 0.000062, 0.000086]. 분산을 보고 특별히 제외한 물체들이 단순히 같은 비율로 제외할 때보다 좋은 선택이었다는 근거도 얻지 못했다. [보수적 규칙 검산](conservative46.json) · [Within-bin shuffled control](score46.json)
