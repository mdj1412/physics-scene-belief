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

## 4. 왜 source의 이득이 새 family로 옮겨가지 않았나

**Source에서의 개선은 실제였지만 family를 바꾸자 사라졌다.** V45 안에서 catalog 하나를 통째로 제외하고 나머지 11개에서 구간 경계와 risk를 다시 학습하면, joint−refit x는 평균 −0.000220 [기술적 95% CI −0.000288, −0.000152]였다. 반면 fold만 학습해 ripple을 시험하거나 그 반대로 시험하면 영상 기반 joint와 refit-x가 같은 물체에서 같은 선택을 했다. Source의 단순 catalog split이 새 family에 대한 검증을 대신하지 못했다. [Source 교차검증](source_cv46.json)

**광학적으로 측정한 신호와 상대 손익은 달랐다.** 새 family에서 위치 MAE는 0.09333 mm였다. 영상 분산과 같은 길이의 해석적 secant로 구한 참 분산 사이의 Spearman ρ는 terrace 0.672, dimple 0.645였다. 전체 절대 분산 오차의 median은 0.00160 rad²이지만 p90은 0.17765 rad²였다. 사후에 동일 물체 55,296개 장면을 독립 camera nuisance로 재촬영했을 때 분산 순위의 재현성은 ρ=0.485·0.443까지 낮아졌다. 따라서 법선 변화의 일부를 영상으로 읽을 수 있지만, 특히 꼬리 오차가 크며 새 family의 loss 차이를 맞춘다는 근거는 별개다. [광학 검산](vision_audit46.json) · [독립 재촬영](repeat_camera46.json)

**측정 오차를 없애도 always-original은 이기지 못했다.** Source 단계부터 정확한 x와 해석적 normal variance를 사용한 oracle joint의 finite Brier는 **0.161492**였다. 정확한 x만 쓴 oracle의 0.162889보다 −0.001397 [두 진단 동시 95% CI −0.001543, −0.001250] 개선됐지만, always-original의 0.155170보다 여전히 높다. 이 차이에는 feature의 측정 정밀도와 해당 feature로 source에서 다시 학습한 표의 변화가 함께 포함돼 있다. 또 target의 11개 catalog에 label 접근을 허용하고 같은 source bin·shrinkage로 다시 fit하면, 남은 catalog에서 x-only와 joint 모두 always-original의 Brier와 같았다. 이 진단은 같은 target family의 label을 쓴 별도 실험이다. Source만으로 가능한 운영 성과로 계산하지 않았다. [Oracle 및 원 점수](score46.json) · [Leave-one-target-catalog-out](transfer46.json)

**실제로 지워지는 profile 방향이 달라졌다.** Source axis와 실제 hollow↔bulge 차이 벡터의 squared cosine을 계산했다. V45의 fold·ripple에서 이 축은 변형 신호 energy의 67.93%(19 mm)·58.28%(31 mm)를 제거했다. 새 terrace에서는 3.39%·11.23%, dimple에서는 19.19%·**0.036%**였다. 즉 같은 연산이 새 물체의 같은 종류의 변형을 제거한다는 가정이 깨졌다. [Paired profile의 각도 검산](axis_transfer46.json)

**축만 교체한 직접 대조.** 이 관찰이 손실에도 영향을 주었는지 확인하기 위해 primary 결과 확인 후 추가 GPU 실험을 했다. 새 family·명목 높이별 실제 hollow−bulge profile 차이의 단위 방향을 주고, 각 물체의 영상 route, evidence, decoder bank와 target label은 그대로 두었다. Family·짝지은 variant를 아는 privileged 진단이다. 축을 구성할 때 target outcome을 쓰지 않았고 배포 지표에는 포함하지 않았다. [진단 예측 코드](target_axis46.py) · [축 교차 평가](aligned_axis_score46.json)

| 같은 영상 route에서 decoder 축만 교체 | Finite pressure Brier | Rigid PyBullet Brier |
|---|---:|---:|
| Always-original | **0.155170** | **0.161305** |
| Joint + v45 source axis | 0.162760 | 0.169232 |
| Joint + target 변형 방향 | 0.155897 | 0.162009 |
| 축 교체 후 남은 손해 [세 진단 동시 95% CI] | **+0.000727** [+0.000468, +0.000985] | **+0.000704** [+0.000450, +0.000959] |

Finite pressure에서 축 교체는 Brier를 **0.006863** 낮췄다 [−0.007324, −0.006401]. 점추정으로 기존 초과 손실의 약 **90.4%**가 줄었다. 반면 교체한 축을 쓰는 상태에서도 joint와 위치 전용의 차이는 −0.0000043 [−0.0000120, +0.0000035]였다. 같은 잠재 물성·표면·최초 위치에서 hollow를 bulge로 바꾸면 finite pressure의 stick label 자체가 terrace의 16.20%, dimple의 18.25%에서 달라졌다. 변형에 물리적 예측 신호가 포함되는 사례가 있음을 뜻한다. 이 전환율만으로 남은 손해 전부가 그 경로에서 왔다고 단정할 수는 없다. [같은 route의 교차 실험](aligned_axis_score46.json) · [동일 물체 변형 대조](paired_variants46.json)

**Physics 설정의 민감도.** 전체 pressure 조건에서 quadrature 해상도를 두 배로 올리자 11개의 label이 바뀌었다. Stick threshold를 0.010 또는 0.040 m/s로 바꿔도 joint−always-original은 +0.007535~+0.007589였다(기본 0.020 포함). Rigid 검증에서도 전체 884,736회에서 접촉이 확인됐고 지정 위치와 실제 최초 접촉 x의 MAE는 0.00204 mm였다. V45 fold는 압력 면적을 넓히자 항상 projection하는 손해가 커졌지만, v46에서는 terrace −0.001746, dimple −0.001124만큼 손해가 줄었다. 두 family 모두 넓은 면적에서도 평균적으로 original이 유리했다. 따라서 v45의 면적 효과 부호 역시 geometry를 넘어 보존되는 법칙으로 쓸 수 없다. [Physics 검산](PHYSICS_RESULTS_KO.md)

## 5. 논문 기여 가능성과 다음 판정 기준

**방법 논문으로는 주가설이 기각됐다.** 등록한 joint rule은 두 baseline을 이기지 못했고, 추가 GPU 계산을 사용했다. Source catalog 내부의 개선이나 부분적으로 양호한 normal 측정만을 제시하면 family transfer를 과대평가하게 된다. V45의 양성 결과도 당시 fold·ripple 혼합, source axis와 접촉 조건의 범위에서 보고하는 것이 타당하다. [V45 결과](../image_pressure_v45/REPORT_KO.md) · [V46 primary](score46.json)

**기여 가능성이 커진 부분은 projection이 실패하는 이유를 분리한 실험이다.** 동일한 sensor·decoder 경로에서 (1) 실제 기하를 주면 위치 전용보다 나아지는 양, (2) 같은 route에서 축만 바꾸면 회복되는 양, (3) 변형 자체가 미래 물리 결과를 바꾸는 빈도를 각각 측정했다. 특히 31 mm의 dimple에서 거의 직교하는 축을 제거하고도 이를 올바른 nuisance 제거로 해석할 수 없다는 점, 그리고 정렬된 축으로 바꿔도 original이 이긴다는 점을 두 contact solver에서 동시에 확인했다. Geometry를 측정할 수 있다는 사실과 그 geometry를 지울 때 기대 loss가 감소한다는 판단 사이에 검증이 필요한 두 단계가 있다. [Axis 검산](axis_transfer46.json) · [교차 진단](aligned_axis_score46.json)

**다음 한 번의 확인 실험은 feature를 더 넣기 전에 축과 label의 관계부터 가르는 것이 좋다.** 완전히 새로운 여러 family를 정하고, 같은 x와 비슷한 normal variance를 갖되 변형 때문에 stick 결과가 바뀌는 쌍과 보존되는 쌍을 source 단계에서 설계한다. 배포 관측으로 추정한 axis alignment, 보수적인 always-original fallback, 현재 joint를 미리 고정한다. 새 family에서 original보다 낮은 Brier와 위치 전용보다 낮은 paired loss가 함께 재현될 때만 adaptive projection의 방법적 기여를 주장한다. 실물이나 독립 카메라가 확보되면 calibration session을 분리해 pressure·충돌 속도를 별도 계측하는 검증이 필요하다. 현재 결과는 두 합성 가족과 extruded 2D 외곽, 같은 물성 field 및 사실 관측 장치를 공유하는 simulator들에 한정된다. [사전 계획 및 한계](PLAN_KO.md) · [광학 대조](VISION_KO.md)

## 문헌 검토 · 무엇을 측정할 수 있고 무엇을 예측할 수 있는가

V46과 직접 연결되는 원저 다섯 편을 골랐다. 문헌을 비교할 때 normal estimation, contact localization, downstream decision이라는 서로 다른 endpoint를 구분했다. 표본 단위가 point cloud, robot episode, 분류 표본으로 달라 논문 간 효과량을 합산하지 않았다.

| 원저와 연구 질문 | 방법·표본 | 관찰과 결론 | v46에 적용할 때의 한계 |
|---|---|---|---|
| Mitra & Nguyen, *Estimating Surface Normals in Noisy Point Cloud Data*. 어느 크기의 이웃으로 normal을 추정해야 하는가? | Noisy 2D curve와 3D surface에서 local least-squares plane의 각도 오차를 분석했다. 반복 수치실험 50회의 평균 오차와 Stanford bunny 점군 예시를 제시했다. | Curvature, sampling density, noise, neighborhood size가 normal 추정 오차를 함께 결정했다. | 하나의 normal 오차에 관한 결과를 여러 normal 각도의 분산이나 미래 Brier의 보증으로 옮길 수 없다. [원문 §§2–5](https://geometry.stanford.edu/lgl_2024/papers/mn-esnnpcd-03/mn-esnnpcd-03.pdf) |
| Pauly, Keiser & Gross, *Multi-scale Feature Extraction on Point-Sampled Surfaces*. 기하 특징을 여러 scale에서 안정적으로 찾을 수 있는가? | Point neighborhood의 PCA 기반 surface variation과 여러 이웃 크기에서의 지속성을 이용해 feature line을 추출했다. Bunny 등 3D 표면과 noise 예제를 비교했다. | 서로 다른 scale에서 나타나는 variation으로 특징을 찾았으며 너무 큰 이웃은 다른 표면 부분까지 포함할 수 있었다. | 이 논문의 PCA surface variation과 v46의 일곱 secant normal angle의 분산은 서로 다른 통계량이다. [원문](https://graphics.stanford.edu/courses/cs468-03-fall/Papers/Pauly_FeatureExtraction.pdf) |
| Kim et al., *Im2Contact*. 외부 camera만으로 물체 간 접촉 위치를 찾을 수 있는가? | 단일 RGB-D, reference depth, optical flow를 결합했다. 정량 평가는 simulation 500 episodes 및 실물 30 episodes(12,362 frames)였다. | 여러 contact type에서 zero-shot 실물 localization을 보였고, 가림과 모호한 물체 모양은 어려운 조건이었다. | 이미 접촉하면서 생기는 motion cue가 들어갈 수 있다. V46은 접근 중 frame만 허용한다. [원문 §§3–5](https://proceedings.mlr.press/v229/kim23b/kim23b.pdf) |
| Meißenhelter et al., *Uncertain Physics for Robot Simulation in a Game Engine*. 위치 불확실성이 만드는 normal과 충돌 후 상태 분포를 어떻게 근사하는가? | Gaussian 위치 불확실성과 sphere hierarchy를 사용해 Armadillo·Bunny·Cup mesh의 sampling 및 linearization을 비교했다. | 평가한 사례에서 surrogate가 단순 선형화보다 sampling 결과에 가까웠고 대량 sampling보다 빨랐다. | 위치 오차 때문에 생기는 normal 분포와, 한 물체의 고정된 공간 범위에서 normal이 변하는 양은 구별해야 한다. [원문 §III](https://cgvr.cs.uni-bremen.de/papers/icra40_24/ICRA40_2024_UncertainPhysics.pdf) |
| Mozannar & Sontag, *Consistent Estimators for Learning to Defer to an Expert*. 어느 입력에서 expert에게 맡길지 어떻게 학습하는가? | Cost-sensitive surrogate를 제안하고 CIFAR-10/100, CIFAR10H crowd annotation, 텍스트와 흉부 영상 분류 등에서 검증했다. | 상대적인 오류 비용을 반영한 선택 규칙을 이론과 실험으로 분석했다. | 분류와 expert deferral 조건에서의 일관성은 합성 contact law의 Brier나 새 shape에서의 조건부 이득을 보장하지 않는다. [원문 §§3–6](https://proceedings.mlr.press/v119/mozannar20b/mozannar20b.pdf) |

**개념 간 연결.** 점이나 silhouette에서 local normal을 복원하고 이웃 범위를 선택하는 일은 geometry estimation이다. 그 위치와 법선이 충돌 결과에 어떻게 전파되는지는 dynamics의 문제다. 두 decoder 중 하나를 선택할 때 필요한 양은 geometry 오차 자체가 아니라, 그 geometry를 관측한 뒤 예상되는 두 decoder의 loss 차이다. V46에서는 이 세 단계를 각각 영상–해석적 geometry 일치도, 두 contact solver, 같은 source로 학습한 두 routing rule로 시험했다. [Mitra & Nguyen](https://geometry.stanford.edu/lgl_2024/papers/mn-esnnpcd-03/mn-esnnpcd-03.pdf) · [Meißenhelter et al.](https://cgvr.cs.uni-bremen.de/papers/icra40_24/ICRA40_2024_UncertainPhysics.pdf) · [Mozannar & Sontag](https://proceedings.mlr.press/v119/mozannar20b/mozannar20b.pdf)

**합의와 겉으로 상충하는 결과.** Local geometry의 추정 품질이 관측 scale과 noise에 의존한다는 근거는 수치해석과 점군 실험에서 일관된다. 이웃을 넓혀 noise를 평균하는 효과와, sharp detail을 잃거나 관계없는 표면을 섞는 효과가 함께 있어 항상 한 방향의 개선을 기대하기 어렵다. 접촉 위치에 대해 실물 측정은 있지만, 고정 scale의 normal variance가 새로운 물체에서 projection 손익을 결정한다는 직접 근거는 이 다섯 편에서 찾지 못했다. 관측 및 학습 조건이 다르므로 실물 contact localization이나 일반 분류 routing의 성공을 v46의 재현 증거로 세지 않았다. [Mitra & Nguyen](https://geometry.stanford.edu/lgl_2024/papers/mn-esnnpcd-03/mn-esnnpcd-03.pdf) · [Pauly et al.](https://graphics.stanford.edu/courses/cs468-03-fall/Papers/Pauly_FeatureExtraction.pdf) · [Kim et al.](https://proceedings.mlr.press/v229/kim23b/kim23b.pdf) · [Mozannar & Sontag](https://proceedings.mlr.press/v119/mozannar20b/mozannar20b.pdf)

### 근거에서 이끌어낸 열 가지 후속 질문

| 질문 | 어떤 공백이 중요한가 | 변수와 검증 방법 |
|---|---|---|
| 1. 같은 분산에서 경사 방향을 뒤집으면 선택도 달라져야 하는가? | Normal 분포가 impulse에 관여하지만 분산은 부호를 지운다. [Meißenhelter et al.](https://cgvr.cs.uni-bremen.de/papers/icra40_24/ICRA40_2024_UncertainPhysics.pdf) | Mirrored profile 쌍에서 signed mean angle을 추가한 rule과 variance-only의 paired loss를 비교한다. |
| 2. 좁은 contact와 넓은 contact에 같은 image window가 적절한가? | 선택한 위치 scale에 따라 geometry 특징이 달라진다. [Pauly et al.](https://graphics.stanford.edu/courses/cs468-03-fall/Papers/Pauly_FeatureExtraction.pdf) | 실제 pressure radius와 feature window를 독립적으로 교차한다. |
| 3. 영상 해상도를 높이면 손익 추정도 좋아지는가? | Sampling density는 normal 측정 오차에 관여한다. [Mitra & Nguyen](https://geometry.stanford.edu/lgl_2024/papers/mn-esnnpcd-03/mn-esnnpcd-03.pdf) | 동일 장면을 여러 px/mm로 렌더링해 normal rank와 held-out Brier를 각각 비교한다. |
| 4. Scale을 선택할 때 noise 수준을 함께 쓰는 것이 유리한가? | Normal 추정에는 noise 감소와 smoothing bias가 함께 있다. [Mitra & Nguyen](https://geometry.stanford.edu/lgl_2024/papers/mn-esnnpcd-03/mn-esnnpcd-03.pdf) | 촬영을 반복해 noise를 추정하고, 같은 compute에서 adaptive scale과 고정 scale을 비교한다. |
| 5. 가림이 심하면 geometry를 쓰지 않는 선택이 유리한가? | 외부 camera의 contact evidence는 visibility에 의존한다. [Kim et al.](https://proceedings.mlr.press/v229/kim23b/kim23b.pdf) | 가림을 무작위로 조작하고 confidence-conditioned fallback의 Brier와 latency를 비교한다. |
| 6. 위치의 확률분포까지 적분하면 hard x bin의 오류를 줄이는가? | 위치 불확실성은 접촉 방향 분포를 바꾼다. [Meißenhelter et al.](https://cgvr.cs.uni-bremen.de/papers/icra40_24/ICRA40_2024_UncertainPhysics.pdf) | 같은 clip에서 x posterior와 point estimate를 만들고 solver·decoder를 고정해 비교한다. |
| 7. 2D silhouette가 가리는 3D normal component는 어느 정도 중요한가? | 3D 점군 기하와 단일 시점의 visual contact는 서로 다른 정보를 갖는다. [Pauly et al.](https://graphics.stanford.edu/courses/cs468-03-fall/Papers/Pauly_FeatureExtraction.pdf) · [Kim et al.](https://proceedings.mlr.press/v229/kim23b/kim23b.pdf) | 같은 2D 외곽에 서로 다른 out-of-plane curvature를 주고 stereo 또는 RGB-D의 추가 가치를 평가한다. |
| 8. Source에서 family 수를 늘리면 조건별 상대 risk가 전이되는가? | 입력별 상대 성능에 따라 선택기를 학습하려면 대표성 있는 expert 결과가 필요하다. [Mozannar & Sontag](https://proceedings.mlr.press/v119/mozannar20b/mozannar20b.pdf) | Source family 수와 표본 수를 독립적으로 바꾸고 전부 새로운 family를 통째로 hold out한다. |
| 9. 영상에서 알아낸 물리 통계와 decoder의 취약한 축을 함께 쓰는 것이 유리한가? | 충돌 결과 추정과 두 predictor 사이의 비용 차이는 다른 endpoint다. [Meißenhelter et al.](https://cgvr.cs.uni-bremen.de/papers/icra40_24/ICRA40_2024_UncertainPhysics.pdf) · [Mozannar & Sontag](https://proceedings.mlr.press/v119/mozannar20b/mozannar20b.pdf) | Profile 변화의 projection-axis alignment를 별개 feature로 등록하고 x·variance-only 대조를 유지한다. |
| 10. 실물에서 calibration·시간 동기화가 바뀌어도 같은 차이가 보이는가? | 실물 영상의 가림과 접촉 시점은 simulation에서 통제한 것과 다를 수 있다. [Kim et al.](https://proceedings.mlr.press/v229/kim23b/kim23b.pdf) | 독립 surface session 별로 접촉 전 영상, 최초 접촉 센서, collision velocity를 동기 측정하고 session을 hold out한다. |

## 재현 자료 및 서버 기록

서버 경로는 `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/normal_variance_v46`다. 두 idle GPU allocation에서 각 6개 catalog의 prediction을 계산하고 latency는 같은 단일 GPU에서 측정했다. Source calibration·pressure·rigid·paired statistics도 할당된 서버 compute node에서 실행했다. 다른 작업이 쓰던 두 GPU는 사용하지 않았다.

| 재현 단계 | 코드·데이터 |
|---|---|
| 사전 계획, geometry, source/target 경계 | [분석 계획](PLAN_KO.md) · [Geometry](geometry46.py) · [Source router](router46.json) · [Target label 실행 시점](TARGET_GO) |
| Raw clip 재생, source 학습, geometry audit | [영상 생성](capture46.py) · [Variance estimator](camera46.py) · [Source fit](fit_route46.py) · [Source CV](source_cv46.json) · [CPU/GPU route replay](route_replay46.json) |
| Target probability·ground truth | [GPU prediction](runner46.py) · [Pressure와 Bullet](mechanics46.py) · [압력·rigid archive SHA-256](mechanics_outputs46.sha256) |
| Primary 채점·GPU timing | [Paired Brier](score46.py) · [Timing](timing46.py) · [두 번째 randomized crossover](audit_latency46_repeat.json) · [GPU 사용 종료](gpu_release46.log) |
| 원인 분리와 추가 검산 | [Target cross-fit](transfer46.py) · [Target axis 진단](aligned_axis_score46.py) · [Paired variant](paired_variants46.py) · [보수적 위치 baseline](conservative46.py) · [그림 생성](figures46.py) |
