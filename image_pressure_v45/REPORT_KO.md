# v45 연구 보고서 · 영상에서 읽은 접촉 위치가 projection 선택에 주는 가치

**접촉 전 합성 영상에서 추정한 위치만으로도 어느 decoder를 쓸지 고르는 데 작은 이득이 남았다.** 새 catalog 12개에서 두 유한 pressure area를 균등 평균한 held-out Brier loss는 always-original **0.150023**, visual **0.149655**, oracle 위치 **0.149622**였다. Visual의 개선량은 **0.000368**(원래 loss의 약 0.25%)이었고, 명목 높이만 사용하는 정책보다도 **0.000207** 낮았다. 효과는 형상에 따라 반대였다. **Fold에서는 visual이 0.001267 개선했고, ripple에서는 0.000531 악화됐다.**

**접촉 위치가 같아도 압력이 퍼지는 면적은 fold에서 중요한 변수였다.** 같은 물체가 같은 지점에 닿고 총 normal impulse도 같을 때, 넓은 patch를 쓰면 fold에서 변형 방향을 지우는 손해가 **0.003738** 커졌다. Ripple은 개별 충돌의 약 3.58%에서 stick/slip 결과가 달라졌지만 두 방향의 변화가 평균에서 상쇄됐다. 두 형상 모두 좁은 면적과 넓은 면적에서 평균적으로 contour 정보를 보존하는 쪽이 유리했다.

**별도 rigid solver로 전체 물체를 다시 충돌시켰을 때도 visual은 always-original 대비 Brier를 0.000711 낮췄다.** 다만 이 solver에서는 영상의 세밀한 위치 정보가 명목 높이 위에 더한 개선의 구간이 0을 포함했다. 같은 GPU에서 512개 물체의 통제된 반복 p90은 always-original 465.19 ms, visual 468.19 ms였다.

![영상 기반 선택의 Brier 차이와 면적에 따른 형상별 projection 손익](image_pressure_v45.png)

왼쪽은 visual의 Brier에서 각 비교 방법의 Brier를 뺀 값이다. 오른쪽은 면적을 넓혔을 때 projection 손해가 얼마나 변했는지 나타낸다. 막대는 각 그림의 세 비교에 대해 보정한 동시 95% CI다.

## 1. 무엇을 감추고 무엇을 바꿨나

**선택과 예측은 다른 단계다.** Original은 7점 query profile을 모두 사용한다. Projection은 같은 profile에서 과거 데이터로 추정한 변형 방향 하나를 지운다. 두 decoder는 같은 factual evidence와 이미 구매한 sensor를 받아 미래 충돌의 stick 확률을 출력한다. 두 decoder 중 어느 쪽을 호출할지는 영상이 추정한 1차원 접촉 좌표 하나로 정했다. 배포 시 shape 이름, pressure area, 실제 offset, 미래 충돌 결과는 선택기에 주지 않았다. 명목 높이별 projection axis와 likelihood template은 미리 준비했다. [동결한 설계](PLAN_KO.md) · [Source axis와 GPU 구현](GPU_KO.md)

- **Source와 target.** V44의 과거 paired profile에서 hollow·bulge 차이를 모아, 19·31 mm에서 각각 공통 axis 하나를 추정했다. 각 높이에서 11,040쌍을 사용했다. 새 target은 catalog support 500–511의 6,144회 물체 추출이며, 중복을 제외하면 5,960개 prototype이다. Fold·ripple, 두 surface, 두 law, 세 contour variant, 두 높이를 교차하고, 세 위치와 세 pressure 조건을 같은 물체에서 비교했다. 이전에 본 family 내부의 새 물체·환경이다. [Source artifact](source_axis45.json) · [예측 검산](audit_gpu45.json)
- **고정한 선택 규칙.** V43의 두 shape를 평균한 historical Brier 차이에 따라 추정 좌표가 여섯 후보 중 30.5 또는 31.0 mm에 가장 가까울 때만 projection한다. Oracle은 같은 규칙에 올바른 좌표를 입력한다. Nominal-only는 실제 접촉점이 어긋나도 명령 높이인 19 또는 31 mm를 넣는다. 따라서 nominal-only 대비 개선은 같은 명목 높이 안에서 다른 선택을 한 효과를 검증한다. Within-height shuffled control은 동일 물체·높이의 세 영상 위치를 서로 바꿔 넣은 모든 permutation의 기대 Brier다. [규칙과 카메라](VISION_KO.md) · [평가 코드](score45_checked.py)
- **동결과 실행.** 2026년 11월 16일 한국 시각 19:12:57에 primary 분석 계획을 동결했다. 19:15:55에 camera와 source lookup을 동결했고, 축을 일치시킨 별도의 source-only control까지 준비한 뒤 19:16:28에 pressure target 생성을 시작하도록 허용했다. Rigid PyBullet 검산도 별도 코드·시각을 기록했다. 새 catalog의 영상·확률은 target outcome을 읽지 않고 생성했다. [사전 기록](analysis_freeze45_utc.txt) · [Camera 동결](camera_freeze45.sha256) · [Target 실행 시각](PRESSURE_GO)
- **신뢰구간.** 같은 물체의 네 sensor noise와 반복된 충돌을 catalog 안에서 먼저 평균했다. 독립 단위는 새 catalog 12개다. 세 primary Brier 비교와 세 pressure contrast 각각에 paired t 및 Bonferroni simultaneous 95% CI를 썼다. 효과 크기와 구간은 테스트한 형상·장비의 가중치에 조건부다. [평가식과 exposure](PLAN_KO.md)

## 2. 위치를 얼마나 정확히 읽었고 그 위치가 prediction을 바꿨나

![접촉 전 동일한 물체를 찍은 세 합성 grayscale frame](camera_examples45.png)

96×32 pixel의 세 frame에 보이는 두 rail marker로 좌표계를 복원하고, 어두운 tip의 이동을 회색 surface 경계까지 외삽했다. 등속 접근이며 마지막 frame도 접촉 전이다. 카메라 배율, 이동, 노이즈, 부분 가림을 변화시켰다. 두 면적에서는 같은 영상을 쓴다. 총 221,184개 clip에서 위치 MAE는 **0.09535 mm**, 절대 오차 p90은 **0.19599 mm**였다. 여섯 위치 bin의 정확도는 97.11%, oracle과 같은 decoder 선택은 99.27%였다. 19 mm 주변에서는 세 위치 모두 original을 고르므로 좌표 오차가 route를 바꾸지 않는다. [영상 방법·한계·수치](VISION_KO.md)

| Source-axis 정책 | Held-out Brier ↓ | 해석 |
|---|---:|---|
| Always-original | 0.150023 | 모든 contour 정보를 유지한다. |
| Always-project | 0.157022 | 모든 곳에서 같은 축을 지우면 손해다. |
| Nominal-only | 0.149862 | 명목 높이별 route만 바꾼다. |
| Within-height shuffled | 0.149947 | 위치별 선택 빈도를 보존하고 해당 물체의 사건과의 대응을 끊는다. |
| Oracle 위치 | 0.149622 | 같은 고정 규칙에 정확한 최초 접촉 위치를 준다. |
| Visual 위치 | **0.149655** | 실제 추정한 좌표로 32.80%에서 projection을 선택했다. |

| 사전 지정 paired 비교 | Brier 차이 [동시 95% CI] |
|---|---:|
| Visual − always-original | **−0.000368** [−0.000631, −0.000105] |
| Visual − oracle 위치 | +0.000034 [−0.000008, +0.000076] |
| Visual − nominal-only | **−0.000207** [−0.000325, −0.000089] |

Visual은 12개 catalog 중 10개에서 always-original보다 낮은 loss를 보였다. 점추정으로 oracle의 평균 개선량 중 **91.6%**를 회수했다. Visual과 oracle 차이는 0을 포함하며 사전에 정한 noninferiority margin은 없다. 두 방법이 동등하다는 판정은 유보한다. Visual−shuffled도 −0.000292 [보조 95% CI −0.000339, −0.000244]였다. 명목 높이와 선택 빈도를 유지해도 어떤 접촉점에 projection을 적용했는지가 결과에 영향을 주었다. [Primary·보조 결과](score45.json)

**평균 이득의 출처.** Visual−original은 fold에서 −0.001267 [두 형상 동시 95% CI −0.001633, −0.000901], ripple에서 +0.000531 [+0.000300, +0.000761]이었다. 특히 30.5 mm에서 fold의 차이는 −0.006441, ripple은 +0.001818이었다. 같은 x가 두 형상에서 같은 선택을 강제하기 때문이다. 관찰한 두 형상의 조건부 loss가 그대로라는 가정 아래, 단순 가중 평균이 이득으로 바뀌는 fold 비중은 약 29.5%다. 이 수치는 다른 물체군의 적정 threshold를 측정한 값이 아니라 이번 결과의 가중 민감도를 나타낸다. [위치별 paired 결과](score45.json)

**다른 source rule과 camera 오차.** Source-axis와 같은 축으로 과거 point outcome만 재채점한 별도 규칙은 30.5 mm에서만 projection을 선택했다. 이 경우에도 visual Brier는 0.149664, visual−original은 −0.000359 [−0.000488, −0.000231]였다. Source에서 31 mm의 손익 부호는 leave-one-catalog-out으로 12회 중 6회 달라졌다. Primary는 마지막 frame을 가려도 visual−original −0.000340 [−0.000621, −0.000059]였지만, 좌표에 −0.35 mm bias를 주면 −0.000220 [−0.000511, +0.000072]로 우위를 구분하지 못했다. +0.35 mm bias에서는 primary 이득이 남았고 source-matched 규칙의 이득은 사라졌다. 위치 오차의 방향과 rule의 경계가 함께 중요하다. [Source-matched 결과](score45_matched.json) · [Source-only 추정](matched_source45.json)

## 3. 같은 접촉 위치에서 pressure area의 효과

세 물리 조건은 정확히 같은 x에서 접촉을 시작했다. 정규화한 원형 pressure profile은 접촉 동안 반지름이 0에서 peak까지 커졌다가 줄어든다. Point control, peak 반지름 0.25 mm의 narrow, 2 mm의 broad를 비교했다. 매 순간 pressure centroid와 총 scalar normal impulse(물체별 0.910–2.048 N·s)는 맞췄다. 같은 하중을 더 넓게 분포시키므로 peak pressure는 낮아진다. Local normal의 방향이 달라질 수 있어 합성된 impulse vector 방향은 자유롭게 두었다. Effective pressure area는 (∫p dA)²/∫p² dA이며, 곡면의 면적 요소로도 계산했다. 이 개입은 최초 접촉점까지 움직였던 v44의 rigid sole 변화와 구별된다. [개입 식과 neutral control](PRESSURE_PROTOCOL_KO.md) · [모형 변수 검산](pressure_audit45.json)

D는 해당 면적에서 **항상 projection했을 때의 Brier − 항상 original을 썼을 때의 Brier**다. 앞 절의 선택 정책 효과와 구분한다. Narrow·broad D 구간은 네 비교에, broad−narrow와 형상 interaction의 구간은 세 비교에 대해 보정했다. [사전 지정 pressure 비교](score45.json)

| Shape | 곡면 effective area, narrow → broad (mm²) | Narrow D [동시 95% CI] | Broad D [동시 95% CI] | Broad−narrow D [동시 95% CI] |
|---|---:|---:|---:|---:|
| Fold | 0.18095 → 11.65051 | +0.009794 [+0.008909, +0.010679] | +0.013531 [+0.012592, +0.014470] | **+0.003738** [+0.003403, +0.004072] |
| Ripple | 0.18060 → 11.55776 | +0.002356 [+0.001745, +0.002967] | +0.002314 [+0.001691, +0.002937] | **−0.000042** [−0.000141, +0.000056] |

Fold−ripple interaction은 **+0.003780 [+0.003436, +0.004124]**였다. Fold에서는 면적을 넓힐 때 stick→slip과 slip→stick이 각각 D에 +0.002323과 +0.001415를 보탰다. Ripple에서는 두 전환의 기여가 +0.000563과 −0.000605였다. 즉 ripple에서도 약 3.58%의 label이 바뀌지만, 평균 Brier 손익은 상쇄된다. 사후 factorial ablation에서 위치별 normal만 바꿔도 fold의 면적 효과 +0.003730이 재현됐고, normal을 고정하고 위치별 friction만 남기면 +0.000009였다. 이 결과는 처방한 physics model 안에서 패치가 서로 다른 기울기를 포함하게 되는 경로를 지지한다. [전환별 분해와 ablation](PRESSURE_RESULTS_KO.md)

**결과를 바꿀 수 있는 정의.** Stick proxy는 충돌 후 중심 접선 속도 ≤0.020 m/s로 정의했다. 회전을 고정하고 하나의 translational slip mode에 대해 Coulomb capacity를 적분했다. Flat·uniform surface에서는 이론대로 면적에 따른 속도 차이가 사라졌다. 2,654,208개 충돌 조건 전체에서 수치 적분 해상도를 두 배로 높이면 label 956개(0.0360%)가 바뀌었고, fold의 broad−narrow D는 +0.003650으로 유지됐다. Ripple은 0.010 m/s 기준에서 −0.000138, 0.040 m/s에서 +0.000247로 방향이 바뀌었다. 따라서 ripple의 평균 손익이 작다는 해석에는 threshold를 붙여야 한다. 실제 pad의 탄성, 자연 형성되는 pressure, local incipient slip은 별도 측정이 필요하다. [정밀도·threshold 재검산](pressure_mechanism45.json) · [실물 partial slip 측정](https://people.csail.mit.edu/yuan_wz/GelSight1/ICRA15_2740_FI.pdf)

## 4. 같은 GPU에서의 배포 비용

서버의 네 GPU allocation 중 다른 process·점유 메모리가 없던 두 장에서 예측을 나눠 계산했다. Latency는 한 RTX PRO 6000 Blackwell에서 512개 물체, 1회 sensor draw, 같은 source bank로 비교했다. Visual은 4,718,592 byte의 uint8 frame을 host에서 GPU로 전송하고 영상 해석과 route 선택 후 **각 물체에서 선택된 하나의 decoder만** 계산했다. 각 방법에는 K256 선택, paid sensor routing, physics posterior, GPU 동기화가 포함된다. Actual image acquisition·robot reset·offline rendering·target simulation은 별도 비용이다. [측정 코드와 input 범위](GPU_KO.md)

| 정책 | Primary held-out Brier ↓ | 첫 측정 GPU p90 (ms) | CPU 배치를 통제한 반복 GPU p90 (ms) [95% CI] |
|---|---:|---:|---:|
| Always-original | 0.150023 | 769.38 | **465.19** [459.79, 467.10] |
| Oracle 위치 | 0.149622 | 735.37 | **465.69** [462.43, 465.85] |
| Visual 위치 | 0.149655 | 785.50 | **468.19** [465.69, 481.65] |
| Source-matched oracle | 0.149635 | 737.89 | 464.42 [462.41, 465.46] |
| Source-matched visual | 0.149664 | 654.04 | 467.75 [466.12, 478.36] |

첫 crossover에서는 일부 panel에서 모든 경로에 큰 지연이 있었고 visual−original p90 차이의 paired 95% CI가 −101.00~+262.44 ms였다. 같은 GPU·같은 순서·같은 입력에서 CPU core를 고정하고 다른 v45 계산을 해당 node 밖으로 옮겨 재측정했다. 각 crossover에서 정책당 warm-up 후 36회, panel 단위 paired bootstrap 16,000회로 구간을 추정했다. 통제한 반복의 visual−original은 **+2.99 ms [−0.30, +17.29]**, visual−oracle은 **+2.50 ms [+0.88, +16.06]**였다. 영상 전송·위치 추정·lookup만의 p90은 3.16 ms였다. 작지만 양수인 image 처리 비용과 전체 GPU p90 추정 오차를 함께 고려해야 한다. 최초 측정의 변동 원인을 특정 process에 귀속하지 않았다. [양쪽 측정 전체](timing_comparison45.json) · [GPU 반환 기록](gpu_release45.log)

## 5. 독립 PyBullet에서도 영상의 이득이 남았나

Rigid sphere(반지름 0.5 mm)가 같은 요청 위치에 처음 닿도록 초기 상태를 조정하고 PyBullet의 manifold에서 첫 positive-force 좌표를 직접 읽었다. Oracle에는 이 실측 좌표를 주었다. 전체에서 지정 위치와의 MAE는 0.005105 mm, 절대 오차 p90은 0.010893 mm였다. 여섯 bin 사이의 oracle 선택이 바뀐 일은 없었다. Primary의 point model과 rigid solver 간 stick/slip은 2.29% 달랐다. 이 실험은 독립 query dynamics에 대한 대조다. 같은 pre-impact pin 영상을 rigid 접근의 대리 관측으로 사용했으며, sphere를 촬영한 실물 영상은 없다. 분포 면적의 효과는 앞 절의 pressure 실험에서 측정했다. [Rigid setup과 입력 정보](BULLET_PROTOCOL_KO.md) · [전체 검산](external_full_score45.json)

사전 지정한 첫 8개 물체/환경만 보면 visual−original은 **+0.000532**로 이득이 확인되지 않았다. 동일 물체의 prescribed point와 finite pressure에서도 각각 +0.000659와 +0.000791이었다. 표본 구성과 solver 변경을 분리하기 위해 규칙을 그대로 두고 나머지 120개/환경까지 rigid 충돌을 사후 확장했다. 추가 표본은 같은 12개 catalog에서 추출됐고, 첫 8개의 연속 속도·접촉점은 재실행에서 일치했다.

| Visual − original, 같은 물체끼리 비교 | Prescribed finite 평균 | Prescribed point | Rigid PyBullet [3개 정책 비교 동시 95% CI] |
|---|---:|---:|---:|
| 처음 8개/환경 | +0.000791 | +0.000659 | +0.000532 [−0.000823, +0.001887] |
| 추가 120개/환경 | −0.000445 | −0.000649 | **−0.000793** [−0.001139, −0.000448] |
| 전체 128개/환경 | −0.000368 | −0.000567 | **−0.000711** [−0.001005, −0.000416] |

전체 884,736개 rigid impact에서 Brier는 always-original **0.150661**, nominal-only **0.150038**, oracle **0.149899**, visual **0.149950**이었다. Visual−nominal은 −0.000088 [−0.000187, +0.000010]으로, rigid solver에서 세밀한 위치가 명목 높이 위에 더하는 이득은 primary만큼 확실하지 않았다. Visual−oracle은 **+0.000051** [+0.000008, +0.000095]로 작지만 exact location의 이득을 구별할 수 있었다. Fold의 visual−original은 −0.001549 [−0.001927, −0.001170], ripple은 +0.000127 [−0.000120, +0.000375]였다. 방향이 다른 소표본 결과를 solver 교체 하나로 설명할 수 없고, 전체 표본의 일치도는 사후 확장에 해당한다. [초기 소표본](external_score45.json) · [동일 물체 전체와 추가 표본](external_full_score45.json)

## 6. 논문 기여 가능성과 남은 대안 설명

**논문의 중심으로 삼을 결과는 두 가지다.** 첫째, 이미 고정된 shape-blind decoder의 선택에 pre-impact contact coordinate가 어느 정도 가치가 있는지, 그리고 그중 명목 높이만으로 얻을 수 있는 양이 얼마인지 분리했다. Primary에서는 visual이 두 기준을 모두 이겼고, 독립 rigid solver 전체에서도 always-original 대비 이득이 남았다. 둘째, 최초 위치와 scalar impulse를 고정했는데도 pressure 폭이 fold의 projection 손해를 키웠다. 같은 비율로 event가 바뀐 ripple에서 두 방향이 상쇄된다는 paired 분해는 shape별 차이가 왜 생기는지 설명한다. 기존 연구에는 영상 contact localization과 pressure integration 각각의 선례가 충분하다. 이 연구의 후보 기여는 이 둘을 **미래 충돌 예측에 어떤 contour 정보를 남길지**라는 하나의 결정으로 연결하고, 기대 이득과 실패 조건을 함께 측정한 것이다. [V44 선행 실험](../contact_patch_v44/REPORT_KO.md) · [Im2Contact](https://proceedings.mlr.press/v229/kim23b/kim23b.pdf) · [Elandt et al.](https://arxiv.org/pdf/1904.11433)

**현재 근거로 설정할 주장 범위.** 같은 위치에서 두 형상이 반대 손익을 보이므로 x 하나로 모든 물체의 최적 decoder를 고를 수 없다. 양의 평균 개선은 이 실험의 균등한 shape 혼합에 의존한다. V44의 손익과 v45의 손익은 projection axis, source rule, 물체와 target dynamics가 함께 달라진 결과여서 버전 간 크기 차이를 camera 오차로 취급할 수 없다. Optical calibration·등속 접근·보이는 fiducial·훈련 때의 paired variant 구성이 필요하다. GPU p90은 실물 촬영이나 reset의 가치를 판정하지 않으며, 알려진 pressure 폭은 배포 입력에 포함되지 않는다. 시뮬레이션 기전에 대한 근거는 강해졌지만, 새 실물과 자연적으로 생긴 contact patch에서도 손실이 줄어든다는 주장은 아직 검증이 필요하다. [정보 접근·GPU 범위](GPU_KO.md) · [영상 생성](VISION_KO.md) · [Pressure 명세](PRESSURE_PROTOCOL_KO.md)

**다음에 가를 두 설명.** 같은 x에서 fold와 ripple의 손익이 갈린 원인이 보이는 국소 curvature라면, label 없이 영상에서 추정한 **접촉점 주변 normal의 분산**을 x에 추가한 rule이 x-only보다 개선돼야 한다. 새 contour family 두 개 이상에서 x-only·x+normal dispersion·always-original을 먼저 고정하고, 순위를 예측한 뒤 검증하는 편이 좋다. 실물 또는 자연 형성되는 patch의 외부 검증에서는 같은 최초 위치를 고속 영상으로 확인하고, 별도 pressure array로 면적·centroid·정규하중을 동시에 측정해야 한다. Surface session을 통째로 분리해 1차 비교를 visual−nominal로 두면, 절대 높이를 외운 효과와 관측의 가치를 구분할 수 있다. [Pressure ablation](pressure_mechanism45.json) · [물리 pressure 측정과 전이](https://arxiv.org/html/2501.08077v1)

## 문헌에서 분리한 세 가지 측정 문제

문헌 검토 범위는 외부 camera가 접촉 위치를 관찰하는 연구, 재료와 형상에서 pressure field를 계산하는 연구, 실제 slip을 측정하는 연구의 원저 다섯 편이다. 아래 문헌적 주장은 이 다섯 편에 근거한다. 각 논문의 관측 시점과 정답이 다르므로 보고된 오차를 하나의 표준화한 효과로 합치지 않았다.

| 논문과 연구 질문 | 방법, 데이터와 표본 단위 | 핵심 결과와 저자 결론 | v45에 남기는 한계 |
|---|---|---|---|
| Kim et al., *Im2Contact*. 외부 camera 하나로 로봇이 든 물체의 접촉을 어디까지 찾는가? | Simulation에서 학습한 depth 기반 network에 reference image와 optical flow를 결합했다. 정량 평가는 simulation 500 episodes(75,000 frames), 실물 30 episodes(12,362 frames)였다. | Occlusion·모호한 형태가 단일 depth 추정의 실패 원인이었고 추가 영상 단서로 여러 contact type을 추정했다. | 정답은 영상 내 이미 발생한 contact 위치다. 접촉 이후 생긴 motion cue를 아직 일어나지 않은 impact의 예측에 그대로 사용할 수 없다. [원문 §§3–5](https://proceedings.mlr.press/v229/kim23b/kim23b.pdf) |
| Chaudhury et al., *Using Collocated Vision and Tactile Sensors for Visual Servoing and Localization*. 접촉 전의 영상 위치와 접촉 때의 tactile 정보를 어떻게 결합하는가? | Camera와 GelSight를 같은 축에 둔 실물 장비에서 물체별 접근과 접촉을 측정했다. 임의 위치 검증에서는 물체마다 5개 pose를 사용했다. | 풍부한 특징에서 두 sensor가 상호 보완했다. 반복 무늬와 특징 없는 접촉부에서는 tactile refinement가 잘못된 위치에 수렴하기도 했다. | 접근 경로, 교정과 접촉 부위 가시성이 결과에 영향을 준다. 일정한 접근·보이는 기준점에서 얻은 1D 오차를 일반 장면에 옮기기 어렵다. [원문 §IV](https://arkadeepnc.github.io/projects/collocated_vision_touch/img/ICRA_RAL_FINAL_Formatted-final-compressed.pdf) |
| Elandt et al., *A pressure field model…*. 분포 접촉이 만드는 net force와 moment를 빠르게 근사할 수 있는가? | 두 물체 내부에 pressure field를 부여하고 접촉면의 traction을 적분했다. 여러 기하 예제와 gripper–pencil simulation으로 거동을 보였다. | 한 점의 힘으로 표현하기 어려운 유한 접촉의 moment를 모델링했다. | 표면·재료에서 생기는 patch를 계산하는 방법이다. 중심을 고정하고 외부에서 면적을 지정한 v45에서는 형성 과정과 torsion을 검증할 수 없다. [원문](https://arxiv.org/pdf/1904.11433) |
| Leins et al., *HydroelasticTouch*. 분포 압력으로 채운 synthetic taxel 데이터가 실물 센서의 물체 상태 예측에 쓰이는가? | Hydroelastic surface에서 pressure를 샘플링했다. 5종 primitive로 21,000개 no-pitch 또는 105,000개 pitch 학습 표본을 구성해 실물 물체 방향 오차를 측정했다. | 일부 실물에서 network의 전이를 보였고 soft brick과 ruler에서는 PCA가 더 정확했다. | 이득이 geometry와 baseline에 따라 달랐다. 이 실험의 endpoint는 물체 방향이며 미래 stick의 Brier와 다르다. [원문 §§IV–VI](https://arxiv.org/html/2501.08077v1) |
| Yuan et al., *Measurement of Shear and Slip with a GelSight Tactile Sensor*. 실제 patch가 미끄러지기 시작하는 시점을 볼 수 있는가? | Elastomer의 marker 이동을 영상으로 추적해 gripper형 장치에서 normal·shear·torsional loading과 sliding 과정을 관찰했다. | 가장자리에서 시작해 안쪽으로 전파되는 incipient slip을 관찰했다. | Aggregate tangential speed는 patch 안의 stick/slip 공존을 구분하지 못한다. [원문](https://people.csail.mit.edu/yuan_wz/GelSight1/ICRA15_2740_FI.pdf) |

**지식 지도.** 접근 영상 → 보이는 경계와 경로 → 최초 접촉 위치의 불확실성은 Kim·Chaudhury가 다룬 관측 문제다. 접촉 위치·재료·형상 → pressure 분포 → 힘과 moment는 Elandt·Leins가 다룬 역학 문제다. 국소 변형 → incipient slip이라는 실제 측정 수준은 Yuan이 다룬다. V45의 연결은 최초 위치 추정이 downstream decoder 선택에 주는 이득을 재고, 같은 위치에서 pressure 폭 개입이 그 decoder의 손익을 바꾸는지 묻는 것이다. [Kim](https://proceedings.mlr.press/v229/kim23b/kim23b.pdf) · [Chaudhury](https://arkadeepnc.github.io/projects/collocated_vision_touch/img/ICRA_RAL_FINAL_Formatted-final-compressed.pdf) · [Elandt](https://arxiv.org/pdf/1904.11433) · [Leins](https://arxiv.org/html/2501.08077v1) · [Yuan](https://people.csail.mit.edu/yuan_wz/GelSight1/ICRA15_2740_FI.pdf)

**근거의 강도와 겉으로 상충하는 결과.** 영상에 접촉 위치 단서가 있고 분포 접촉이 실물에서 공간 구조를 가진다는 점에는 직접 관측 근거가 있다. 고정한 접근 장치에서 얻은 예측이 새 장비·표면에서도 유용하다는 근거는 조건부다. 예컨대 Kim의 optical flow는 환경 물체가 접촉 후 움직인 단서를 활용할 수 있고, Chaudhury의 접촉 전 추정은 그런 단서를 쓸 수 없다. Leins의 network와 PCA 순위가 물체에 따라 바뀌는 결과 역시 어느 한 해법이 접촉 추정 전체에서 우월하다고 보기 어렵게 한다. 이 논문들에서 v45의 Brier 개선량이나 실제 pad의 contact-area 효과를 직접 추정할 근거는 부족하다. [Kim](https://proceedings.mlr.press/v229/kim23b/kim23b.pdf) · [Chaudhury](https://arkadeepnc.github.io/projects/collocated_vision_touch/img/ICRA_RAL_FINAL_Formatted-final-compressed.pdf) · [Leins](https://arxiv.org/html/2501.08077v1)

### 남은 연구 질문과 검증 방법

| 질문 | 근거 및 중요성 | 바꿀 변수와 평가 |
|---|---|---|
| 1. 실제 pre-impact 영상에서도 위치 선택의 이득이 남는가? | 문헌의 contact annotation은 이미 발생한 사건을 포함한다. [Kim](https://proceedings.mlr.press/v229/kim23b/kim23b.pdf) | 접촉 직전 frame만 전달하고 뒤의 force trace로 결과를 가린 채 채점한다. |
| 2. 영상에서 하나의 x 대신 위치 분포를 쓰면 임계점 부근의 손실이 줄어드는가? | Occlusion에서 단일 위치가 불확실하다. [Kim](https://proceedings.mlr.press/v229/kim23b/kim23b.pdf) | 가림·시점을 조작해 hard routing과 위치 posterior를 적분한 routing을 paired Brier로 비교한다. |
| 3. 곡선으로 접근하면 경로 외삽이 언제 실패하는가? | 직선 접근·영상 특징은 pre-contact localization의 가정이었다. [Chaudhury](https://arkadeepnc.github.io/projects/collocated_vision_touch/img/ICRA_RAL_FINAL_Formatted-final-compressed.pdf) | 같은 최종 위치에서 경로 곡률과 가속도만 바꾸고 좌표·route 오차를 함께 잰다. |
| 4. Camera를 다시 장착한 날에도 보정이 유지되는가? | 위치 추정에는 optical geometry와 교정이 개입한다. [Chaudhury](https://arkadeepnc.github.io/projects/collocated_vision_touch/img/ICRA_RAL_FINAL_Formatted-final-compressed.pdf) | 재장착 session을 독립 표본으로 삼고 drift 보정 전후 Brier를 비교한다. |
| 5. 같은 최초 접촉 위치에서 자연 형성되는 contact area도 같은 효과를 내는가? | Hydroelastic contact는 물체 geometry와 deformation을 이용한다. [Elandt](https://arxiv.org/pdf/1904.11433) | 같은 tip과 위치에서 pad compliance·normal load를 교차해 pressure film 실측과 대조한다. |
| 6. 같은 effective area에서 pressure profile의 꼬리만 달라도 효과가 달라지는가? | 같은 patch의 분포 모양이 wrench에 관여한다. [Elandt](https://arxiv.org/pdf/1904.11433) | 균일·Hertz형·annular profile의 총하중과 effective area를 맞춰 비교한다. |
| 7. Pressure sensor의 공간 해상도가 area 효과 추정에 필요한 수준은 어느 정도인가? | Sensor의 receptive region으로 압력을 합칠 때 공간 정보가 달라진다. [Leins](https://arxiv.org/html/2501.08077v1) | 같은 충돌을 여러 taxel 해상도로 재구성하고 물체별 area 오차를 잰다. |
| 8. Rotation을 허용하면 중심을 맞춘 면적 효과가 커지는가? | 접촉면 분포가 torque를 전달할 수 있다. [Elandt](https://arxiv.org/pdf/1904.11433) | 동일 질량과 물체에서 angular inertia·회전 구속을 교차하고 angular slip과 Brier를 기록한다. |
| 9. 국소 incipient slip과 전체 gross slip 중 어느 쪽이 영상 위치로 예측되는가? | 둘은 실제 patch에서 다른 시점에 일어난다. [Yuan](https://people.csail.mit.edu/yuan_wz/GelSight1/ICRA15_2740_FI.pdf) | Tactile flow와 중심 속도로 두 label을 만들어 같은 decoder를 각각 평가한다. |
| 10. 재료를 보지 못한 상황에서 어느 정도의 실물 교정이 필요한가? | Synthetic tactile transfer의 성능 순위가 실제 물체마다 달랐다. [Leins](https://arxiv.org/html/2501.08077v1) | Surface session을 통째로 hold out하고 교정 0·4·16회에서 고정 규칙과 재학습을 비교한다. |

## 서버 및 재현 자료

서버 경로는 `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/image_pressure_v45`다. 영상 추정, pressure solver, GPU 예측과 latency를 나누어 실행했고, statistical scoring은 할당된 compute node에서 시행했다. 기존 v44 파일은 그대로 참조했다.

| 단계 | 실행 자료 |
|---|---|
| 사전 등록과 정보 차단 | [Primary 계획](PLAN_KO.md) · [Source-matched 규칙 사전 정의](SOURCE_CONTROL_KO.md) · [평가 코드·source lookup checksum](evaluation_freeze45.sha256) |
| 영상과 source axis | [Frame 생성·위치 추정](camera45.py) · [Clip 재현과 pixel checksum](capture45.py) · [실제 raw clip 표본](camera45/panel_00/pixels_fold_hollow_h31_d0.npz) · [Axis 추정](source_axis45.py) |
| Held-out prediction과 simulation | [GPU runner](runner45.py) · [Pressure model](pressure45.py) · [Rigid dynamics](bullet45.py) · [전체 물체 확장](bullet_full45.py) · [Rigid archive 검산](BULLET_RESULTS_KO.md) · [Primary physics checksum](pressure_outputs45.sha256) |
| 점수표 및 latency | [Primary 결과](score45.json) · [Source-matched](score45_matched.json) · [Rigid paired 결과](external_full_score45.json) · [두 GPU timing run](timing_comparison45.json) · [그림](display45.py) |

평가 시 보관한 확률의 최댓값이 float64 누적 반올림으로 1.0000000000000004여서 엄격한 입력 검사에서 중단됐다. 입력 검사에만 1e−12 허용오차를 둔 사본으로 서버에서 재실행했다. 확률 clipping이나 표본 제외 없이 같은 제곱오차 식을 사용했고, 직접 계산한 제곱오차와 전개식도 검산했다. 사전 동결한 코드와 실행한 파일을 모두 보관했다. [동결본](score45.py) · [실행본](score45_checked.py)
