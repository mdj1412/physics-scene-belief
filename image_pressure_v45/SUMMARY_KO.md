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
