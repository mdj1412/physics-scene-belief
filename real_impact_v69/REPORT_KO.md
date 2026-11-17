# v69 · 반복 impact의 contact patch와 충돌 직후 tangential velocity

**2026-11-17 · 실물 공개 자료에 대한 external validation**

같은 Wilson tennis ball을 반복해서 충돌시킨 실험에서 **footprint와 충돌 직후 궤적이 연결되는 두 사례**를 찾았다. 별도의 golf ball 실측 1,023회에서는 각 impact의 충돌 전후 속도와 spin으로 두 solver를 직접 비교했다. 이 비교에서 **normal compliance를 추가하는 것만으로는 PyBullet의 속도 오차가 줄지 않았다.** Tangential spring까지 추가하면 일부 속도 오차가 줄었지만, 새 turf에 대한 near-stick Brier loss 개선은 두 평가 방향에서 재현되지 않았다. [Cross 원문](https://doi.org/10.1007/s12283-014-0159-x) · [Biber et al. 원문과 보충자료](https://doi.org/10.1007/s12283-023-00442-4)

## 1. 어떤 실측을 확보했나

| 자료 | 같은 impact에서 연결되는 관측 | 반복 단위와 사용할 수 있는 지표 |
|---|---|---|
| Cross, Wilson tennis ball | 두 대표 사례의 입사 상태, chalk footprint, rebound trajectory | 같은 실물 공으로 조건당 5–6회. 공개 그림에서 두 사례의 COM velocity를 복원했다. 각 반복의 spin이나 dynamic patch mask는 공개되지 않았다. [출판본 Fig. 9](https://media.springernature.com/full/springer-static/image/art%3A10.1007%2Fs12283-014-0159-x/MediaObjects/12283_2014_159_Fig9_HTML.gif) |
| Biber et al., golf ball | 한 영상당 inbound/outbound COM velocity와 spin 6열 | Artificial turf A 330회, natural tee turf B 693회. 각 launcher setting에서 A는 6회, B는 6회 또는 12회 반복. 잔디가 접촉면을 가리고 CSV에 개별 공 serial이 없다. 각 impact의 release kinematics와 파생 event를 평가했다. [측정표](https://static-content.springer.com/esm/art%3A10.1007%2Fs12283-023-00442-4/MediaObjects/12283_2023_442_MOESM2_ESM.zip) · [영상 저장소](https://data.bris.ac.uk/data/dataset/1cwnj42n63q5w2sc843iww7bbd) |
| Cross, 60 mm rubber ball | 600 fps 원영상과 저자가 같은 예로 소개한 chalk mark | 22 frames에서 rebound COM velocity 10.95 m/s, 처리 조건별 10.68–11.48 m/s. 누적 footprint의 장축/단축은 2.14였다. 정식 trial ID와 반복 목록은 제공되지 않는다. [저자 자료](https://www.physics.usyd.edu.au/~cross/HAWKEYE.htm) · [재분석](cross_rubber/READOUT_KO.md) |
| Cross, 별도 normal impact | 같은 Wilson ball의 wire-grid radius 도달 시점과 최대 지름 | 정상 충돌 한 건으로 시간별 patch 팽창을 독립 점검했다. 실험 표면과 입사 방향이 달라 위 oblique 사례와 별도로 평가했다. [원문](https://doi.org/10.1007/s12283-014-0159-x) |
| Arakawa와 Wang의 patch 측정 | 투명 PMMA의 시간별 optical area, 또는 carbon paper의 누적 축 길이 | Arakawa의 접촉 중 평균 속도와 Wang의 기하학적 slip ratio는 각 논문이 실제로 측정한 시점과 정의로만 사용했다. [Arakawa](https://doi.org/10.11395/jjsem.5.356) · [Wang](https://doi.org/10.15632/jtam-pl/146621) |

이번 검색에서는 **같은 specimen의 각 반복에 time-resolved patch, 이탈 직후 속도, 접촉면 내부의 독립 stick/slip 판독을 모두 연결한 공개 자료**를 확인하지 못했다. 따라서 실제 local stick/slip의 held-out Brier loss는 미측정이다. 아래 표의 Brier는 실측 velocity와 spin으로 정의한 별도 endpoint다. 이 구분은 sparse paired data를 활용하면서 측정되지 않은 label을 solver로 만들어 평가하는 순환을 막는다. [데이터 및 원본 checksum](data69/DATA_KO.md)

## 2. 같은 공에서 footprint와 속도를 같이 본 결과

두 oblique impact의 입사 수직속도는 모두 −6.8 m/s, 초기 spin은 0이었다. 출판본 graph에서 처음 네 outbound marker의 수평 기울기를 읽었다. 같은 방법으로 복원한 입사 수평속도는 5.767, 21.098 m/s로 저자 기록 5.8, 21.1과 일치했다. 표의 범위는 tick과 각 marker를 독립적으로 ±1 pixel 바꾼 digitization envelope다. 실험 간 오차나 모집단의 CI는 포함하지 않는다. [공개 원도판](https://media.springernature.com/full/springer-static/image/art%3A10.1007%2Fs12283-014-0159-x/MediaObjects/12283_2014_159_Fig9_HTML.gif) · [판독 좌표와 코드](data69/cross_figure_measurements.py)

| 수평 입사속도 | 누적 footprint, 길이 × 폭 | 실측 rebound COM (m/s) | PyBullet / normal-compliant 예측 (절대오차) | Tangential spring 예측 (절대오차) |
|---:|---:|---:|---:|---:|
| 5.8 m/s | 42 × 40 mm | 2.496 [2.317, 2.678] | 3.480 (0.984) | 2.068 (0.428) |
| 21.1 m/s | 99 × 약 44 mm | 15.914 [15.154, 16.687] | 15.150 (0.764) | 15.150 (0.764) |

두 번째 폭 44±1 mm는 같은 조건 5회의 관찰이며 길이 99 mm는 그림의 대표 사례다. 시뮬레이션은 58.1 g, 반지름 33 mm에 thin-shell inertia ratio 2/3를 가정했다. 중심 예측의 restitution 0.75, friction 0.50, memory stiffness ratio 0.5는 복원한 outbound 수치를 전달받기 전에 정했다. 실제 chalk friction과 shell inertia를 따로 측정한 값은 없다. [사전 계산 조건](tennis/PLAN_KO.md) · [서버 예측 전수](tennis/predictions.csv)

첫 사례에서 flat rigid Coulomb law는 friction을 아무리 높여도 slip이 0이 되는 지점에서 추가 impulse를 멈춘다. 이 공에 가정한 inertia에서는 도달 가능한 최소 COM velocity가 5.8/(1+2/3) = **3.480 m/s**로 digitization 상한 2.678을 넘는다. Tangential spring은 slip이 0이 된 뒤에도 저장된 변형 에너지가 운동으로 돌아올 수 있다. 미리 정한 parameter grid에서 예측 범위는 1.450–4.073 m/s였으며, 실측은 그 안에 있다. 두 번째 사례에서는 PyBullet 12.660–18.856 m/s와 spring 11.036–18.856 m/s가 모두 실측을 포함했다. 짧은 footprint 한 건에서의 차이는 물체·표면 전체에서의 승률이나 footprint의 인과 효과를 정하지 않는다. [전체 조건과 코드](tennis/run_tennis.py)

**시간별 patch에 대한 별도 점검.** Wilson ball의 normal impact에서는 5, 10, 15, 20 mm 반경에 도달한 시각이 각각 0.070, 0.218, 0.698, 1.408 ms였다. 실측 duration 4.72 ms와 입사속도 7.7 m/s를 주고, normal oscillator에 spherical-cap readout을 붙이면 restitution 0.75에서 0.049, 0.202, 0.479, 0.964 ms가 된다. 최대 지름도 실측 42±1 mm에 대해 47.17 mm로 컸다. 이 결과는 특정 readout의 말기 patch 팽창이 너무 빠르다는 진단이다. 시간 정보를 전혀 주지 않고 예측한 시험이나 oblique impact의 patch 검증으로 계산하지 않았다. [독립 wire-grid 검증](data69/wire_grid/RESULT_KO.md) · [서버 출력](data69/wire_grid/wire_grid_results.csv) · [원 측정](https://doi.org/10.1007/s12283-014-0159-x)

## 3. 1,023회 paired solver benchmark

각 impact에서 같은 inbound vector를 모든 solver에 주었다. 명목 golf ball radius 21.35 mm로 저자 단위를 SI로 환산했다. 반시계 방향 spin을 양으로 할 때 중심의 수평속도는 $v_x$, 강체 공의 최저점 수평속도는 $u=v_x+Rω$다. 모델은 **수평 COM 오차**와 **회전을 포함한 최저점 속도 오차**로 각각 평가했다. [Biber et al. §2.2–2.4](https://doi.org/10.1007/s12283-023-00442-4)

- **Cross-campaign holdout.** 한 turf에서 fitting·calibration을 마치고 다른 turf 전체를 예측했다. A source는 240 fit + 90 calibration, B source는 525 + 168이다. Source에서 연속 24개 행씩 묶어 block index modulo 4가 1인 블록을 residual calibration에 사용했다. Target outcome은 parameter·threshold·잔차 보정에 쓰지 않았다. 이 split은 표면과 함께 촬영 조건·입사 상태 분포도 달라진다.
- **Solver와 기준선.** 실제 PyBullet DIRECT, v68의 비대칭 normal spring과 Coulomb friction을 회전 구에 맞춘 normal-compliant, 여기에 tangential spring memory를 더한 변형을 비교했다. v68의 Wendland·Dispersive contact 자체를 재실행한 실험과는 관측 대상이 다르다. Source frequency, affine 및 source-trained two-regime affine를 함께 평가했다. 주 결과를 본 뒤 대안 설명으로 constant effective inclination을 가진 rigid model도 source에서만 fit했다. [Solver 설계](solver/plan.md) · [Empirical baseline 설계](analysis/PLAN_KO.md) · [Inclined model](analysis/inclined.py)
- **두 event.** Primary near-stick은 $|u_{out}|≤0.50$ m/s이다. Secondary sign reversal은 $u_{in}u_{out}<0$이다. 후자는 원 논문이 접촉 중 rolling을 거쳤을 가능성을 판단한 규칙을 따른다. 확률은 held-back source의 signed velocity residual을 예측에 더해 event 빈도를 구하고 $(k+0.5)/(n+1)$로 보정했다. 따라서 Brier는 **solver와 residual calibration의 결합 성능**이다. 특정 interface point가 실제로 붙어 있었는지는 별도 판독이 필요하다. [Solver 보고서](solver/SOLVER_KO.md) · [원문의 event 정의](https://doi.org/10.1007/s12283-023-00442-4)

| Fit → test | 방법 | COM MAE (m/s) ↓ | 최저점 MAE (m/s) ↓ | Near-stick Brier ↓ | Reversal Brier ↓ |
|---|---|---:|---:|---:|---:|
| A → B, 693회 | PyBullet / normal-compliant | 6.803 | 5.617 | .08276 | .18217 |
| | Tangential spring | 6.648 | 5.392 | .08231 | .25414 |
| | Inclined rigid, exploratory | 2.976 | 3.754 | .08109 | .17417 |
| | Two-regime affine | 2.913 | 3.766 | .07216 | .20602 |
| | Source event frequency | — | — | .07469 | .18323 |
| B → A, 330회 | PyBullet / normal-compliant | 4.100 | 3.704 | .02832 | .09556 |
| | Tangential spring | 4.150 | 3.949 | .04163 | .12672 |
| | Inclined rigid, exploratory | 2.014 | 3.023 | .03571 | .11445 |
| | Two-regime affine | 2.800 | 3.083 | .03348 | .11867 |
| | Source event frequency | — | — | .02493 | .23764 |

각 방향에서 near-stick은 56/693회(8.08%), 7/330회(2.12%)였고, sign reversal은 539/693회, 215/330회였다. **Source event frequency가 두 방향 모두에서 세 flat solver의 near-stick Brier보다 낮았다.** 반면 B → A에서 PyBullet의 reversal Brier는 .09556으로, event frequency의 .23764보다 낮았다. 같은 물리 예측이라도 어떤 event를 맞혀야 하는지에 따라 정보의 가치가 달라졌다. [RMSE·threshold 전수](solver/metrics.json) · [희귀 event 기준선](solver/marginal_baselines.json) · [Empirical model](analysis/empirical_summary.json)

![두 turf 사이의 속도 오차와 Brier, Wilson ball 두 사례의 복원 속도](analysis/real_impacts_v69.png)

마지막 panel의 파란색은 PyBullet/normal-compliant, 주황색은 tangential spring의 중심 예측이다. 검은 사각형과 가로선은 실제 trajectory의 속도 추정과 digitization envelope를 나타낸다. [벡터 그림](analysis/real_impacts_v69.pdf)

## 4. 차이는 어느 정도 믿을 수 있나

아래 Δ는 tangential spring에서 PyBullet을 뺀 값이다. 괄호는 같은 target 행을 paired resampling한 24행 block bootstrap의 95% 구간이다. 4,000회 resampling했고 12·48행 결과도 저장했다. 임의의 인접 block은 실제 session ID와 다르다. 이 구간은 정해진 두 turf 안의 효과를 기술하며, 새 turf 집단에 대한 CI나 여러 지표의 동시 CI는 아니다. [효과 차이와 민감도](analysis/paired_audit.json)

| Fit → test | Δ COM MAE (m/s) | Δ 최저점 MAE (m/s) | Δ near-stick Brier | Δ reversal Brier |
|---|---:|---:|---:|---:|
| A → B | −.155 [−.324, +.009] | −.225 [−.769, +.286] | −.00045 [−.00753, +.00594] | +.07197 [+.04491, +.10107] |
| B → A | +.050 [+.008, +.104] | +.246 [+.115, +.419] | +.01331 [+.00777, +.01891] | +.03116 [+.01667, +.05118] |

A → B에서는 관측 속도가 primary threshold의 ±0.25 m/s 안에 있는 81개 label을 달리 판독하면 Δ Brier가 −.00549부터 +.00323까지 가능했다. B → A에서는 같은 조건의 5개를 모두 고려해도 +.01331부터 +.01478이었다. 측정 오차의 분포를 알 수 없어 이 수치는 독립적인 worst-case 재판독 범위로 산출했다. Solver별 numerical check에서 PyBullet time step을 절반으로 해도 출력은 같았고, tangential spring을 phase당 80→160 steps로 늘려도 두 방향의 primary 비교 부호는 유지됐다. [재판독 결과](analysis/paired_audit.json) · [수치 검증](solver/SOLVER_KO.md)

**같은 turf의 반복 impact를 holdout한 경우.** 각 campaign에서 24행 block을 4-fold로 교대하여 모든 impact를 정확히 한 번 평가했다. Artificial turf의 near-stick Brier는 PyBullet .02083, memory .02614였고, natural turf에서는 .07467, .08453이었다. Artificial turf의 최저점 MAE는 memory가 4.636 → 3.347 m/s로 줄였다. 즉, 속도 오차의 개선이 event 확률의 개선으로 이어지지 않는 현상은 같은 campaign에서도 관찰됐다. 실제 specimen 단위의 holdout 여부는 확인할 수 없다. [Within-campaign out-of-fold 결과](solver/within_metrics.json)

## 5. 왜 solver가 같았고, 어느 가정이 오차를 남겼나

**평평한 고정면·한 접점·탄성 기억이 없는 Coulomb friction**에서는 수평 impulse가 마찰 한계에 도달하거나 최저점의 상대속도가 0이 될 때 멈춘다. 같은 normal impulse와 같은 inertia를 쓰면 접촉력의 시간 파형이 달라도 최종 수평속도가 같다. 실제 PyBullet 결과와 독립 rigid impulse 계산은 최대 7.2×10⁻¹⁵ m/s, normal-compliant와 독립 계산은 최대 3.6×10⁻¹⁴ m/s 차이였다. 서버에서 PyBullet 구현을 별도로 실행해 확인한 동등성이다. [구현과 오차](solver/SOLVER_KO.md)

**실측이 이 가정에서 얼마나 벗어나는가.** $k=I/(mR²)$일 때 flat single-contact의 수평 운동량·각운동량 식은 $Δv_x=kΔ(Rω)$를 동시에 만족해야 한다. 측정한 2차원 벡터 $(Δv_x,Δ(Rω))$와 이 직선 사이의 거리는 $|Δv_x-kΔ(Rω)|/√(1+k²)$다. 매 trial마다 정답을 본 oracle이 임의의 tangential impulse를 골라도 이 수직거리는 남는다. 각 성분을 같은 단위·가중치로 놓고 $k=0.4$를 쓰면, 두 성분을 합친 RMSE의 하한은 natural turf **5.923 m/s**, artificial turf **3.769 m/s**였다. 이는 각 target에서 실제 PyBullet의 joint MSE 중 **76.1%, 66.6%**에 해당한다. $k=2/3$로 바꿔도 하한은 3.863, 2.115 m/s였다. 이 하한은 noisy measured endpoint에 대한 조건부 진단이며, 오차 중 그 비율이 실제 patch 하나 때문에 생겼다는 분해가 아니다. [계산식과 sensitivity](analysis/paired_audit.py) · [결과](analysis/paired_audit.json)

**다른 설명도 실측을 예측한다.** Source에서 유효 contact normal의 기울기 하나를 friction·restitution과 함께 고르게 하자 각 source에서 11°, 20°가 선택됐다. Held-out COM MAE는 6.803 → 2.976, 4.100 → 2.014 m/s로 줄었다. 이 각도는 turf의 실측 기울기가 아니며, patch 이동·합력 방향·잔디 변형 등의 효과를 흡수할 수 있다. Near-stick Brier는 첫 방향에서만 줄었다. Biber et al.도 inclined-surface와 piecewise model을 이미 비교했다. 이 결과에서 독창성이 생길 지점은 inclination 자체보다, **실제로 관찰한 patch가 이 강한 대안에 추가하는 예측 정보가 있는가**다. [Exploratory fit](analysis/inclined.json) · [Biber et al. §2.5](https://doi.org/10.1007/s12283-023-00442-4)

## 6. 논문에 넣을 결론과 다음 검증

- **본문에 넣을 수 있는 주장.** 한 flat point-contact 가정을 공유하는 독립 구현 두 개가 실제 rebound에서도 같은 예측을 내고, 접선 변형의 기억을 도입해도 두 turf에서 event 예측 우위가 재현되지 않았다. 수평속도·spin에 대한 impulse 제약 위반 정도와 단순한 대안 모델을 함께 제시하면, solver 선택과 측정 대상 선택을 분리해 설명할 수 있다.
- **아직 열려 있는 주장.** 같은 instantaneous contact patch에서 실제 local stick/slip을 더 잘 맞히는 solver, 실물에서의 Wendland·Dispersive projection 손익 반전, patch를 추가 구매할 때의 decision gain은 이번 실측으로 판정할 수 없다. Wilson의 두 footprint는 geometry와 운동을 같은 물체에서 연결하지만, 두 사례로 형상별 효과나 calibration을 추정하기에는 부족하다.
- **가장 값진 다음 검증.** 같은 물체·같은 입사 조건에서 투명면 underside imaging과 side-view spin을 동기화한 trial에서, independently tracked local slip을 label로 고정한다. 비교군은 source-only inclined rigid와 two-regime affine, 실험군은 같은 입력에 time-resolved patch를 추가한 모델로 둔다. Object와 surface를 통째로 나눈 holdout에서 **속도와 patch-label Brier를 둘 다** 측정해야 patch의 추가 가치를 판정할 수 있다. 실물 장비 없이 먼저 확장할 수 있는 범위는 공개 racket 데이터의 surface-transfer velocity 비교다. [공개 racket 데이터](https://github.com/gossardt/tt_rackets)

## 7. 실행과 재현

- 서버 디렉터리는 `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/real_impact_v69`이다. Slurm allocation 2744471의 node56에서 사용 전 compute PID 없음, GPU memory 0 MiB를 확인했다. NVIDIA RTX PRO 6000에서 compliant integration과 solver scoring을 실행했고 PyBullet DIRECT는 동일 allocation의 CPU에서 구동했다. [GPU 확인](solver/gpu_before.txt) · [실행 기록](solver/rerun.log) · [Tennis 기록](tennis/run.json)
- [원 저자 데이터와 SHA-256](data69/DATA_KO.md), [Cross-campaign의 impact별 예측](solver/predictions.csv), [Within-campaign의 impact별 예측](solver/within_predictions.csv), [Tennis solver 예측](tennis/predictions.csv), [Paired uncertainty](analysis/paired_audit.json)를 함께 보관했다. Primary solver의 첫 수치 점검에서 Bullet default contact slop의 영향을 확인해 0으로 고정하고 전체를 다시 실행했다. [변경 이유](solver/numerical_note.md)
- 문헌 원문 검토, 측정자료 확보·figure 복원, 독립 solver 구현을 분담했다. 아래 부록에는 연구별 질문·표본·방법·한계, evidence synthesis, 상충하는 결과, 지식 지도, 후속 질문 10개와 현재 논문에 대한 검토를 수록했다.

## 부록 · 9가지 기준으로 검토한 문헌과 공개 자료

### 1. 연구별 비교

조사 기준은 반복한 물체의 identity, 개별 impact의 식별자, contact patch의 측정 시점·방식, 이탈 직후의 tangential velocity, 독립 stick/slip label이었다. Oblique impact, rebound spin, time-resolved contact area, footprint, grip/slip을 golf, tennis, rubber, softball, table tennis, tactile sensing과 조합해 검색하고 원 논문의 Methods, 저자 저장소, 저자가 제공한 영상·데이터를 대조했다. 아래의 부재 판단은 확인한 공개 자료 범위이다.

| 연구와 질문 | 방법·표본 | 결과와 결론 | 독립 검증에 사용할 때의 제약 |
|---|---|---|---|
| Arakawa 외 (2005). 입사 속도와 윤활이 면적과 운동을 어떻게 바꾸는가? | 42.6 mm, 46 g 두 겹 golf ball, PMMA, 30° 입사, 10–50 μs/frame. Fig. 2의 31.5 m/s impact에 Fig. 5와 8의 조건이 대응한다. | 입사 속도가 높으면 peak apparent contact area가 커졌다. 오일 처리는 면적·접촉시간 변화가 작아도 접선 이동과 회전을 바꿨다. | Fig. 11의 속도는 접촉 구간 변위/접촉시간이다. 저자가 공개한 값으로 이탈 직후 속도를 대체하면 시점이 달라진다. [논문 §2–7](https://www.jstage.jst.go.jp/article/jjsem/5/4/5_4_356/_pdf/-char/ja) |
| Arakawa (2017). 면적과 sliding speed를 사용해 회전을 설명할 수 있는가? | 3-piece golf ball, 28–61 m/s. PMMA에서 underside imaging, steel에서 high-speed angular tracking, roughness 통제를 위해 새 ball 사용. | 변하는 contact area를 포함한 friction model이 관측된 angular velocity 변화와 맞았다. | PMMA와 steel은 다른 target이다. Fig. 1의 두 조건을 event-level pair로 조인하거나 동일 specimen 반복으로 셀 수 없다. [Methods·Fig. 1–3](https://pmc.ncbi.nlm.nih.gov/articles/PMC5213996/) |
| Wang 외 (2022). initial spin은 접촉 흔적을 어떻게 바꾸는가? | steel sphere의 forward, backward, no spin, steel·rubber 조건. 얇은 carbon paper에 축 길이와 footprint를 기록. | spin 방향에 따라 기록된 footprint 치수가 달라졌고 rubber cushion이 그 효과를 바꿨다. | 유한 시간의 mark를 한 장에 누적하며 논문의 slip ratio는 footprint에서 유도한다. 각 pixel의 순간 slip 정답과 대응할 때 별도 검증이 필요하다. [논문·방법](https://doi.org/10.15632/jtam-pl/146621) |
| Cross (2002)와 저자 공개 실험 (2014). grip이 있으면 이탈 시까지 유지되는가? | 다섯 ball type의 normal·friction force 및 운동 측정. 2014 공개 예는 별도 실험으로, 60 mm rubber ball의 chalk footprint와 600 fps 영상을 연결한다. | 접촉 중 friction force 방향이 바뀌고 grip과 slip이 이어질 수 있다. 누적 footprint에 미끄러진 구간과 굴러간 구간이 함께 남을 수 있다. | 힘의 부호나 footprint 정지사진만으로 terminal slip을 정의할 수 없다. 2014 예의 속도를 동영상에서 별도로 추적했다. [논문](https://www.physics.sydney.edu.au/~cross/PUBLICATIONS/GripSlip.pdf) · [공개 실험](https://www.physics.usyd.edu.au/~cross/HAWKEYE.htm) · [v69 판독](cross_rubber/READOUT_KO.md) |
| Biber 외 (2023). 착지 전 속도와 spin에서 이탈 상태를 얼마나 정확히 예측하는가? | polyurethane cover의 multi-piece golf ball, 인조 turf와 천연 tee, 합계 1,000건 이상의 impact. 영상의 전후 kinematics와 반복성 평가. | 시험한 rigid law보다 piecewise-affine model의 적합도가 좋았다. 접촉 중 rolling을 거친 것으로 해석한 사례도 이탈 시 slip을 보였다. | 변형하는 turf의 local marker나 시간별 pressure map을 그들의 전후 속도로 역산하면 평가가 순환한다. [논문](https://doi.org/10.1007/s12283-023-00442-4) · [데이터](https://data.bris.ac.uk/data/dataset/1cwnj42n63q5w2sc843iww7bbd) |
| Mabbs·Smith (2026). 질량 분포 및 friction law가 softball impact 예측을 바꾸는가? | 0.2 kg softball, steel plate, 20.1–63.5 m/s 범위. 150 kHz force와 4,700 fps 측면 추적. 대표 sliding/gripping 각 1건을 14,100 fps로 분석. | 시간에 따라 friction을 바꾼 FE가 constant dynamic friction보다 각속도 normalized RMSE를 29% 줄였다. 질량 분포를 교정하면 이탈 각속도의 과대예측도 줄었다. | raw data는 저자 요청 방식이고 수평·회전 측정에 기반한 regime과 local patch label을 구분해야 한다. [Methods·Data availability](https://doi.org/10.1007/s12283-026-00564-5) |
| Gossard (2026, preprint). 서로 다른 racket rubber에서 state-dependent impulse의 예측력은 어떠한가? | 10가지 고정 racket 구성. 50 ms local regression으로 속도, dotted ball로 spin 추정. 저자 저장소는 original 8,184건, fixed train 6,490건, fixed test 1,623건을 명시. | Gaussian process로 추정한 상태별 계수가 고정 계수보다 이탈 속도·spin 오차를 줄였다고 보고한다. | 논문에 수록된 수집 수(8,194건)와 공개 original 수가 다르다. CSV에는 frame-level patch·trial별 ball serial·독립 stick/slip label 열이 없다. [논문](https://arxiv.org/html/2604.11349v1) · [확인한 저장소 revision](https://github.com/gossardt/tt_rackets/tree/159c6f040dd7832570b537b3b5fd00a97fc28c15) |

저자 저장소 30개 CSV의 실제 행 수, 열 이름 및 SHA-256은 [schema audit](gossard_schema_audit.json)에 기록했다.

### 2. 주제별 근거

실제 접촉면을 보려면 투명한 target 뒤에서 경계를 관측하는 방법이 있다. 이때 dimple을 포함한 apparent ellipse와 실제 닿는 면적을 분리할 수 있다. 반면 carbon paper나 chalk는 접촉점이 지나간 모든 위치가 겹쳐 남아, 최대 순간 면적과 누적한 면적이 다를 수 있다. [Arakawa 외](https://www.jstage.jst.go.jp/article/jjsem/5/4/5_4_356/_pdf/-char/ja) · [Wang 외](https://doi.org/10.15632/jtam-pl/146621) · [Cross](https://www.physics.usyd.edu.au/~cross/HAWKEYE.htm)

이탈 운동은 free-flight frames를 사용해 위치와 orientation을 각각 적합할 수 있다. Biber의 turf와 Gossard의 racket은 그 분석을 할 수 있는 공개 예이다. 별도의 TT3D도 130개 trajectory와 200 Hz multi-camera reconstruction을 제공하지만, 공개 명세의 목적은 bounce 위치와 궤적 복원이다. EventTouch는 실제 tactile sensor, camera frames와 FEM node 데이터를 구분해서 배포하며, pose·press depth별 접촉 실험이므로 free rebound 평가로 옮기면 측정 과제가 달라진다. [Biber](https://doi.org/10.1007/s12283-023-00442-4) · [Gossard](https://github.com/gossardt/tt_rackets) · [TT3D 저자 저장소](https://cogsys-tuebingen.github.io/tt3d/) · [EventTouch 공식 명세](https://huggingface.co/datasets/BAAI/EventTouch)

### 3. Knowledge map

입사 속도·spin·ball inertia와 표면 반응 → 접촉 중 deformation, 면적, local stick/slip → 접선·법선 force와 torque의 시간적 적분 → 이탈 시 center-of-mass velocity와 spin → 실험자가 선택한 kinematic event와 Brier score로 연결된다. Arakawa는 앞쪽 면적, Cross와 Mabbs·Smith는 가운데의 힘·회전, Biber와 Gossard는 뒤쪽의 전후 상태를 주로 측정했다. 한 층에서의 높은 예측력이 이웃한 층을 식별하는지는 별도로 시험해야 한다. [Arakawa](https://pmc.ncbi.nlm.nih.gov/articles/PMC5213996/) · [Cross](https://www.physics.sydney.edu.au/~cross/PUBLICATIONS/GripSlip.pdf) · [Mabbs·Smith](https://doi.org/10.1007/s12283-026-00564-5) · [Biber](https://doi.org/10.1007/s12283-023-00442-4) · [Gossard](https://arxiv.org/html/2604.11349v1)

### 4. Research gaps

- **동일 impact에서의 공간·시간 정렬.** PMMA 연구의 평균 속도와 turf 연구의 이탈 속도는 각각 측정 정의가 다르다. 두 사실만으로 시간별 patch와 이탈 오차의 event-level 관계를 알 수 없다. [Arakawa 외](https://www.jstage.jst.go.jp/article/jjsem/5/4/5_4_356/_pdf/-char/ja) · [Biber 외](https://doi.org/10.1007/s12283-023-00442-4)
- **Specimen과 wear.** 매번 새 공을 쓴 연구와 ball serial을 제공하지 않은 배열은 같은 실물의 반복성과 개체 간 변동을 분리하는 자료로 쓰기 어렵다. [Arakawa Methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC5213996/) · [Gossard 데이터](https://github.com/gossardt/tt_rackets)
- **Local slip의 독립성.** Force reversal, 이탈 spin 및 흔적 모양은 서로 관련되지만, 한 측정량으로 나머지를 자동 판정할 근거는 약하다. 접촉 중 grip과 말기 재슬립이 함께 일어나는 사례가 있다. [Cross](https://www.physics.sydney.edu.au/~cross/PUBLICATIONS/GripSlip.pdf) · [Biber](https://doi.org/10.1007/s12283-023-00442-4)

### 5. 상충해 보이는 결과

PMMA에서는 윤활에 따라 접선 운동은 달라져도 peak patch의 변화가 작았고, steel/rubber에 initial spin을 가한 실험에서는 footprint 크기가 달라졌다. 조작한 변수(윤활 / spin), 접촉쌍(공·PMMA / steel·cushion), endpoint(순간 광학 면적 / 누적 자국)가 모두 다르므로 어느 하나를 일반 법칙으로 선택할 근거가 부족하다. Softball FE에서 temporal friction을 넣어 각속도 예측이 좋아진 결과와, turf에서 간단한 piecewise model이 선전한 결과 역시 매질·예측 대상·비교군의 차이를 수반한다. [Arakawa 외](https://www.jstage.jst.go.jp/article/jjsem/5/4/5_4_356/_pdf/-char/ja) · [Wang 외](https://doi.org/10.15632/jtam-pl/146621) · [Mabbs·Smith](https://doi.org/10.1007/s12283-026-00564-5) · [Biber](https://doi.org/10.1007/s12283-023-00442-4)

### 6. 방법이 결론에 미치는 영향

| 방법 | 식별하는 물리량 | 남는 오차 |
|---|---|---|
| Under-target high-speed imaging | 특정 시각의 optical contact mask | 광학 threshold, dimple 포함 여부, roughness에 따라 면적 변화. [Arakawa 외](https://www.jstage.jst.go.jp/article/jjsem/5/4/5_4_356/_pdf/-char/ja) |
| Carbon/chalk footprint | 접촉의 누적 범위, 이동 방향 단서 | 접촉 시각과 sliding distance가 중첩. [Wang](https://doi.org/10.15632/jtam-pl/146621) · [Cross](https://www.physics.usyd.edu.au/~cross/HAWKEYE.htm) |
| 두 개 이상의 free-flight frames와 marker tracking | COM velocity, 각속도, 전후 spin 방향 | 감쇠, frame rate, 표면의 움직임 및 rigid-radius 근사. [Biber](https://doi.org/10.1007/s12283-023-00442-4) · [Gossard](https://arxiv.org/html/2604.11349v1) |
| Force plate와 FE | Normal·shear resultant 및 그 시간적 구조 | 같은 합력이 다른 traction 분포에서 생길 수 있다. Mabbs·Smith 연구에서 peak shear timing 오차가 남았다. [Mabbs·Smith](https://doi.org/10.1007/s12283-026-00564-5) |

### 7. Evidence synthesis

**강한 근거**는 여러 실물 시스템에서 이탈 translation과 rotation이 서로 다른 측정량이며, 특정 조건에서 grip 뒤에 다시 slip이 생긴다는 점이다. **중간 근거**는 상태나 재료에 맞춘 friction response가 일부 검증 조건에서 예측 오차를 줄인다는 것이다. **부족한 근거**는 이 결과가 특정 v68 projection의 우위를 결정하거나, 소수의 누적 footprint만으로 새 contour에서 local stick/slip Brier를 예측한다는 주장이다. [Cross](https://www.physics.sydney.edu.au/~cross/PUBLICATIONS/GripSlip.pdf) · [Biber](https://doi.org/10.1007/s12283-023-00442-4) · [Mabbs·Smith](https://doi.org/10.1007/s12283-026-00564-5) · [Gossard](https://arxiv.org/html/2604.11349v1)

### 8. 근거에서 도출한 후속 질문 10개

| 질문 | 중요한 이유·측정 변수 | 식별 설계 및 근거 |
|---|---|---|
| 1. 누적 면적이 순간 최대 면적을 얼마나 과대평가하는가? | footprint와 peak area의 불일치 | 투과 영상 mask를 시간에 걸쳐 합친 면적과 각 frame 최대값을 같은 trial에서 비교. [Arakawa 외](https://www.jstage.jst.go.jp/article/jjsem/5/4/5_4_356/_pdf/-char/ja) |
| 2. 윤활이 같은 면적에서 접선 속도를 바꾸는가? | 면적만 쓰는 predictor의 실패 조건 | 같은 공·입사각·속도로 dry/oil 순서를 무작위화하고 마찰 변화와 separation velocity를 함께 측정. [Arakawa 외](https://www.jstage.jst.go.jp/article/jjsem/5/4/5_4_356/_pdf/-char/ja) |
| 3. Spin 방향이 mark의 이동 길이와 넓이를 각각 바꾸는가? | footprint 축 비율의 물리적 해석 | matched spin magnitude로 forward/backward를 바꾸고 고속 영상의 centroid path와 자국을 조인. [Wang 외](https://doi.org/10.15632/jtam-pl/146621) |
| 4. Terminal low-speed가 mid-contact grip을 얼마나 놓치는가? | label의 시간축 검증 | 동일 impact를 local optical flow track과 이탈 rigid kinematics로 이중 판독. [Cross](https://www.physics.usyd.edu.au/~cross/HAWKEYE.htm) · [Biber](https://doi.org/10.1007/s12283-023-00442-4) |
| 5. 겉면만 교체해도 solver 순위가 바뀌는가? | roughness·compliance 교란 | 같은 blade에서 rubber를 바꾸고 incoming state를 매칭해 velocity MAE를 추적. [Gossard](https://arxiv.org/html/2604.11349v1) |
| 6. 알려진 inertia를 넣으면 속도와 spin 순위가 함께 변하는가? | geometry 외 대안 설명 | 측정 inertia를 고정·교체한 predictive ablation을 같은 test shots에 수행. [Mabbs·Smith](https://doi.org/10.1007/s12283-026-00564-5) |
| 7. Surface-relative velocity가 COM 기준보다 더 안정적인가? | 움직이는 표면에서의 정의 | 표면 marker displacement를 빼기 전후의 event rate와 Brier를 비교. [Biber](https://doi.org/10.1007/s12283-023-00442-4) |
| 8. Rebound velocity fit window가 solver 순위를 바꾸는가? | 운동 추정 자체의 민감도 | 동일 영상을 여러 짧은 window로 처리하고 paired velocity difference를 재계산. [Biber](https://doi.org/10.1007/s12283-023-00442-4) · [Gossard](https://arxiv.org/html/2604.11349v1) |
| 9. 같은 물체의 마모가 specimen 간 차이만큼 큰가? | repeat와 독립 표본의 구분 | object serial, session, impact order를 기록해 variance components 추정. [Arakawa](https://pmc.ncbi.nlm.nih.gov/articles/PMC5213996/) |
| 10. Local patch를 추가 측정할 비용이 예측 개선을 정당화하는가? | v68과 연결할 수 있는 contribution | 서로 독립적으로 기록한 전후 운동·patch·label로 acquisition 전후 Brier를 비교하고 held-out surfaces에서 재평가. [Arakawa 외](https://www.jstage.jst.go.jp/article/jjsem/5/4/5_4_356/_pdf/-char/ja) · [Biber](https://doi.org/10.1007/s12283-023-00442-4) |

### 9. 현재 논문에 대한 심사 의견

논문에는 각 표의 **관측 단위와 label 생성식**을 넣는 편이 좋다. 예컨대 Biber의 velocity benchmark, 이탈 rigid surface speed로 만든 event, local stick/slip을 세 줄로 구분하면 같은 단어가 서로 다른 측정을 가리키는 문제를 피할 수 있다. v68의 contour별 projection 부호가 실물에서도 유지된다고 주장하려면 같은 contour의 반복 trial에서 patch와 local motion을 독립적으로 동기화해야 한다. 현재 자료로는 공개 velocity에 대한 예측 실패, 특정 solver 가정이 남기는 residual, 이탈 event 정의에 대한 민감도가 더 직접적인 기여다. 강한 비교군은 source-only fitted piecewise model이나 rubber-dependent impulse model이며, 물체·표면 전체를 분리한 검증이 뒤따라야 한다. [Biber](https://doi.org/10.1007/s12283-023-00442-4) · [Gossard](https://arxiv.org/html/2604.11349v1)
