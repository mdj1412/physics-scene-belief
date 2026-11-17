# v49 · 어느 위치에서 가볍게 누른 변형이 이후의 stick/slip을 예측하는가?

**약한 preload의 변형 방향이 full load까지 유지되면, 독립적인 PyBullet contact solver에서도 query-local direct calibration이 같은 예산의 fixed probe와 projection gate를 모두 이겼다.** 방향이 항상 유지되고 공간적 연결이 중간인 조건에서 held-out Brier loss는 각각 **0.144499, 0.151312, 0.152049**였다. 반면 방향이 항상 뒤집힐 때 기존 direct calibration은 **0.191095**까지 악화됐다. 같은 측정값을 반대 부호로 해석하는 진단에서는 **0.144941**까지 회복됐다. 이 결과는 한 번의 접촉이 무엇을 알려주는지와 decoder가 그 신호를 제대로 사용하는지를 구별한다.

**논문에서 주장할 수 있는 범위**는 물체마다 query 근처를 누르는 것이 언제 유용한지를 두 가지 물리 조건으로 나누고, 같은 측정값·같은 예산 아래에서 그 경계가 solver를 바꿔도 얼마나 유지되는지 보였다는 것이다. 개별 곡면 label은 solver 해상도를 바꾸면 최대 5.56% 달라졌고, 정확한 교차점도 이동했다. 아래의 유지율은 simulator에 지정한 값이다. 실제 재질의 유지율이나 배포 시 이미 알고 있는 값으로 해석할 수 없다.

## 1. 무엇을 바꿨고 무엇을 고정했나

- **위치 간 mechanical coupling, ρ.** Query station과 12 mm 떨어진 station이 같은 weak mechanical state를 복사할 확률을 0, 0.5, 1로 바꿨다. 복사하지 않으면 inert·negative·positive를 각각 1/2·1/4·1/4 확률로 독립 추출한다. 예컨대 ρ=0에서도 우연히 같은 상태가 될 확률은 37.5%다. ρ=1이면 두 위치의 상태가 항상 같지만 station별 sensor noise와 gain은 별개다.
- **하중 간 방향 유지확률, κ.** Query에서 active인 상태의 weak-to-full 부호가 유지될 확률을 0, 0.25, 0.5, 0.75, 1로 지정했다. 나머지는 부호를 뒤집고 inert는 유지했다. κ=0.5에서는 active인지에 관한 정보가 남지만, active의 부호는 full-load 부호를 알려주지 않는다. κ=0은 측정 부호를 뒤집으면 완전히 예측 가능한 환경이다.
- **측정과 정책.** Query 높이는 19 또는 31 mm다. 0.4 N preload를 한 번 가하고 두 fiducial 사이의 signed differential displacement 하나를 읽는다. Local direct는 query 위치, fixed direct는 source에서 선택한 31 mm에서 같은 센서를 사용한다. 둘 다 original posterior의 stick 확률을 보정한다. Local gate는 같은 query 측정으로 original 또는 projected decoder를 선택한다. Source에서 동결한 geometry·calibration 계수·fixed station을 썼다. 세 정책 모두 조건 ID, ρ, κ, target label을 볼 수 없다. Fixed의 remote 계수는 v48의 공간 계수 0.60에서 학습됐다. [정책 구현](policy49.py) · [동결한 예측](forecast_freeze49.sha256)
- **평가 단위.** v47에서 학습한 네 family와 분리된 v48의 ogee·cove·reed·flute, 12개 catalog의 6,144개 object draw를 두 solver에 똑같이 넣었다. 각 catalog의 512개 draw에 두 높이·두 tip radius·세 offset·네 noise draw 등 matched 조건을 적용했다. 15개 cell은 같은 물체들에 대한 factorial intervention이다. 새 target acquisition이나 15개의 독립 dataset으로 세지 않았다. [사전 계획](PLAN_KO.md)

## 2. Contact solver의 독립성과 검산

v48은 분포를 지정한 pressure footprint로 impact를 계산했다. v49는 locked-spin sphere를 변형된 contour의 triangle mesh에 충돌시키고, Bullet이 접촉 normal 및 normal/friction impulse를 계산하도록 했다. 충돌 시작 시 위치·속도·표면에 따른 friction을 지정하며 두 rate law를 사용한다. 같은 수치 0.25/2 mm는 v49에서 **구형 tip의 radius**, v48에서 **pressure footprint의 radius**다. 두 모델의 차이에는 contact law 및 radius의 물리적 의미가 함께 포함된다. Weak-load sensor와 full-load mesh 형상은 주어진 변형 모형을 공유한다. Bullet이 고무의 elasticity나 weak-to-full 대응 자체를 예측한 실험은 아니다. [Solver 구현](solver49.py) · [독립 solver protocol](SOLVER49_PROTOCOL.md) · [Bullet 공식 문서](https://github.com/bulletphysics/bullet3/blob/master/docs/pybullet_quickstart_guide/PyBulletQuickstartGuide.md.html)

Primary endpoint는 충돌 후 첫 contact tangent 방향 잔류 속도가 0.020 m/s 이하인 사건이다. 884,736개 factorial label slot 중 family 사이에서 동일한 inert 형상을 공유해 실제로 계산한 primary trajectory는 663,552개다. 평판에서는 tangent speed와 restitution이 해석해와 약 10⁻¹⁶ 정밀도까지 맞았다. 곡면에서는 두 solver의 label이 state prior로 가중한 3.91%에서 달랐다. 같은 물체에서 full-load 부호를 −에서 +로 바꾸면 label이 바뀌는 비율은 v48의 34.27%에서 v49의 35.83%로 커졌다. v49에서 평균 stick 비율은 inert 53.31%, negative 68.13%, positive 32.36%였다. [Solver 검산](solver49_audit.json) · [평판 결과](solver49_flat.json)

## 3. Held-out Brier loss가 낮아지는 조건

중간 수준의 공간 연결(ρ=0.5)에서 추가 접촉을 쓰지 않는 original의 Brier는 0.162529, 접촉 없이 source 보정만 적용한 direct-none은 0.158422였다. 마지막 열은 κ를 알고 측정값의 부호에 따른 두 예측을 혼합할 때의 진단이다. 배포 정책 비교의 winner를 정할 때는 제외했다.

| 방향 유지확률 κ | Query-local direct | Fixed-site direct | Query-local projection gate | κ를 알려준 direct diagnostic |
|---:|---:|---:|---:|---:|
| 0 | 0.191095 | 0.176339 | 0.162494 | 0.144941 |
| 0.25 | 0.179446 | 0.170082 | 0.159883 | 0.154936 |
| 0.50 | 0.167797 | 0.163826 | 0.157272 | 0.158194 |
| 0.75 | 0.156148 | 0.157569 | 0.154660 | 0.154715 |
| 1 | **0.144499** | 0.151312 | 0.152049 | 0.144499 |

아래는 각 baseline에서 뺀 **local direct의 Brier 차이**다. 음수면 local direct가 더 정확하다. †는 사전 지정한 30개 비교를 동시에 보정한 95% CI가 0을 제외한다는 뜻이다. 두 local 정책은 같은 위치에서 관측하기 때문에 이들 간 차이는 ρ와 관계없다.

| κ | Fixed 대비, ρ=0 | Fixed 대비, ρ=0.5 | Fixed 대비, ρ=1 | Local gate 대비, 모든 ρ |
|---:|---:|---:|---:|---:|
| 0 | +0.019315† | +0.014756† | +0.010196† | +0.028601† |
| 0.25 | +0.011519† | +0.009363† | +0.007208† | +0.019563† |
| 0.50 | +0.003723† | +0.003971† | +0.004219† | +0.010525† |
| 0.75 | −0.004073† | −0.001421† | +0.001231† | +0.001488 |
| 1 | −0.011869† | −0.006813† | −0.001758† | −0.007550† |

- **두 baseline에 대한 동시 우위**는 15개 grid cell 중 κ=1인 3개에서 확인했다. 각 cell의 각 비교에서 12개 catalog가 모두 같은 방향이었다. ρ=0.5, κ=1에서 local − fixed는 −0.006813 [−0.008066, −0.005561], local − gate는 −0.007550 [−0.010229, −0.004871]이었다.
- **경계에 가까운 κ=0.75에서는 어떤 baseline을 고르느냐가 중요하다.** ρ=0과 0.5에서 fixed보다 좋았지만, gate 대비 차이 +0.001488의 simultaneous CI는 [−0.000650, +0.003625]였다. ρ=1에서는 fixed가 유의하게 앞섰다. κ≤0.5에서는 세 공간 조건 모두에서 frozen local direct가 두 baseline에 졌다.
- **연속 경계의 점추정.** 이 실험의 exact mixture에서 기대 차이는 ρ를 고정하면 κ에 선형이다. 평균 Brier가 두 baseline을 모두 앞서기 시작하는 κ는 ρ=0, 0.5, 1에서 각각 **0.791, 0.791, 0.853**이었다. Catalog bootstrap의 exploratory pointwise 95% 구간은 각각 [0.764, 0.820], [0.764, 0.820], [0.826, 0.886]이다. 각각의 경험적 위험곡선이 교차하는 위치이고, 새 소재에서 보장되는 유지율은 아니다. [모든 cell의 Brier와 CI](score49.json) · [경계 검산](crossings49.json)

![방향 유지율에 따른 Brier loss, fixed direct 대비 차이, local gate 대비 차이](contact_transfer_v49.png)

그림 B·C의 음영은 두 primary contrast와 15개 cell 전체를 동시에 보정한 구간이다. 각 catalog에서 noise와 condition을 먼저 평균한 뒤, 12개 catalog의 paired 차이에 Student t와 Bonferroni를 적용했다. 표본 수 12에서의 interval은 catalog 간 차이에 관한 추론이다. Conditional state law를 정확히 적분했고, 별도로 binary label에 대한 squared error를 직접 계산했다. [채점 코드](score49.py) · [Catalog별 결과](per_catalog49.npz)

## 4. 센서와 calibration의 기여를 어떻게 분리했나

**위치의 효과.** Query가 31 mm이면 local과 fixed가 같은 위치·같은 측정을 쓰고 모든 예측이 정확히 일치했다. ρ=0.5, κ=1에서 19 mm에 한정하면 local 0.154943, fixed 0.168570이었다. 31 mm에서는 둘 다 0.134054였다. 두 위치의 상태가 같은 ρ=1에서도 원래 local이 fixed보다 앞선 결과에는 source에서 학습한 remote calibration의 영향이 섞인다. Fixed 측정에도 동일한 local 계수를 쓰면 κ=1에서 local − fixed는 **+0.000138 [−0.000607, +0.000883]**이었다. 이 조건에서 순수한 위치 이득을 확인할 수 없었다. [동일 calibration control](score49.json)

**측정 부호의 해석.** 알려진 κ로 두 예측을 혼합하는 진단은 실제로 읽은 noisy scalar d에 대해 κ·p(d)+(1−κ)·p(−d)를 출력한다. 숨은 개별 물체의 state나 outcome은 쓰지 않는다. 항상 반전하는 환경에서는 부호를 뒤집기만 해도 Brier가 0.191095에서 0.144941로 내려갔다. κ=0.5에서는 같은 진단의 0.158194와 direct-none의 0.158422 사이의 차이 −0.000227 [−0.000523, +0.000068]에 0이 포함됐다. 규칙적으로 반전하는 환경과 방향이 무작위인 환경은 서로 다른 실패다. Target label 없이 새로운 환경의 κ를 추정할 수 있는지는 다음 과제로 남는다. [부호 대조 실험](transfer_controls49.json)

**Solver의 효과.** 예측과 두 transfer factor를 그대로 두고 v48 label로 다시 채점하면 local direct와 gate의 교차점은 κ=0.727이었다. PyBullet에서는 0.791로 이동했다. ρ=0.5, κ=0.75에서 local − gate 점추정이 −0.000790에서 +0.001488로 바뀌었다. 두 solver 사이 paired contrast 변화량은 +0.002277 [0.001407, 0.003148]이었다. 각각의 solver에서 단독으로 이 조건의 winner를 확정할 수는 없다. [동일 예측의 solver 교체](transfer_controls49.json)

## 5. GPU 시간과 물리 sensing·reset 비용

서버에 할당된 네 GPU 중 두 장은 다른 process가 각각 약 40 GB를 사용하고 있었다. 빈 두 allocation에서 역할을 나눴다. node55의 Slurm CPU에서는 Bullet label과 통계 검산을, node56의 RTX PRO 6000 Blackwell에서는 frozen forecast와 GPU crossover를 실행했다. Timing은 전체 factual acquisition, 선택된 sensor 포장, gate 또는 direct update, 실제 선택한 decoder 한 번, GPU synchronization을 포함한다. 별도 simulator가 label을 생성하는 시간은 배포 경로에 들어가지 않는다.

| 배포 정책 | GPU p90, ms/512 objects [pointwise 95% CI] | 추가 touch / reset, 물체당 | 기본 sensing + reset, 물체당 |
|---|---:|---:|---:|
| Always-original | 432.12 [431.79, 432.74] | 0 / 0 | 0 |
| Query-local direct | 431.87 [431.67, 432.07] | 1 / 1 | 250 + 190 ms |
| Fixed-site direct | 432.19 [431.90, 432.85] | 1 / 1 | 250 + 190 ms |
| Query-local gate | 436.03 [435.59, 436.40] | 1 / 1 | 250 + 190 ms |

(ρ,κ)=(0,0),(1,1) 두 극단에서 12개 catalog마다 warmup 뒤 3회씩, 정책당 72개 timed batch를 무작위 순서로 실행했다. 16,000회 paired catalog bootstrap에서 **local direct − fixed direct는 −0.32 ms [−0.88, +0.02]**, **local direct − local gate는 −4.15 ms [−4.59, −3.79]**였다. Local과 fixed 간 compute 순위는 확정할 수 없다. GPU에서 실행한 예측과 보관한 예측의 최대 확률 차이는 4.44×10⁻¹⁶이었다. [GPU timing](timing49.json) · [측정·검산 상세](GPU_KO.md) · [별도 policy replay](audit_policy49.json)

**물리 장비 시간은 가정값이다.** Approach 100 ms, 0.4 N ramp 100 ms, readout 50 ms, unloading 50 ms, retreat 100 ms, settle 40 ms를 두면 횡이동을 제외한 추가 시간은 440 ms/object다. Query에서 출발해 복귀하고 60 mm/s로 이동하면 fixed만 19 mm trial에서 24 mm를 왕복한다. 두 높이를 똑같이 가중하면 local/gate는 440 ms, fixed는 640 ms다. 두 station의 중점에서 출발해 복귀하면 모두 640 ms다. Force 명령만으로 소비 energy를 추정하지 않았다. 실제 장비 wall time, reset failure, 손상·잔류 변형은 미측정이다.

**돈을 내고 한 번 더 누를 이유도 방향 유지율에 달려 있다.** 같은 direct-none과 비교한 local의 평균 Brier 이득은 κ=1에서 0.013923, κ=0.75에서 0.002274, κ=0.5에서는 오히려 0.009375 손해였다. 기본 cycle 0.44초를 쓸 때 이득이 시간 비용을 상쇄하는 최대 penalty는 κ=1에서 약 0.03164 Brier/s, κ=0.75에서 약 0.00517 Brier/s다. 이 penalty는 로봇의 실제 비용을 정한 값이 아니라 배치 조건에 대입할 손익분기점이다. [No-extra 대비 paired 결과](score49.json)

## 6. 수치 민감도와 논문 기여

처음 계획한 finer subset에서 label 차이가 보여 같은 12개 catalog 전체를 두 단계 더 높은 해상도에서 계산했다. 모든 행에서 source calibration과 예측은 그대로다. 아래 loss는 ρ=0.5, κ=1에서의 값이다.

| Solver 해상도 | Time step / mesh spacing | 전체 weighted stick 비율 | Local direct | Fixed direct | Local gate | Local − gate 교차 κ 점추정 |
|---|---|---:|---:|---:|---:|---:|
| Primary | 0.025 ms / 0.050 mm | 51.78% | 0.144499 | 0.151312 | 0.152049 | 0.791 |
| Fine | 0.0125 ms / 0.025 mm | 50.07% | 0.145539 | 0.152551 | 0.152178 | 0.819 |
| Ultra | 0.00625 ms / 0.0125 mm | 46.55% | 0.150738 | 0.157894 | 0.155882 | 0.862 |

**세 해상도 모두 두 baseline에 대한 동시 우위는 κ=1의 세 cell에서 재현됐다. 다만 gate와의 우열이 바뀌는 교차점 추정은 0.791에서 0.862까지 이동했다.** Primary와 fine은 label의 2.13%, fine과 ultra는 3.61%, primary와 ultra는 5.56%에서 달랐다. 초기 계획의 1/16 subset에서 첫 impulse 이후 측정까지의 물리 시간도 맞췄지만 primary–ultra 차이는 5.75%에서 5.67%로만 줄었다. 곡면의 label·절대 Brier·정확한 경계가 수치적으로 수렴했다고 할 수 없다. Threshold를 0.010·0.020·0.040 m/s로 바꾼 별도 검산에서도 90개 contrast의 simultaneous interval 기준 동시 우위 cell은 같은 셋이었다. [Fine 재채점](refinement49.json) · [Ultra 재채점](ultra_policy49.json) · [측정 시각 control](clock_audit49.json) · [Threshold 검산](threshold49.json) · [Solver 상세](SOLVER_KO.md)

**논문 기여로 발전시킬 부분은 같은 센서의 성공과 실패를 예측하는 조건 비교다.** 공간적으로 다른 물체에서 local 측정의 이득은 방향이 전달될 때 커지고, 체계적으로 반전할 때는 강한 신호가 오히려 frozen decoder의 loss를 키웠다. 동일 calibration 대조와 부호 반전 대조 덕분에 위치 자체의 이득, predictor의 transfer 실패, 신호가 실제로 약한 조건을 구별할 수 있다. 두 solver에서 극단의 순위가 재현된 점은 개념적 근거를 강화한다. 두 solver에서 공유한 contour와 sensor의 하중 응답 가정, 12개 catalog의 재사용, 곡면 해상도 의존성은 실물 일반화와 정밀한 cutpoint 주장을 제약한다.

**다음 확인 실험.** 새 물체의 두 station에서 weak·full load의 signed displacement를 독립 영상으로 읽어 ρ와 κ에 해당하는 경험적 양을 먼저 추정한다. 실제 측정에서는 near-zero와 부호 판독 실패도 별도 기록한다. 다른 session에서 local/fixed/no-extra를 무작위 배정하고 force·reset displacement·cycle time·full-load stick/slip을 독립 계측한다. Source에서 정한 방향을 그대로 쓰는 정책, 소수의 환경 calibration으로 polarity를 추정하는 정책, 부호를 무시하는 정책을 같은 sensing budget으로 비교하면 현 단계의 가장 큰 약점인 *현장에서 어느 신호를 믿을지 알 수 있는가*를 직접 시험할 수 있다.

## 문헌 검토 · 한 번의 preload가 다른 접촉을 예측하는 조건

기준 질문은 *같은 예산에서 접촉할 위치를 미리 가볍게 누르면, 다른 위치를 눌렀을 때보다 이후 별도의 full-load 사건을 더 정확히 예측하는가*이다. 아래 여섯 편은 실물에서 확인된 접촉 신호, 반복 측정의 해석, 독립 solver 검증을 다룬다. 같은 접촉 중의 slip detection과 reset 뒤 다른 접촉의 slip forecasting은 시간 순서와 예측 대상이 다르므로 수치를 합산하지 않았다.

| 연구 | Research question과 methodology | 표본·측정 단위 | 핵심 결과와 결론 | v49로 옮길 때의 한계 |
|---|---|---|---|---|
| Yu et al., IROS 2016 | 마찰이 위치·방향·접촉 조건에 따라 얼마나 달라지는가? Robot planar pushing에서 pose와 force를 기록했다. | object·surface 조합당 약 6,000 pushes, push당 평균 약 200 time samples | 공간과 반복에 걸친 friction 변동을 관찰했다. 제목의 million은 push 수가 아니라 motion·force records를 센 것이다. | Pushing trajectory에서 별도의 weak-to-full 변형 방향을 측정하지 않았다. [원문](https://arxiv.org/pdf/1604.04038) |
| Ma & Rodriguez, RA-L 2018 | 반복 push의 결과가 여러 군집으로 나뉘는 이유는 무엇인가? 앞 연구의 데이터를 다시 분석하고 anisotropic friction simulation과 대조했다. | 같은 자료 중 2,000회 반복한 push sequence. 독립적인 두 번째 실물 cohort로 세면 안 된다. | 시작 orientation과 수집 순서의 편향, material heterogeneity, friction anisotropy를 조합하면 관측된 구조의 상당 부분을 설명할 수 있었다. | 같은 관측을 설명할 수 있는 원인을 좁힌 결과다. 새 물체의 국소 부호나 모든 변동의 결정론적 원인을 확정하지 않는다. [원문](https://arxiv.org/pdf/1802.10089) |
| du Bois de Dunilac et al., J. R. Soc. Interface 2023 | Normal force가 torsional slip을 어떻게 바꾸는가? 유리판을 눌러 회전하며 optical imaging으로 피부 표면을 추적했다. | 성인 7명의 오른쪽 집게손가락, normal force 0.5–10 N, 별도의 speed sweep | 가장자리부터 slip이 진행됐다. Normal force가 높으면 대체로 full slip에 도달하기 위한 회전각이 커졌고 개인 간 차이도 컸다. | 사람 피부와 유리의 한 contact patch 안에서 본 현상이다. 두 station 간 transfer나 물체의 weak/full polarity 전환율을 측정하지 않았다. [원문](https://pmc.ncbi.nlm.nih.gov/articles/PMC10113816/) |
| Willemet et al., ICRA 2024 | Tactile deformation으로 slip까지 남은 safety margin을 추정해 grasp force를 조절할 수 있는가? 82개 marker의 3D 변형에서 CNN을 학습했다. | 실물 gripper, 제어 실증은 딸기·귤·바나나 세 종류. 실험 grasp force 범위 0.25–3.2 N | 같은 접촉에서 safety margin을 연속 추정해 낮은 grasp force를 유지했다. 작은 contact와 미끄러운 표면에서는 정확도가 제한됐다. | Direct prediction 자체의 선행 사례다. Dense observation과 지속적 feedback을 쓰므로 one-shot scalar에서 미래 충돌의 Brier 개선을 보장하지 않는다. [원문](https://pure.tudelft.nl/ws/portalfiles/portal/220701333/Learning_to_estimate_incipient_slip_with_tactile_sensing_to_gently_grasp_objects.pdf) |
| Li et al., 2025 공개 원고 | 국소 strain rate로 incipient slip의 위치와 시점을 알 수 있는가? Finite-element simulation, Tac3D bench, robot grasp에서 strain-rate extrema를 평가했다. | Tac3D 20×20 지점, 30 fps. Bench에서 다섯 tangential speed에 각 10회 반복 | Tangential load가 역전될 때 stick 영역이 일시적으로 회복되고 다시 바뀌는 과정까지 포착했다. | 외부에서 shear 방향을 역전한 조건이다. 같은 방향의 preload에서 full load로 바뀔 때 물체의 변형 부호가 저절로 역전되는 증거로 사용할 수 없다. [원문](https://arxiv.org/pdf/2502.17335) |
| Acosta et al., RA-L 2022 | Contact solver가 실제 impact를 얼마나 재현하는가? Drake·MuJoCo·Bullet의 contact parameter를 실물 trajectory에 맞추고 비교했다. | acrylic cube toss 550개 trajectory, Cassie jump 22개 | 거의 inelastic한 cube impact와 Cassie의 큰 동작을 재현했지만 elastic impact에는 한계가 있었다. Cassie 기록 중 slip regime이 없어 friction을 잘 식별하기 어려웠다. | 독립 solver의 일치와 실제 stick/slip 검증의 차이를 보여준다. 서로 다른 하중에서 같은 변형 방향을 추적한 자료는 없다. [원문](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf) |

### 연구 흐름과 근거의 강도

**공통되는 관찰.** Contact patch 안에서는 위치와 하중에 따라 slip이 동시에 일어나지 않는다. Skin 영상과 Tac3D에서 각기 다른 방식으로 관찰됐다. 수집한 위치나 초기 상태에 따라 반복 데이터의 분포가 달라질 수 있다는 점도 planar pushing 실험과 그 재분석에서 확인된다. 두 연구가 같은 pushing dataset을 사용했다는 점은 독립 재현 횟수를 셀 때 중요하다. [du Bois de Dunilac et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC10113816/) · [Li et al.](https://arxiv.org/pdf/2502.17335) · [Yu et al.](https://arxiv.org/pdf/1604.04038) · [Ma & Rodriguez](https://arxiv.org/pdf/1802.10089)

**조건부 근거.** Tactile deformation에서 편리한 scalar를 학습해 제어하는 방법과, deformation field에 mechanics를 적용해 stick 영역을 추적하는 방법은 모두 실제 접촉에서 성과가 있었다. 전자는 grasp safety margin, 후자는 현재 local slip field를 측정한다. 어느 접근이 향후 독립 충돌에 더 잘 전달되는지는 이 결과들로 판정할 수 없다. v49의 projection gate는 이 논문들이 실제로 시험한 baseline과 동일한 알고리즘이 아니다. [Willemet et al.](https://pure.tudelft.nl/ws/portalfiles/portal/220701333/Learning_to_estimate_incipient_slip_with_tactile_sensing_to_gently_grasp_objects.pdf) · [Li et al.](https://arxiv.org/pdf/2502.17335)

**엇갈리는 해석.** 같은 동작의 결과가 여러 무리로 갈라지면 재현 불가능한 noise처럼 보일 수 있다. Yu et al.은 변동 자체를 기록했고 Ma & Rodriguez는 같은 데이터에서 초기 orientation과 reset 절차를 따로 조사했다. 이를 독립된 반대 결과로 세는 것보다 두 번째 분석이 첫 번째 관측의 대안 설명을 제시한 것으로 읽는 편이 타당하다. V49에서 두 station의 차이를 해석할 때도 station 순서, 접근 경로, reset에 의한 residual strain이 위치와 함께 바뀌는지 살펴야 한다. [Yu et al.](https://arxiv.org/pdf/1604.04038) · [Ma & Rodriguez](https://arxiv.org/pdf/1802.10089)

**지식 지도와 미해결 관계.** Spatial heterogeneity와 reset history는 측정할 때 마주치는 접촉 상태를 바꾼다. Normal load와 local geometry는 strain 및 partial slip을 바꾸고, sensor와 estimator는 이 분포를 일부 요약한다. V49에 필요한 마지막 연결은 *약한 접촉에서 얻은 signed scalar → reset → full-load의 deformation → stick 확률*이다. 이 여섯 편에는 이 연결을 두 station에서 같은 비용으로 비교한 실물 측정이 없다. Solver를 바꿔 얻는 결과는 contact formulation에 대한 민감도를 평가한다. 실제 material에서의 방향 보존율은 별도로 추정해야 한다. [Ma & Rodriguez](https://arxiv.org/pdf/1802.10089) · [du Bois de Dunilac et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC10113816/) · [Willemet et al.](https://pure.tudelft.nl/ws/portalfiles/portal/220701333/Learning_to_estimate_incipient_slip_with_tactile_sensing_to_gently_grasp_objects.pdf) · [Acosta et al.](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf)

### 결과를 읽는 데 필요한 연구 검토

- *약한 접촉에서 이미 slip을 검출했다*고 쓰려면 preload 동안의 slip ground truth가 필요하다. 이번 질문의 endpoint는 이후 full-load 결과다. 실물에서 이 둘을 구별하려면 optical marker와 contact force를 각각 기록하고, reset 이후의 full-load label은 별도 판독해야 한다. [Li et al.](https://arxiv.org/pdf/2502.17335) · [du Bois de Dunilac et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC10113816/)
- *Query-local이 물리적으로 항상 우월하다*는 주장에는 근거가 부족하다. Frozen direct calibration이 손해를 봐도 관측 자체의 정보가 사라졌다고 단정할 수 없다. 방향 관계가 일관되게 바뀌면 측정은 여전히 informative이고 predictor가 잘못 해석했을 수 있다. Source-only와 별도로 표기한 diagnostic refit을 비교하면 두 가능성을 좁힐 수 있다. 이 해석은 distribution shift에 대한 가설이다. 실물 접촉에서 위치·방향·수집 절차의 효과가 함께 나타난 사례가 있다. [Ma & Rodriguez](https://arxiv.org/pdf/1802.10089)
- *독립 solver가 실물 결과를 검증했다*는 주장은 현재 근거를 넘는다. 실물에서 필요한 주요 비교는 같은 물체를 같은 예산으로 local와 fixed에 무작위 배정한 뒤, 반복 순서·마모·reset 시간까지 남기는 것이다. 특히 stick과 slip이 모두 생기는 물체·속도 범위에서 접촉을 관찰해야 한다. [Acosta et al.](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf) · [Yu et al.](https://arxiv.org/pdf/1604.04038)

### 다음 연구 질문 열 가지

아래는 문헌이 실증한 결론이 아니라, 각 결과와 현재 endpoint 사이의 간격을 좁히는 후속 실험 제안이다.

| 질문 | 필요한 이유 | 변수와 검증 방법 |
|---|---|---|
| 1. 실제 물체에서 weak-to-full 부호가 몇 번 유지되는가? | 기존 strain 측정은 주로 같은 contact 중의 변화다. | 물체·station별 weak/full signed displacement를 독립 영상으로 읽고 sign agreement와 near-zero 비율을 함께 보고한다. [du Bois de Dunilac et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC10113816/) |
| 2. 국소성은 어느 거리에서 사라지는가? | 두 위치만으로 실제 spatial correlation length를 알 수 없다. | Station 간 거리와 방향을 교차하고 물체별 semivariogram과 out-of-object Brier를 추정한다. [Yu et al.](https://arxiv.org/pdf/1604.04038) |
| 3. Reset의 잔류 효과가 위치 이득처럼 보이는가? | 자동화된 반복 절차가 다음 trial의 초기조건을 바꿀 수 있다. | Local-first/fixed-first와 대기시간을 무작위화하고 baseline displacement 및 Brier의 순서 효과를 분석한다. [Ma & Rodriguez](https://arxiv.org/pdf/1802.10089) |
| 4. 0.4 N에서 물체 변형과 sensor 자체 변형을 분리할 수 있는가? | Tactile marker는 sensor skin의 변형도 읽는다. | 강체 fixture와 물체 표면의 독립 marker를 함께 기록하고, 두 signal로부터 방향별 bias와 noise floor를 구한다. [Willemet et al.](https://pure.tudelft.nl/ws/portalfiles/portal/220701333/Learning_to_estimate_incipient_slip_with_tactile_sensing_to_gently_grasp_objects.pdf) |
| 5. 부호 반전은 어느 load 범위에 집중되는가? | 일정한 방향의 load 증가와 외부 shear reversal은 다른 조작이다. | Preload–full-load ramp를 여러 force에 걸쳐 촬영하고 loading/unloading, sign change와 slip onset을 따로 채점한다. [Li et al.](https://arxiv.org/pdf/2502.17335) |
| 6. 같은 touch의 짧은 동영상이 scalar의 실패를 복원하는가? | 접촉 중 strain rate에는 단일 시점이 놓칠 수 있는 정보가 있다. | 총 dwell time을 고정하고 scalar·frame pair·strain-rate sequence를 비교해 Brier와 GPU p90을 함께 잰다. [Li et al.](https://arxiv.org/pdf/2502.17335) |
| 7. Source에서 고정한 decoder가 부호만 잘못 해석하는가? | 강한 변형 신호와 calibration transfer 실패는 동시에 일어날 수 있다. | Frozen direct, source mixture, 소량의 별도 labeled calibration을 허용한 diagnostic을 구분해 probability calibration을 비교한다. [Willemet et al.](https://pure.tudelft.nl/ws/portalfiles/portal/220701333/Learning_to_estimate_incipient_slip_with_tactile_sensing_to_gently_grasp_objects.pdf) |
| 8. 동일 초기조건에서 solver끼리 label이 다른 접촉이 순위를 좌우하는가? | 실제 impact의 재현 정도가 regime에 따라 달랐다. | 같은 incoming state를 여러 solver에 넣고 contact duration·post-impact tangent speed·label disagreement와 policy 차이를 비교한다. [Acosta et al.](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf) |
| 9. 실물에서 몇 초까지 preload를 정당화할 수 있는가? | 실시간 제어에서의 sensor rate는 reset 후 한 번 더 누르는 비용을 알려주지 않는다. | 물체별 approach·readout·retract·reset failure를 timing하고 Brier와 시간의 Pareto frontier를 계산한다. [Willemet et al.](https://pure.tudelft.nl/ws/portalfiles/portal/220701333/Learning_to_estimate_incipient_slip_with_tactile_sensing_to_gently_grasp_objects.pdf) |
| 10. 물체가 아니라 session을 바꿔도 경계가 재현되는가? | Surface, humidity, 시작 상태의 변화로 조건별 결과가 달라질 수 있다. | Object family·surface·day 단위 holdout에 대해 예측한 부호와 Brier 순위를 사전 등록하고 session 단위로 재표집한다. [Ma & Rodriguez](https://arxiv.org/pdf/1802.10089) · [Willemet et al.](https://pure.tudelft.nl/ws/portalfiles/portal/220701333/Learning_to_estimate_incipient_slip_with_tactile_sensing_to_gently_grasp_objects.pdf) |

## 재현 자료

서버 프로젝트는 `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/contact_transfer_v49/`다. 사전 계획은 2026-11-16 12:04:46 UTC에 저장했고 solver는 12:08:16 UTC에 동결했다. 멀티 에이전트로 독립 solver·수치 검산, frozen policy·GPU timing, 원논문 검토를 나눴고 paired scoring은 별도 구현했다. Target label을 읽지 않고 frozen forecast를 생성했다. 여섯 기존 policy 경로 각각에서 예측 2,359,296개를 재생해 최대 오차 5.55×10⁻¹⁶을 확인했다. Raw forecast는 서버의 `forecast49/`, primary/fine/ultra outcome은 `labels49/`, `labels49_fine_all/`, `labels49_ultra_all/`, method 순서와 시간은 `timing49/`에 있다. [Plan checksum](plan49.sha256) · [Solver checksum](solver49.sha256) · [Timing 원자료](timing49) · [열린 연구 질문과 원문 목록](LITERATURE_KO.md)

저장된 bundle에서는 Slurm compute node에서 프로젝트 하위 `contact_transfer_v49`로 이동한 뒤 `../continuous_v2/.venv/bin/python score49.py`로 primary 수치를 재현할 수 있다. 이후 `transfer_controls49.py`, `threshold49.py`, `refinement49.py fine`, `refinement49.py ultra`를 같은 Python으로 실행하면 보조 비교가 재현된다. 물리 데이터 생성과 GPU timing 상세 설정은 각 protocol과 실행 코드에 남겼다.
