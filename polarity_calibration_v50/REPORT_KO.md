# v50 · 새 물체와 표면에서 네 쌍의 측정으로 preload 보정 방향을 정할 수 있는가?

**같은 표면 환경에서 먼저 측정한 네 쌍의 weak/full displacement로 이후 다른 물체에 적용할 보정 부호를 바꾸면 예측이 개선됐다.** 새로운 네 contour family와 두 surface에서, 15개 조건을 균등하게 평균한 held-out Brier loss는 query-local hard sign이 **0.158165**였다. Frozen local은 **0.169653**, 똑같은 calibration으로 부호를 정한 fixed probe는 **0.159373**, 부호를 버리고 source에서 학습한 unsigned policy는 **0.159297**이었다. 사전 지정한 다섯 비교 모두 12개 catalog에서 같은 방향이었다.

**부호를 단정할 필요는 없었다.** 한 방향을 고르는 대신 추정한 유지율로 두 예측을 섞은 query-local soft policy의 Brier는 **0.155886**이었다. 단순한 sign vote를 써도 soft policy가 **0.155941**이어서, 이득의 상당 부분은 복잡한 estimator 없이도 얻을 수 있었다. 네 쌍으로 극단적인 환경의 **방향**은 잘 판별했지만, **유지율 값 자체**는 0.5 쪽으로 크게 수축했다.

## 1. 배포 시 실제로 무엇을 관측하는가

1. 새 surface session에서 calibration용 물체 K개를 뽑는다. 각 물체의 같은 위치를 **0.4 N**으로 눌러 두 fiducial의 signed displacement를 읽고 reset한다. 이어 **2.0 N reference load**에서 한 번 더 측정하고 reset한다. **K=4는 물체 네 개, 접촉 여덟 번, reset 여덟 번**이다. 이때 stick/slip label은 제공하지 않는다.
2. Active 변형의 weak-to-full 부호 유지확률을 κ라고 한다. Estimator는 inert·negative·positive의 가능성과 관측 오차를 적분한다. Signed displacement가 weak에서 0.018 mm, reference에서 0.120 mm보다 작은 경우 그 쌍은 방향의 증거로 쓰지 않는다. Uniform prior와 source 설계에서 고정한 noise model로 posterior mean κ̂를 계산한다. [Estimator](estimator50.py) · [측정·추정 검산](ESTIMATOR_KO.md)
3. 그 session의 **다른 물체**에서는 query 위치에 weak preload를 한 번만 가한다. Frozen direct가 측정값 d에서 내는 stick 확률을 p(d)라고 할 때, hard policy는 κ̂<0.5이면 p(−d), 그 외에는 p(d)를 사용한다. Soft policy는 κ̂p(d)+(1−κ̂)p(−d)를 사용한다. 두 경우 모두 지정된 부호에서 **측정값을 바꾸어 원래 보정식을 평가**한다. Stick 확률 자체를 1−p로 바꾸지 않는다. Fixed probe는 항상 31 mm에서 똑같은 예산으로 측정한다. [배포 정책](policy50.py)
4. Calibration object와 deployment object는 서로 다른 support와 seed로 뽑았다. 두 집합은 같은 새 family·surface session을 공유한다. Calibration에도 등장하지 않은 surface로 옮겨 성공한다는 주장은 이 split만으로 검증할 수 없다. 각 6,144개 physical parameter vector를 대조해 교집합이 0임을 확인했다. 네 새 family는 saddleback·terrace·swale·scallop이고, 두 새 surface는 높이에 따른 friction 변화가 서로 다른 mosaic과 nap이다. 각 catalog의 네 환경에 두 surface와 두 rate law를 균형 있게 넣었다. Calibration 쪽은 별도 specimen ID에 family별 deforming geometry, specimen gain·bias와 독립 측정 noise를 부여해 두 displacement를 생성했다. Calibration object의 PyBullet stick label을 학습에 쓰는 절차는 없다. [World 검산](PHYSICS_KO.md) · [Calibration 생성](calibration50.py)

ρ는 12 mm 떨어진 위치가 query 위치의 weak state를 복사할 확률이다. 복사하지 않으면 동일 prior에서 독립 추출한다. ρ={0,0.5,1}와 κ={0,0.25,0.5,0.75,1}을 교차했고, inert는 항상 inert로 유지했다. κ는 calibration과 deployment가 같은 session에서 공유하는 생성 조건이다. 정책이 받는 입력에는 ρ·κ·surface 이름·target outcome이 없다. Primary 추정은 각 catalog·환경·κ에서 서로 다른 calibration prefix 256개를 생성해 평균했다. 이는 같은 calibration population에서의 반복이며 독립적인 새 재질 256개를 뜻하지 않는다. [사전 계획](PLAN_KO.md)

## 2. 네 쌍으로 유지율과 보정 방향을 얼마나 잘 추정했나

| 실제 κ | 평균 κ̂ | κ 추정 RMSE | 네 쌍에서 보정 부호를 뒤집은 비율 |
|---:|---:|---:|---:|
| 0 | 0.272 | 0.284 | 93.77% |
| 0.25 | 0.385 | 0.201 | 73.78% |
| 0.50 | 0.499 | 0.166 | 47.66% |
| 0.75 | 0.612 | 0.204 | 21.61% |
| 1 | 0.728 | 0.284 | 0.41% |

네 쌍 중 평균 **2.035쌍**이 양쪽 readout threshold를 넘었다. κ=0에서 방향 판별 성공률은 93.77%, κ=1에서는 99.59%였다. 방향 증거가 동률이면 frozen 방향을 유지하기 때문에 두 성공률이 비대칭이다. κ=0.5에는 정답 부호가 없다. 여덟 쌍에서는 양 극단의 올바른 방향 선택률이 각각 99.61%, 99.91%로 상승했다. 같은 weak observation은 κ 전체에서 bitwise 동일했다. Full reading을 다른 물체의 weak reading과 짝지으면 κ=0과 1에서 평균 κ̂가 각각 0.498과 0.502가 됐다. 이 설계에서 정보를 주는 것은 **같은 물체·같은 위치에서 얻은 두 하중의 대응**이다. [추정 오차·반전율](calibration50.json)

## 3. Held-out Brier loss와 probe budget

각 행은 같은 K와 deployment preload 1회를 허용한다. Fixed frozen·query frozen·unsigned는 calibration prefix를 제공받아도 사용하지 않으므로 loss가 K와 무관하다. 표는 사용 가능한 budget에 대한 비교이며 실제로 calibration을 생략할 때의 비용 절감은 §6에 계산했다.

| 환경당 calibration 쌍 K | Query hard | Frozen local | Frozen fixed | 같은 K를 쓴 fixed hard | Unsigned source fit | Query soft | 같은 K를 쓴 fixed soft |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.169653 | 0.169653 | 0.165598 | 0.165598 | 0.159297 | 0.159636 | 0.159938 |
| 1 | 0.163643 | 0.169653 | 0.165598 | 0.162339 | 0.159297 | 0.158198 | 0.159173 |
| **4** | **0.158165** | 0.169653 | 0.165598 | 0.159373 | 0.159297 | **0.155886** | 0.157948 |
| 8 | 0.156772 | 0.169653 | 0.165598 | 0.158618 | 0.159297 | 0.154618 | 0.157284 |

**Hard policy는 시험한 budget 중 네 쌍부터** 균등 평균에서 다섯 primary comparator를 모두 앞섰다. 2·3쌍은 시험하지 않아 최솟값은 모른다. K=4에서 아래 차이는 *query hard에서 비교 정책을 뺀 값*이다. 통계 단위는 독립 catalog 12개이며, 각각 내부에서 모든 조건과 calibration 반복을 먼저 평균했다. 다섯 비교에 Bonferroni를 적용한 paired 95% simultaneous CI다.

| K=4 비교 정책 | Brier 차이 [simultaneous 95% CI] |
|---|---:|
| Frozen local | −0.011488 [−0.012053, −0.010923] |
| Frozen fixed | −0.007433 [−0.008305, −0.006561] |
| 같은 calibration을 쓴 fixed hard | −0.001209 [−0.001466, −0.000951] |
| Source-only unsigned fit | −0.001132 [−0.001499, −0.000765] |
| 단순 sign-symmetric 평균 | −0.001472 [−0.001830, −0.001114] |

Frozen local 대비 큰 차이에는 환경에 따라 부호를 바꾸는 효과가 포함된다. 같은 calibration으로 부호를 바꾸는 fixed와 비교한 **0.001209**가 측정 위치와 그 위치에 맞는 frozen decoder를 함께 바꾼 크기다. Query가 31 mm면 두 정책의 예측이 모든 noise에서 bitwise 같았다. K=4에서 query soft − query hard는 **−0.002279 [−0.002649, −0.001910]**, query soft − fixed soft는 **−0.002063 [−0.002272, −0.001853]**이었다(28개 secondary 비교를 보정). 참 κ를 알려주고 동일한 식으로 혼합하는 privileged reference의 평균은 0.152629였다. [전체 15개 cell과 통계](score50.json) · [Catalog별 결과](catalog_scores50.npz)

![관측 수에 따른 유지율 추정, 정책별 Brier loss 및 calibration과 deployment가 달라질 때의 손익](polarity_calibration_v50.png)

## 4. 보정 부호를 언제 바꿀 것인가

아래는 K=4에서 세 공간 조건을 동일 가중한 결과다.

| 실제 κ | Query hard | Frozen local | Adaptive fixed hard | Unsigned fit | Query soft |
|---:|---:|---:|---:|---:|---:|
| 0 | **0.148761** | 0.193821 | 0.154280 | 0.159417 | 0.151172 |
| 0.25 | 0.164000 | 0.181737 | 0.162529 | 0.159357 | **0.158360** |
| 0.50 | 0.169640 | 0.169653 | 0.165581 | **0.159297** | 0.160738 |
| 0.75 | 0.162741 | **0.157569** | 0.161851 | 0.159237 | 0.158275 |
| 1 | 0.145681 | **0.145485** | 0.152626 | 0.159177 | 0.150884 |

**부호가 일관되게 뒤집히는 환경**에서는 hard flip의 이득이 컸다. **부호 관계가 약한 환경**에서는 한쪽을 확정하는 비용이 컸다. κ=0.5에서는 unsigned fit이 더 좋았고, κ=1에서는 이미 올바른 부호를 쓰는 frozen local이 앞섰다. 조건별 60개 simultaneous contrast에서 hard가 네 주요 대조를 모두 이긴 cell은 κ=0이고 ρ=0 또는 0.5인 두 개였다. 15개 조건을 균등하게 평균한 결론을 모든 재질에 적용할 수 없다. [조건별 paired 결과](score50.json)

**배포 환경 빈도에도 민감하다.** κ=0·1 내부는 균등, κ=0.25·0.5·0.75 내부도 균등하게 두고 극단 환경의 비중만 바꿨다. Hard가 unsigned fit을 앞서기 위한 극단 환경 비중의 점추정은 **33.8%**(catalog bootstrap 95% CI 32.6–35.0%)였다. 이는 지정한 두 분포 사이의 민감도 분석이다. 또 κ=0에서 calibration하고 κ=1인 session에 그대로 옮기면 hard − frozen local은 **+0.045037**이었다. 실제 배포에서 calibration 결과를 언제 폐기할지가 중요한 이유다. [환경 빈도 민감도](mix50.json) · [Session 이동 결과](score50.json)

## 5. 다른 무엇이 결과를 설명하는가

- **단순한 부호 집계.** 같은 threshold를 넘은 쌍에서 부호 일치·반전만 세고 Beta(1,1)로 보정했다. K=4에서 이 방법의 hard/soft는 각각 0.158306/0.155941이었다. Likelihood estimator의 추가 개선은 hard **0.000141**, soft **0.000055**였다. 반면 같은 환경에서 full reading을 다른 specimen에 붙이면 likelihood hard/soft가 **0.169594/0.160644**로 악화됐다. 측정 쌍을 올바르게 만드는 효과가 estimator 정교화의 효과보다 컸다. [Vote·shuffle 대조](controls50.json)
- **Solver 수치 민감도.** PyBullet primary에서 각 해상도당 663,552개 trajectory를 계산했다. Full state의 부호만 뒤집으면 36.72%에서 stick/slip label이 바뀌었다. Time step과 mesh spacing을 절반으로 줄이고 반복 횟수를 두 배로 늘리자 전체 label의 3.365%가 바뀌었다. 같은 K=4에서 ultra 해상도의 hard는 0.163606, frozen local은 0.175178, adaptive fixed는 0.165026, unsigned fit은 0.165128, soft는 0.161446이었다. 다섯 primary 비교 모두 개선이 유지됐지만 절대 확률의 수치 수렴까지 입증하지는 않는다. [Ultra 결과](score50_ultra.json) · [물리 검산](world50_audit.json)
- **한 환경에서 실제로 한 번만 calibration하는 경우.** Primary는 주어진 생성 조건에서 기대되는 손실을 정밀하게 비교한다. 별도의 realized rollout에서는 48개 환경에 각각 K=4 prefix 하나만 배정하고, 6,144개 물체당 실제 binary outcome 한 개만 사용했다. Hard 0.153375, frozen local 0.167913, adaptive fixed 0.155433, unsigned fit 0.156065, soft 0.152439였다. 순위의 점추정은 유지됐지만 다섯 개의 동시 CI 모두 0을 포함했다. 예컨대 hard − unsigned fit은 −0.002690 [−0.009073, +0.003692]였다. 한 번의 배포로 작은 이득을 재현하려면 독립 session 수가 더 필요하다. [단일 배포 rollout](rollout50.json)

## 6. GPU latency와 물리 sensing·reset 비용

기존 작업이 돌던 두 allocation은 건드리지 않았다. 빈 RTX PRO 6000 Blackwell 두 장에서 forecast를 나눠 실행하고, GPU p90은 node56 한 장에서 비교했다. 물체 512개는 서로 다른 4개 환경의 128개 물체로 구성한다. 매번 estimator가 필요한 정책은 측정값 4쌍으로 환경별 posterior를 다시 계산한 뒤, factual acquisition·선택한 sensor 포장·original posterior 한 번·보정·GPU synchronization까지 실행했다. Simulated sensor 생성, Bullet label 생성, robot 동작 시간은 GPU latency에 포함하지 않았다.

| K=4 정책 | GPU p90, ms/512 [pointwise 95% CI] |
|---|---:|
| Query-local hard | 470.26 [461.06, 474.83] |
| Frozen local | 463.56 [458.20, 466.20] |
| Frozen fixed | 464.44 [460.31, 474.42] |
| Adaptive fixed hard | 461.68 [458.40, 464.26] |
| Unsigned source fit | 462.15 [458.13, 467.28] |
| Sign-symmetric 평균 | 462.76 [458.00, 468.05] |
| Query-local soft | 462.32 [458.47, 465.58] |

12개 catalog에서 정책당 warmup 1회 후 4회씩, 48개 batch를 무작위 순서로 쟀다. Paired catalog bootstrap은 16,000회다. Hard − frozen local은 +6.70 ms [−2.28, +15.85], hard − unsigned fit은 +8.11 ms [−0.98, +16.00], hard − adaptive fixed는 +8.58 ms [+0.85, +15.21]였다. 동일 estimator를 쓰는 두 hard 정책의 차이까지 나타났기 때문에 이를 calibration 계산 자체의 비용으로 귀속할 수 없다. 별도로 잰 **네 환경 전체의 host estimator p90**은 K=0·1·4·8에서 각각 0.0027·0.0436·0.0454·0.0481 ms였다. [Primary GPU timing](timing50.json) · [온라인 실행 경로](bench50.py)

이 차이의 재현성을 확인하려 같은 GPU와 12개 catalog에서 실행 순서와 직전 정책을 균형화한 Williams crossover를 반복했다. Hard local 469.92, frozen local 465.53, fixed hard 470.36, local soft 465.83 ms였다. **Hard local − fixed hard는 −0.44 ms [−14.81, +9.56]로 첫 측정의 차이가 재현되지 않았다.** Hard local − frozen local은 +4.39 ms [+0.37, +12.64], hard local − soft는 +4.09 ms [+0.07, +12.43]였다. 두 실행을 보존했으며, GPU 속도 개선이나 latency equivalence는 주장하지 않는다. [순서 균형 crossover](balance50.json)

**물리 동작은 횟수와 시간 가정을 따로 계산했다.** 모든 유료 정책은 배포 물체당 weak contact/reset 1회를 쓴다. K쌍을 같은 환경의 배포 물체 B개에 재사용하면 추가 calibration 비용은 물체당 **2K/B contact와 2K/B reset**이다. K=4, B=128이면 deployment까지 합쳐 물체당 평균 **1.0625 contact와 1.0625 reset**이다. Calibration을 생략할 수 있는 frozen·unsigned 정책은 물체당 각각 1회로 운용할 수 있다. Uniform grid의 Brier 수치는 생략 전후가 같다.

다음은 **실측하지 않은 물리 시간 시나리오**다. 각 접촉에 접근·측정·후퇴·안정화 100+50+100+40 ms를 공통 적용하고, loading 4 N/s 및 unloading 8 N/s로 가정했다. 0.4 N weak 접촉은 440 ms, 2.0 N reference 접촉은 1,040 ms, calibration 한 쌍은 1.48 s다. K=4는 환경당 5.92 s, 128개 물체에 재사용하면 **46.25 ms/object**다. 공통 deployment 접촉까지 합치면 486.25 ms/object이고, 생략 가능한 baseline은 440 ms/object다. 물체 교체·횡이동·reset 실패·재시도는 별도다. Query에서 출발하고 복귀하는 로봇이 60 mm/s로 이동한다고 가정하면 19 mm query에서 31 mm fixed station까지의 왕복은 400 ms를 더한다. 두 높이 동일 가중 평균에서는 fixed에 200 ms/object가 추가된다.

실제 장비에서 calibration을 생략하는 unsigned fit과 비교하면 K=4의 hard는 0.001132, soft는 0.003411 Brier를 얻었다. 위 시나리오에서 추가 1초의 시간 비용이 각각 **0.0245 또는 0.0738 Brier-equivalent**보다 낮아야 평균 손익이 양수다. 한 session에서 물체를 몇 개 다루는지가 도입 판단에 직접 영향을 준다.

## 7. 논문 기여와 다음 검증

1. **근거가 생긴 주장.** 새 object와 surface에서 같은 물체의 하중 두 개를 짝지어 측정하면, 별도 배포 물체에 적용할 보정 방향을 정할 수 있다. 같은 비용의 adaptive fixed, 독립적으로 학습한 unsigned fit, 다른 물체끼리 측정 쌍을 섞은 control, 다른 solver 해상도를 한 endpoint로 비교했다. 부호를 결정하는 것과 κ 자체를 정확히 복원하는 것의 난이도도 분리했다.
2. **주장을 좁혀야 하는 부분.** Soft mixture는 일반적인 posterior averaging이며 단순 sign count로도 비슷한 loss를 얻었다. Estimator 자체의 독창성이나 GPU 속도 우위를 내세울 근거는 약하다. 중간 κ에서는 unsigned가 이길 수 있고, session이 달라지면 틀린 부호를 높은 확신으로 옮기는 위험이 있다. 이 연구의 물리적 핵심 가정은 **독립된 2.0 N reference에서 본 방향과 실제 충돌 중 변형 방향의 대응**이다. 센서와 solver 양쪽이 지정한 deformation family를 공유하므로 이 관계는 독립적으로 검증되지 않았다.
3. **다음 판별 실험.** 실제 물체에서 0.4 N·2.0 N의 변위를 독립 optical marker로 측정하고, 별도의 high-speed camera로 query impact 중 실제 변형 부호와 stick/slip을 동시에 기록한다. 물체·station·날짜를 바꿔 reference-to-impact 불일치율, 마모와 reset 후 잔류 변형을 측정한다. Primary 비교는 source-only unsigned와 네 쌍을 쓴 query soft로 사전 고정하고, 단순 vote soft를 함께 둔다. 평가 단위는 session으로 잡는다. 한 session의 128개 물체를 독립 환경 128개처럼 세지 않는다.

## 선행연구와 v50의 기여 범위

검토 질문은 *새 표면에서 소수의 paired load 측정으로 이후 다른 물체의 예측 방향을 결정할 수 있는가*이다. 아래 여섯 원논문에서 측정 대상, 실험 단위, 방법과 한계를 비교했다.

| 연구 | 질문·방법 | 표본 또는 데이터 | 결과·결론 | v50에서 남는 간격 |
|---|---|---|---|---|
| Su et al., 2012 | Robot이 normal force를 제어하며 변위를 측정하면 compliance를 구분할 수 있는가? | 실물 aluminum plate와 서로 다른 rubber 4종, 목표 10 N | 하중과 tactile signal의 관계로 material을 구분했다. 작은 힘에서 sensor skin·joint cable 변형이 섞였다. | 0.4 N에서 관측한 변위가 물체 자체의 변형임을 별도 계측해야 한다. [원문](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2012.00007/full) |
| Fishel & Loeb, 2012 | Exploratory movement를 어떻게 고르면 접촉 횟수를 줄일 수 있는가? Force·velocity를 조절해 posterior를 갱신했다. | 117개 등록 texture와 세 종류의 sliding movement | 등록된 texture 식별에서 median 5회, 정확도 95.4%를 보고했다. | 이미 등록한 texture의 identity와 새 environment의 연속적인 κ는 서로 다른 endpoint다. [원문](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2012.00004/full) |
| Ma & Rodriguez, 2018 | 어떤 요인이 반복 push 결과의 여러 군집을 설명하는가? 기존 실물 자료를 재분석하고 anisotropic friction simulation과 비교했다. | 2,000회 push sequence, 네 가지 support material을 논의 | 초기 자세와 수집과정의 편향 및 anisotropic friction이 상당 부분을 설명했다. | Reset 뒤 상태가 달라지면 동일한 κ를 공유한다는 가정도 흔들린다. 실물 연구의 반복 수는 독립 session 수와 다르다. [원문](https://arxiv.org/pdf/1802.10089) |
| Boonstra·Willemet et al., ICRA 2024 | Dense tactile marker로 같은 접촉의 slip safety margin을 학습해 힘을 제어할 수 있는가? | 82개 marker, 딸기·귤·바나나 grasp 평가, 조건당 5회 | Safety-margin target을 높이면 대체로 grasp force가 커지고 무거운 과일의 slip이 줄었다. | 현재 접촉의 feedback을 사용했다. 다른 물체의 reset 이후 충돌은 검증 범위 밖이다. [원문](https://pure.tudelft.nl/ws/portalfiles/portal/220701333/Learning_to_estimate_incipient_slip_with_tactile_sensing_to_gently_grasp_objects.pdf) |
| Wong & Zhu, 2026 | 물체별 calibration 없이 현재 slip에 대응할 수 있는가? 손가락별 tri-axial force 변화로 보정했다. | 실물 aluminum rod·sponge·water bottle, rod의 세 lifting speed | 세 물체에서 slip recovery를 보였다. 가장 높은 속도에서는 control setpoint를 조절했다. | 현재 slip 이후 반응할 수 있을 때 쓸 수 있는 강한 대안이다. v50의 미래 충돌 전 확률 예측과 endpoint가 다르다. [원문](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2026.1735467/full) |
| Hui & Walter, 1980 | 불완전한 두 측정의 오류와 잠재 상태 비율을 함께 식별할 수 있는가? Latent-class likelihood 분석. | Prevalence가 다른 두 모집단의 paired Mantoux/Tine test 자료 | 두 모집단, 공통 error rate 및 조건부 독립과 같은 조건 아래 식별할 수 있음을 보였다. | 통계적 유추다. 로봇 연구가 아니며, 실제 시스템에서 두 sensor error 또는 그 독립성이 보장된다는 근거는 없다. [원문](https://jhanley.biostat.mcgill.ca/bios601/CandH-ch0102/HuiWalter1980.pdf) |

**연구 지도와 근거의 강도.** 재료와 계측기의 compliance → load별 displacement → 짝지은 관측으로 환경의 관계 추정 → 별도 물체의 action이라는 흐름에서, 실물에서 상대적으로 근거가 강한 부분은 탐색 동작·접촉 하중이 읽을 수 있는 물리 정보를 바꾼다는 점이다. 약한 구간에서는 sensor 자체가 신호를 차지할 수 있다. 몇 번의 접촉으로 식별이 가능하다는 근거는 이미 관측한 texture를 재식별하는 상황에 강하다. 다른 물체·session의 weak-to-full polarity transfer까지 보장하는 근거는 이 문헌군에서 부족하다. [Su et al.](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2012.00007/full) · [Fishel & Loeb](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2012.00004/full) · [Ma & Rodriguez](https://arxiv.org/pdf/1802.10089)

**엇갈리는 결과를 읽는 법.** 촉각으로 slip을 배울 수 있다는 결과와 물체별 calibration 없이 slip을 제어할 수 있다는 결과는 현재 grasp에 feedback을 허용한 실험에서 나왔다. 전자는 safety margin을 연속 추정하고 후자는 slip 이후 손가락 단위로 대응한다. V50은 독립 물체에 사고가 일어나기 전에 stick 확률을 내야 한다. 이 시간 차이를 맞추기 전에는 어느 정책이 더 보편적으로 우수하다고 할 수 없다. [Boonstra·Willemet et al.](https://pure.tudelft.nl/ws/portalfiles/portal/220701333/Learning_to_estimate_incipient_slip_with_tactile_sensing_to_gently_grasp_objects.pdf) · [Wong & Zhu](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2026.1735467/full)

**현재 연구에 대한 반론.** Paired reading은 상태 자체가 아니라 오차가 있는 두 계측값이다. 둘의 부호가 같이 틀릴 때 높은 일치율을 얻을 수 있다. Reference channel의 polarity, 약한 힘에서의 분해능, reset 순서를 각각 검산해야 한다. 센서 bias와 κ까지 같은 소수 관측에서 자유롭게 추정한 것처럼 쓰면 식별성 가정을 숨기게 된다. 또한 비교 정책이 calibration을 사지 않아도 되는 현장에서는 추가 contact의 총비용을 청구해야 한다. [Hui & Walter](https://jhanley.biostat.mcgill.ca/bios601/CandH-ch0102/HuiWalter1980.pdf) · [Su et al.](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2012.00007/full) · [Wong & Zhu](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2026.1735467/full)

### 후속 연구 질문 10개

| 질문 | 간격과 필요성 | 변수·시험 방법 |
|---|---|---|
| 1. 0.4 N에서 실제 물체 부호를 읽는가? | Apparatus deformation이 물체 변형과 섞일 수 있다. | 강체 reference와 독립 카메라로 force–displacement curve와 error floor를 측정한다. [Su et al.](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2012.00007/full) |
| 2. 2 N reference 부호가 실제 impact 부호인가? | 정적 compliance와 다른 충돌의 transfer는 검증할 간격이다. | 같은 station에 sacrificial specimen을 배정해 reference gauge와 high-speed impact 영상을 비교한다. [Su et al.](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2012.00007/full) |
| 3. 두 계측기의 오류 상관이 κ를 과대추정하는가? | Conditional independence는 스스로 검증되지 않는다. | Fiducial을 독립 교정하고 공통 optical bias를 주입해 추정 변화를 잰다. [Hui & Walter](https://jhanley.biostat.mcgill.ca/bios601/CandH-ch0102/HuiWalter1980.pdf) |
| 4. κ가 날마다 유지되는가? | 반복 실험의 수집 과정이 초기 조건을 바꿀 수 있다. | 동일 surface를 날짜·마모 session으로 나누고 cross-session Brier를 잰다. [Ma & Rodriguez](https://arxiv.org/pdf/1802.10089) |
| 5. Probe 순서와 reset 시간이 방향을 바꾸는가? | 자동 reset은 물체를 특정 자세로 모을 수 있다. | Weak-first와 reference-first, 짧고 긴 reset을 별도 specimen에 무작위 배정한다. [Ma & Rodriguez](https://arxiv.org/pdf/1802.10089) |
| 6. Inert trial이 많을 때 probe 위치나 힘을 고르면 이득인가? | 등록 texture에서는 탐색 선택에 따라 필요한 움직임 수가 달랐다. | Fixed vs adaptive 위치·하중을 같은 contact budget에서 비교하고 손상률을 기록한다. [Fishel & Loeb](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2012.00004/full) |
| 7. 같은 budget에서 여러 물체와 한 물체 반복 중 어느 쪽이 낫나? | 한 표본에 반복된 관측과 환경 내 이질성은 구분해야 한다. | 4개 × 1쌍 대 1개 × 4쌍의 out-of-object 오차를 비교한다. [Ma & Rodriguez](https://arxiv.org/pdf/1802.10089) |
| 8. 짧은 tactile trajectory가 scalar의 부호 모호성을 해소하는가? | 현재 접촉에서 dense deformation으로 관측 가능한 정보가 있다. | 동일 dwell time에서 단일 scalar·marker trajectory의 Brier와 latency를 비교한다. [Boonstra·Willemet et al.](https://pure.tudelft.nl/ws/portalfiles/portal/220701333/Learning_to_estimate_incipient_slip_with_tactile_sensing_to_gently_grasp_objects.pdf) |
| 9. 사전 forecast와 reactive feedback 중 언제 어느 쪽의 비용이 낮은가? | Slip 후 복구가 가능한 과업에서는 사전 probe 예산의 가치가 달라진다. | 낙하 비용·복구 허용시간을 바꾼 실물 과업에서 총 contact·손실을 센다. [Wong & Zhu](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2026.1735467/full) |
| 10. 새로운 sensor나 다른 손가락에도 readout sign을 옮길 수 있는가? | 하나의 platform에서 조정한 threshold와 계측 축이 장비 간에 달라질 수 있다. | Device holdout에서 zero check·축 교정·reference 횟수를 분리한다. [Wong & Zhu](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2026.1735467/full) · [Hui & Walter](https://jhanley.biostat.mcgill.ca/bios601/CandH-ch0102/HuiWalter1980.pdf) |

## 재현 자료

서버 경로는 `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/polarity_calibration_v50/`다. 비교 계획은 2026-11-16 14:01:58 UTC에 저장했다. 멀티 에이전트로 world·solver와 split audit, blind forecast·GPU timing, calibration·문헌 검토를 나눴고 별도 코드로 Brier를 채점했다. 물리·센서·정책·채점 코드는 checksum과 함께 보존했다. 5개 policy 경로 각각에서 예측 2,359,296개를 독립 재계산하여 최대 확률 오차 5.55×10⁻¹⁶을 확인했다. [실행 계획](PLAN_KO.md) · [World·solver 검산](PHYSICS_KO.md) · [Policy replay](audit_policy50.json) · [Estimator 검산](ESTIMATOR_KO.md) · [문헌·후속 연구 질문](LITERATURE_KO.md)

실제 binary label과 관측은 `labels50/`, `labels50_ultra/`, `cal50/`에, target label을 보지 않은 forecast는 `forecast50/`에 저장했다. Slurm compute node에서 이 폴더로 이동해 `../continuous_v2/.venv/bin/python score50.py`를 실행하면 primary 결과를 재현한다. `score50.py --labels labels50_ultra --tag 50_ultra`는 두 번째 해상도를 평가한다. 별도 코드로 source-fit 대 simple vote, paired 대 shuffled, session drift와 한 번의 배포 rollout을 검증했다. [채점 코드](score50.py) · [대안 설명 대조](controls50.py) · [실현 표본 대조](rollout50.py)
