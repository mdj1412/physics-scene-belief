# v58 · 같은 표면에서 두 물체가 모두 mismatch일 때, 물리가 지속된 것인가 센서가 같은 오류를 낸 것인가?

**4×2의 double mismatch는 두 원인 모두에서 생겼다.** 환경 A에서 첫째·둘째 물체를 같은 센서로 읽으면 sensor error만 지속된 조건이 14.00%, 물리 상태만 지속된 조건이 14.45%였다. 첫 물체는 한 센서로, 둘째 물체는 다른 센서로 읽으면 각각 **9.37%와 14.47%**로 나뉘었다. 이 구별은 두 센서의 오류 원인이 서로 독립인 simulation에서 성립했다. 하나의 새 표면에서 8회 impact만 구입하면 물리적 지속 유무의 판별률은 **52.22%**, 이후 어느 정책을 쓸지 맞히는 판별률은 **51.66%**에 그쳤다. [부호 판독 검산](PILOT_KO.md) · [단일 pilot 분석](mechanism58.json) · [정책 채점](SCORE_KO.md)

**정책 선택에서는 같은 4×2에 두 번째 센서를 더한 효과와, 같은 두 센서로 8×1을 4×2로 바꾼 효과가 갈렸다.** 전자는 평균 held-out Brier를 0.934×10⁻⁶ 낮췄고, 후자는 6.661×10⁻⁶ 높였다. Session loss p90의 차이는 두 비교 모두 불확실했다. 실험은 서버의 단일 idle GPU를 사용했다. 물체·센서·reset은 simulation이고, GPU latency는 실제 측정값이다. [정책 비교](score58.json) · [GPU·예산 검증](GPU_WORLD_KO.md)

## 1. 무엇을 바꾸고 무엇을 숨겼는가

두 환경에서 물체의 **mismatch**는 2.0 N reference와 impact 사이에 실제 변형 방향이 바뀌는 사건이다. Session별 mismatch 확률 e의 평균은 모두 29%다. A에서는 e=0인 session이 42%, e=0.5가 58%다. B에서는 e=0.25가 84%, e=0.5가 16%다. 이전 v56의 결과로 배포할 정책을 미리 대응시켰다. A에서는 네 번째 측정을 항상 impact로 정하는 **fixed 3R+I**, B에서는 첫 세 reference를 보고 네 번째 측정을 고르는 **v54 gate**가 유리하다고 가정했다. 새 surface 48개 모두에서 평균 Brier의 방향이 재현됐다. [사전 채점 명세](SCORE_PLAN_KO.md)

- **8×1과 4×2.** Surface별로 서로 다른 sacrificial object 8개에 reference 1회, impact 1회를 가한다. 두 센서가 각 접촉을 동시에 읽는다. 8×1은 매번 새 session을 시작하고, 4×2는 같은 session에서 물체를 한 번 교체한다. 각 물체 쌍의 ID에 대한 공정한 동전 던지기로 AB/BA 순서를 정했고, 원래 첫 물체가 먼저 온 비율은 50.075%였다. 4×2 모든 조건에서 순서별 cross-sensor double mismatch 차이의 95% CI가 0을 포함했다. 이번 simulator에는 실제 마모나 비가역적인 carryover가 구현되어 있지 않다. [순서와 판독 감사](PILOT_KO.md)
- **독립 개입 두 가지.** Physical persistence를 켜면 쌍의 두 물체가 e를 공유한다. 끄면 물체마다 e를 다시 뽑는다. 별도로 **sensor latch**를 켜면 각 센서가 각 session에서 26.58% 확률로 특정 mismatch bit를 계속 가리킨다. 이 값은 물리와 독립적으로 정하며, 두 센서가 latch될지 여부와 출력값도 서로 독립이다. 어느 조건에서든 각 센서의 reference·impact sign은 각각 2% 확률로 잘못 읽혀, 관측 marginal mismatch는 30.6464%다. 변형을 읽을 수 없는 물체가 절반이며 두 센서가 함께 결측 처리한다. [Pilot 사전 명세](PILOT_PLAN_KO.md)
- **Policy의 정보.** 후보 환경의 분포, sensor model과 두 개씩의 persistence·latch 후보는 알려져 있다. 실제 환경·원인·target label은 숨겼다. Dual 4×2 MAP는 관측한 네 쌍에서 두 nuisance parameter를 marginalize한 후 A가 더 유력할 때 fixed를 고른다. 8×1은 센서의 이상 여부에 관한 정보가 있어도 A/B의 likelihood가 같아 사전에 정한 v54 gate를 고른다. Loss-weighted 정책은 v56에서 고정한 odds threshold 2.016을 쓴다. [Frozen selector](route58.py)
- **독립 단위.** 새 catalog 24개, 새 surface 48개, 기존 두 rate law, held-out deployment object 12,288개를 사용했다. 별도 calibration support를 포함한 새 physical parameter vector 36,864개는 검사한 과거 122,880개와 중복되지 않았다. 각 surface·환경·설계·원인 조건에서 2,048개의 pilot을 생성했다. 각 factorial cell의 98,304개 반복은 새로운 독립 표면의 수가 아니다. [생성 결과](world58.json) · [사전 계획](WORLD_PLAN_KO.md)

## 2. 두 센서에서 물리와 센서의 흔적이 어떻게 다른가

양쪽 물체의 변형이 모두 관측된 쌍만 분모에 썼다. **Same-sensor**는 첫째와 둘째 물체 모두를 sensor 0 또는 모두 sensor 1로 읽은 결과를 평균한다. **Cross-sensor**는 0→1 및 1→0의 두 조합을 평균한다. 비교에 쓰는 접촉·물체·순서는 동일하다. 각 cell의 393,216개 물체 쌍 중 98,233개에서 양쪽 부호가 읽혔고, 98,304개 pilot 중 31,153개에서는 그런 쌍이 하나도 없었다. [관측 분모와 clean control](pilot_results58.json)

| 4×2의 실제 생성 조건 | A same-sensor | A cross-sensor | B same-sensor | B cross-sensor |
|---|---:|---:|---:|---:|
| 둘 다 없음 | 9.33% | 9.33% | 9.41% | 9.39% |
| Sensor latch만 있음 | 14.00% | 9.37% | 14.04% | 9.43% |
| Physical persistence만 있음 | 14.45% | 14.47% | 10.14% | 10.13% |
| 둘 다 있음 | 17.75% | 12.12% | 14.58% | 9.84% |

우연히 두 물체가 모두 mismatch일 확률을 빼면 어떤 조건에서 반복이 늘었는지 더 잘 보인다. Sensor latch만 있을 때 A와 B의 cross-sensor covariance는 각각 +0.005 pp [−0.116, +0.132], +0.021 pp [−0.090, +0.139]였다. 둘 다 있을 때는 각각 +2.738 pp [+2.607, +2.872], +0.433 pp [+0.336, +0.528]가 남았다. 아래 오차막대는 24개 catalog의 pointwise 95% bootstrap CI다. [모든 조건](pilot_results58.json)

![동일한 두 물체를 같은 센서로 읽거나 다른 센서로 읽은 경우의 초과 double mismatch](dual_sensor58.png)

**두 원인이 함께 작동할 때의 분해.** 같은 물체를 두 센서로 읽은 covariance를 W, 다른 물체를 다른 센서로 읽은 covariance를 C라 하면, 이번 독립오차 모형에서 C/W는 실제 두 물체 간 mismatch의 intraclass correlation(ICC)이다. 두 센서의 attenuation이 분자와 분모에 같은 비율로 들어가 상쇄되기 때문이다. Sensor latch 비율은 알려진 marginal mismatch와 sign error까지 써서 별도로 추정했다. 다음은 latent label에 접근하지 않고 pooled pilot에서 추정한 값이다. 기여량의 단위는 same-sensor double mismatch에서 marginal의 곱을 뺀 percentage points다. [수식·CI·잔차](PILOT_KO.md) · [관측 기반 추정](moment_results58.json)

| 환경 | 실제 physical ICC | 추정 physical ICC [95% CI] | 추정 latch [95% CI] | 물리 기여 | 센서 기여 |
|---|---:|---:|---:|---:|---:|
| A | 0.2958 | 0.2909 [0.2782, 0.3039] | 26.64% [26.20, 27.08] | 3.732 pp | 4.659 pp |
| B | 0.0408 | 0.0459 [0.0356, 0.0561] | 26.55% [26.16, 26.97] | 0.589 pp | 4.644 pp |

따라서 두 요인이 함께 있는 A에서는 물리와 센서 모두 상당 부분을 설명하고, B에서는 같은 센서에서 관측된 반복의 대부분을 latch가 설명한다. v57에서는 generator가 쌍마다 e를 공유하도록 했고 primary sensor에는 이 latch가 없었다. v58은 그때와 비슷한 관측이 다른 원인에서도 생길 수 있는지를 검증한 실험이다. 실제 표면에서의 원인 비율은 아직 측정하지 않았다.

## 3. 한 표면에서 8회만 지불했을 때의 원인 판별력

같은 simulator의 정확한 likelihood로 각 pilot에서 두 원인의 on/off를 따로 추론했다. 환경 A/B를 모르는 상태에서 합산했고, 4개 원인 조합은 같은 비중으로 채점했다. 아래 CI는 paired catalog bootstrap 10,000회의 pointwise 95% 구간이다. [별도 사전 계획](MECHANISM_PLAN_KO.md) · [구현 및 confusion matrix](mechanism58.json)

| 여덟 impact를 보고 맞히는 대상 | 8×1 | 4×2 | 4×2 − 8×1 |
|---|---:|---:|---:|
| Physical persistence 유무 | 50.00% | 52.22% | +2.22 pp [+2.15, +2.30] |
| Sensor latch 유무 | 68.66% | 67.66% | −1.00 pp [−1.12, −0.87] |
| 두 원인의 조합을 동시에 맞힘 | 34.33% | 35.43% | +1.10 pp [+1.01, +1.18] |

8×1에는 물체별로 독립적인 sensor session이 여덟 개 있어 같은 물체의 두 센서가 불일치하는 빈도를 관찰할 기회가 많았다. 4×2에서는 그중 네 번을 같은 session 내 반복에 쓴다. 실제 physical persistence가 있던 경우에도 4×2의 sensitivity는 38.58%였고, 두 원인이 모두 있던 pilot에서 두 원인을 모두 맞힌 비율은 20.42%였다. **많은 표면을 모아 ICC를 복원하는 정밀도와 개별 표면의 원인을 맞히는 능력에는 큰 차이가 있다.** [모든 class의 confusion matrix](mechanism58.json)

## 4. 같은 예산에서 최종 정책이 좋아졌는가

A/B와 네 persistence × latch 조건을 균등 가중했다. 모든 정책이 동일한 frozen continuous joint decoder를 사용했다. 양성 label은 PyBullet에서 residual tangential speed가 0.020 m/s 이하인 stick이다. 평균 held-out Brier는 새 물체의 binary outcome에 대해 event 확률을 적분했다. Session loss p90은 같은 128개 물체의 binary deployment draw로 얻은 session 평균 loss 분포에서 계산했다. 표의 판별률은 A에서 fixed, B에서 gate를 옳게 고른 balanced accuracy이다. [평가 코드](score58.py) · [정책별 원자료](score58.json)

| 정책 | 정책 판별률 | 평균 held-out Brier ↓ | Session loss p90 ↓ |
|---|---:|---:|---:|
| Dual 8×1 MAP · 항상 v54 gate와 같은 행동 | 50.000% | 0.1605330708 | 0.189880252 |
| Dual 4×2 MAP | **51.659%** | 0.1605397313 | 0.189883530 |
| Single 4×2 MAP | 51.483% | 0.1605406649 | 0.189884558 |
| Dual 4×2 loss-weighted | 50.003% | 0.1605330702 | 0.189880252 |
| 항상 fixed 3R+I | 50.000% | 0.1605543232 | 0.189892814 |
| 실제 환경을 아는 privileged 선택 | 100.000% | 0.1605119246 | 0.189847335 |

- **Grouping의 효과.** Dual 4×2 − dual 8×1에서 판별률은 +1.659 pp [+1.539, +1.789], 평균 Brier는 **+6.661×10⁻⁶** [+4.201, +9.318], session p90은 +3.278×10⁻⁶ [−6.736, +22.100]였다. 첫 두 비교 대상인 8×1과 single 4×2에 대해 endpoint별 동시 95% CI를 썼다. A에서 올바르게 fixed로 바꾸면 평균 42.29×10⁻⁶을 회수하지만 B에서 잘못 바꾸면 84.80×10⁻⁶을 잃는다. 실제 A에서는 38.00%, B에서는 34.68%를 fixed로 바꿨다. 두 환경의 손익이 다른 것이 평균 악화의 이유다. [환경별 손익](SCORE_KO.md)
- **두 번째 센서의 효과.** 같은 4×2에서 dual MAP − single MAP의 판별률은 +0.176 pp [+0.124, +0.223], 평균 Brier는 **−0.934×10⁻⁶** [−1.244, −0.646], p90은 −1.028×10⁻⁶ [−3.874, +0.522]였다. Sensor nuisance를 적분한 강한 single-channel MAP도 v57 단일 규칙과 가능한 390,625개 관측에서 모두 같은 행동을 골랐다. [전수 열거](exact_route58.json)
- **개입별 차이.** 물리적 지속이 없을 때 dual 4×2의 판별률은 latch off/on 각각 50.02%, 49.98%였다. 지속이 있을 때는 각각 53.59%, 53.05%였다. 물리적 지속이 없는 조건에서 A/B의 물체별 관측 분포는 이론적으로 동일하다. 네 조건의 이론적 MAP 정확도는 이번 binary encoding에서 51.699%였다. [조건별 표](SCORE_KO.md)
- **비용을 반영해도 실용적 이득은 불확실했다.** 4×2 loss-weighted는 786,432개 pilot 중 54회만 fixed로 전환했다. Gate 대비 평균 차이 −0.000568×10⁻⁶의 pointwise CI는 [−0.00222, +0.00108]×10⁻⁶이며 p90은 같았다. 독립 binary replay 두 번에서 primary MAP의 8×1 대비 평균 손해는 +7.939, +5.228×10⁻⁶이었다. 같은 solver의 ultra resolution에서도 +8.012×10⁻⁶이었다. [Replay와 solver refinement](score58.json)
- **Raw sign의 정보를 버렸는지도 확인했다.** 사후 점검에서 센서별 mismatch 두 bit로 축약하는 대신 두 reference sign과 두 impact sign을 모두 유지해 동일 MAP를 다시 계산했다. 4×2 판별률은 51.710%로 0.051 pp 더 높아졌으나, 평균 Brier는 0.160539893으로 primary보다 0.162×10⁻⁶ 높았다. 이 추가 분석은 같은 자료의 exploratory check이고 별도 GPU latency를 측정하지 않았다. [Full-sign 코드](fullsign58.py) · [결과](fullsign58.json)

## 5. 경제적 비용과 GPU p90

Pilot은 두 설계 모두 **8 impacts + 8 references, 16 contacts·16 resets**다. 각 접촉을 두 센서가 동시에 읽으므로 최대 32개 sign readout을 얻는다. 물체가 inactive여도 비용을 지불한다. 배포 직후 첫 session의 calibration까지 포함하면 24 contacts·24 resets다. 8×1은 8회, 4×2는 4회 독립 session을 시작하며, 같은 물체에 추가 측정을 하지 않는다. 상수 정책은 동일 예산 비교에서 pilot을 지불하지만 실제 운영 시 그 pilot을 생략해도 예측이 유지된다. [비용 장부](GPU_WORLD_KO.md)

Reference와 impact에 각각 reset 포함 1.04초를 가정하고 extra setup을 0으로 두면 첫 deployment까지 모든 정책이 22.56초다. Impact가 2.08초로 늘면 공통 padded cap은 31.92초다. Padding 전 dual 4×2와 8×1은 각각 31.49초, 31.24초다. 마지막 impact를 58.67%, 35.08% 구입하는 차이에서 비롯된다. 실물의 sensor 설치·동기화·setup·reset 시간은 미측정이다. [정책별 ledger](GPU_WORLD_KO.md)

**GPU p90.** Node55의 Slurm allocation 2744468에서 0 MiB·utilization 0%·compute PID 없음으로 확인한 GPU를 사용했다. Forecast worker가 끝난 뒤 다시 idle을 확인해 같은 GPU로 latency를 측정했다. 각 정책은 24개 catalog에서 총 72회, 회당 512개 held-out object를 처리했다. Raw sign encoding, policy 선택, decoder와 실제 GPU prediction을 포함해 CUDA를 양끝에서 동기화했다. 기록된 GPU forecast는 독립 replay와 일치했다. [Benchmark 설계](TIMING_PLAN_KO.md) · [종료·실행 기록](GPU_WORLD_KO.md)

| 정책 | GPU p90 (ms/512 objects) |
|---|---:|
| Dual 8×1 MAP | 523.374 |
| Dual 4×2 MAP | 521.580 |
| Dual 4×2 loss-weighted | 521.238 |
| Single 4×2 naive MAP | 519.949 |
| 항상 v54 gate | 522.299 |
| 항상 fixed 3R+I | 519.942 |

Dual 4×2 − dual 8×1의 GPU p90 차이는 **−1.793 ms**였고 paired 95% CI는 [−4.326, +1.261] ms, 다섯 비교를 보정한 동시 95% CI는 [−6.244, +1.900] ms였다. 다섯 timing 대조에서 모두 0을 포함했다. 측정한 48개 surface pilot에서 4×2 MAP는 19회, loss-weighted는 0회 fixed를 골랐다. 후자의 희귀 전환 branch만의 GPU p90은 검증되지 않았다. [GPU p90 수치와 순서 균형](timing58.json)

## 6. 무엇을 논문에 주장할 수 있는가

**이 실험이 식별할 수 있는 것은 sensor-specific error다.** 독립 sensor latch만으로는 cross-sensor covariance가 생기지 않는다는 정량 예측, 4개 개입에서의 관측, latent label을 보지 않고 계산한 ICC가 서로 일치했다. 물체 순서 무작위화는 특정 물체 ID가 매번 첫째라는 설명을 배제한다. 같은 fixture가 두 센서에 일으키는 오류, 순서를 막론하고 첫 impact가 다음 표면을 바꾸는 효과는 여전히 가능하다. [계측 결과](PILOT_KO.md)

**두 센서가 모두 동의해도 원인이 식별되지 않는 반례가 있다.** 실제 A의 physical persistence와, 물체는 독립이지만 session의 29.5775%에서 두 센서가 같은 가짜 신호를 받는 common-mode model을 만들었다. 두 모형에서 가능한 25개 pair pattern의 확률이 일치한다. 같은 noise로 대응시킨 262,144개 물체 쌍에서 두 모형이 보고한 sign 기반 관측은 단 하나도 달라지지 않았으나, 실제 두 물체가 모두 mismatch인 비율은 14.47%와 8.33%였다. 29.5775%는 식별 불가능성을 보이는 반례의 설정값이다. 이 반례는 primary 평균에 포함하지 않았다. [수학적 계산](pilot_results58.json) · [독립 구현과 동일 관측 검산](mechanism58.py) · [검산 수치](mechanism58.json)

**논문에는 식별 가능한 조건과 불가능한 조건, 그리고 판별 개선이 행동 개선으로 이어지는 경계를 함께 제시할 수 있다.** 4×2를 일률적으로 추천하는 주장은 이번 결과가 뒷받침하지 않는다. 실물 확인의 우선순위는 표면을 그대로 둔 상태에서 둘째 물체 직전 한 센서만 무작위 reset하고, 독립 장착·독립 clock의 high-speed camera로 일부 실제 변형 방향을 blind audit하는 것이다. Sensor reset을 했을 때 same-sensor 반복만 사라지고 별도 instrument의 cross-sensor 반복과 독립 deployment stick이 유지되는지 평가하면, 단순히 센서 수를 늘렸을 때보다 강한 외부 검증이 된다. [문헌, 대안 설명 및 후속 질문](LITERATURE_KO.md)

## 7. 재현 자료

서버 경로는 `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/dual_sensor_v58`다. 생성과 CPU score, 실제 GPU timing을 분리했다. 새 12,288개 물체에서 두 deformation 방향의 fine/ultra solver endpoint 24,576개 중 897개(3.65%)가 달라졌다. 이 수치는 동일 solver의 해상도 민감도다. 새로운 solver나 실물 장비에 대한 독립 검증은 남아 있다. [Deployment checksum](deployment58.sha256) · [물체·GPU 감사](GPU_WORLD_KO.md) · [채점 checksum](score_results58.sha256) · [사전 명세](MECHANISM_PLAN_KO.md)


## 문헌 검토와 연구 심사

### 1·6. 논문별 질문, 방법, 표본, 결과와 한계

| 논문 | Research question · 방법 | 표본 및 독립 단위 | 관찰 결과 및 결론 | v58과의 차이 |
|---|---|---|---|---|
| Hui & Walter (1980), *Estimating the Error Rates of Diagnostic Tests* | Gold standard 없이 두 binary instrument의 오류율을 구할 수 있는가? 두 population의 joint outcome에 maximum likelihood를 적용했다. | Tuberculosis skin test 두 종. 미국 학교 집단 555명, 요양소 집단 1,322명. | Prevalence가 서로 다르고 error rate가 집단 간 일정하며 conditional independence가 성립하는 모형에서 두 검사와 prevalence를 추정할 수 있었다. | v58은 두 환경의 marginal mismatch를 일부러 같게 한다. 식별 정보는 물체 간 dependence에서 나온다. 이 정리를 그대로 적용할 수 없다. [원문, §2–3](https://www.medicine.mcgill.ca/epidemiology/hanley/bios601/candh-ch0102/HuiWalter1980.pdf) |
| Dendukuri & Joseph (2001), *Bayesian Approaches to Modeling the Conditional Dependence Between Multiple Diagnostic Tests* | 두 검사가 같은 개체에서 함께 틀리면 추론이 어떻게 바뀌는가? Fixed 및 random effects로 dependence를 모형화했다. | 캐나다 정착 캄보디아 난민 162명의 Strongyloides 검사. Stool 40명, serology 125명이 양성이었다. | 해당 두 검사 모형에서 관측표 하나가 여러 latent 설명과 양립했고, dependence에 대한 prior가 posterior를 바꿨다. | 개체 간 순서, sensor reset 개입 또는 향후 policy loss를 다루지 않았다. [원문, Table 1·§3–5](https://joseph.research.mcgill.ca/publications/Methodological/conddep_diag.pdf) |
| Dendukuri et al. (2026, preprint), *Improving interpretation of latent class models … via directed acyclic graphs* | 검사마다 실제로 측정하는 양과 최종 목표를 어떻게 구별하는가? DAG, simulation, 기존 두 dataset 재분석을 수행했다. | 소아 tuberculosis·leptospirosis 자료와 simulation. 임상 데이터의 재분석 단위는 환자다. | 측정 대상인 measurand를 최종 질환 상태와 동일시하면 simulation에서 accuracy·prevalence가 왜곡될 수 있었다. | 변형 방향, sensor 출력, 이후 stick이 다른 변수라는 설계 근거다. Contact에서의 효과 크기는 이 논문으로 정할 수 없다. [원문, §2–5](https://arxiv.org/abs/2607.14473) |
| Li, Dong & Adelson (ICRA 2018), *Slip Detection with Combined Tactile and Visual Information* | GelSight와 외부 영상을 합치면 로봇 grasp의 slip을 감지할 수 있는가? 두 stream을 이용한 DNN을 학습했다. | 일상 물체 94개. 새 물체 10개·152 grasps에서 평가했다. | 시각·촉각을 합친 분류에서 최대 88.03% 정확도를 보고했다. | Slip classification은 두 sensor의 오류가 독립이라는 검정과 다른 endpoint다. 이 결과로 8개 impact의 grouping 또는 Brier 개선 크기를 예측할 수 없다. [논문](https://arxiv.org/abs/1802.10153) |
| Acosta, Yang & Posa (RA-L 2022), *Validating Robotics Simulators on Real-World Impacts* | 시뮬레이터가 실제 impact trajectory를 어디까지 재현하는가? 실측과 여러 engine을 대조했다. | Cube toss 550 trajectory, Cassie jump 22회. | Inelastic cube impact와 로봇의 전체 움직임은 재현되는 경우가 있었으나 elasticity와 착지 세부에 오차가 남았다. | 해당 실측자료에 v58의 물체 쌍별 reference·impact sign과 독립 후속 deployment가 모두 있는 것은 아니다. [원문, §III–IV](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf) |

### 2–3. 주제별 종합과 knowledge map

**Session state → 물체별 실제 reference–impact 변형 부호 → 두 instrument의 판독 → pilot에 근거한 정책 → 별도 물체의 stick 확률**이 v58의 인과 가설이다. 개별 instrument의 latch는 판독 단계에, 두 센서에 공통인 drift는 분기 전 단계에 영향을 준다. Measurand와 최종 목표를 그려 나누는 접근은 진단 문헌에서 제안됐다. 이를 contact와 terminal Brier에 연결한 것은 이번 연구의 설계다. [Dendukuri et al.](https://arxiv.org/abs/2607.14473)

**같은 물체에서 두 센서 간 일치**, **서로 다른 두 물체에서 같은 센서의 반복**, **서로 다른 두 물체와 두 센서를 교차한 반복**은 다른 통계량이다. 첫 번째만으로 error의 출처를 결정하기 어려울 수 있다. 시각·촉각 결합이 slip prediction에 도움이 된 실제 사례는 있지만, 두 채널을 추가하는 것만으로 physical truth와 shared artifact를 분리할 수 있다는 증거는 아니다. [Dendukuri & Joseph](https://joseph.research.mcgill.ca/publications/Methodological/conddep_diag.pdf) · [Li et al.](https://arxiv.org/abs/1802.10153)

### 4–5·7. 조건에 따라 달라지는 결론과 근거 수준

| 근거 | 판단 |
|---|---|
| 가정을 명시한 식별 이론 | Hui–Walter의 결론과 dependence를 허용한 Bayesian 분석은 서로 다른 sample 구조와 가정을 쓴다. 전자는 서로 다른 prevalence가 정보이고 후자는 한 population의 dependence가 불확실성이다. 두 결론을 같은 조건의 실험적 충돌로 보기는 어렵다. [Hui & Walter](https://www.medicine.mcgill.ca/epidemiology/hanley/bios601/candh-ch0102/HuiWalter1980.pdf) · [Dendukuri & Joseph](https://joseph.research.mcgill.ca/publications/Methodological/conddep_diag.pdf) |
| 제한된 경험적 지지 | 시각·촉각이 서로 다른 정보를 줄 수 있다는 것은 실제 grasp에서 관찰됐다. 그 정보가 surface 교체 후 policy 선택의 calibration을 개선하는지는 별도 실험 대상이다. [Li et al.](https://arxiv.org/abs/1802.10153) |
| 외부 타당성 및 공백 | 실제 impact에서 simulator가 놓치는 dynamics가 있다. 센서 오차·물리 state·순서에 따른 마모 중 무엇이 v58의 관측 패턴을 만드는지는 현장 데이터에서 미결이다. [Acosta et al.](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf) · [Dendukuri et al.](https://arxiv.org/abs/2607.14473) |

### 8. 후속 연구 질문 열 가지

| 질문 | 공백과 의미 | 변수 및 검증 방법 |
|---|---|---|
| 1. Sensor만 재초기화하면 같은 센서의 연속 mismatch가 줄어드는가? | Sensor history와 dynamics의 혼동. [Dendukuri & Joseph](https://joseph.research.mcgill.ca/publications/Methodological/conddep_diag.pdf) | Surface는 유지하고 둘째 물체 직전 한 sensor만 무작위 reset한다. |
| 2. 독립 장착·clock에서도 cross-sensor 효과가 남는가? | Shared upstream error는 두 기기에 함께 나타날 수 있다. [Dendukuri et al.](https://arxiv.org/abs/2607.14473) | Mount·clock 공유 여부를 교차하고 cross-sensor covariance를 잰다. |
| 3. 외부 high-speed camera를 audit용으로만 쓰면 얼마나 교정되는가? | 두 센서의 단순 일치는 gold standard가 아니다. [Hui & Walter](https://www.medicine.mcgill.ca/epidemiology/hanley/bios601/candh-ch0102/HuiWalter1980.pdf) | 일부 trial의 변형 방향을 blind annotation으로 판독하고 error model을 추정한다. |
| 4. 같은 session이 언제까지 같은 상태인가? | 순간 contact와 장시간 마모에는 다른 물리가 작용할 수 있다. [Acosta et al.](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf) | 두 물체 간 시간·거리·누적 충격 횟수를 교차한다. |
| 5. 첫 물체의 재질이 두 번째 물체의 부호를 바꾸는가? | 순서 무작위화만으로 carryover 자체가 없어지지는 않는다. [Acosta et al.](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf) | AB/BA와 새 표면 대조를 함께 만들고 둘째 물체의 결과를 비교한다. |
| 6. 두 채널이 부호를 읽지 못하는 물체군이 다른가? | 동일 measurand를 측정한다는 가정은 instrument별로 깨질 수 있다. [Dendukuri et al.](https://arxiv.org/abs/2607.14473) | 투명·반사·compliant sample에서 missingness와 sign error를 따로 추정한다. |
| 7. 연속 deformation field가 부호 이상의 정보를 주는가? | 영상·촉각에서 binary sign으로 줄이며 버리는 단서가 있다. [Li et al.](https://arxiv.org/abs/1802.10153) | 동일 횟수·같은 terminal에서 sign과 displacement field를 맞춘다. |
| 8. Source에서 읽은 sensor prior가 새 장비에 전이되는가? | 모형 식별이 prior에 의존할 수 있다. [Dendukuri & Joseph](https://joseph.research.mcgill.ca/publications/Methodological/conddep_diag.pdf) | Sensor unit·날짜를 통째로 hold out하고 사전 지정한 prior 범위를 비교한다. |
| 9. 잘못된 sensor로부터 오는 비용이 희귀한 slip에 집중되는가? | Classification 지표와 실제 기능에 차이가 날 수 있다. [Li et al.](https://arxiv.org/abs/1802.10153) | 환산 비용을 사전 지정하고 Brier, session tail, 실제 미끄러짐을 함께 측정한다. |
| 10. Independent solver와 실물에서 최적 정책 부호가 일치하는가? | Simulation의 contact dynamics는 표면과 속도에 민감하다. [Acosta et al.](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf) | 같은 intervention을 다른 engine과 실물에 적용하고 새 물체에서 paired loss를 비교한다. |

### 9. 연구 심사

두 센서를 합치는 기법이나 conditional dependence 문제 자체에 최초성을 주장할 근거는 없다. 논문에서 검증할 만한 주장은 **같은 유료 impact에서 물체 간·센서 간 반복을 어떻게 배치하느냐가 원인 식별과 최종 행동에 서로 다른 영향을 준다**는 것이다. Competing sensor model, 같은 센서 수의 대조, sensor-aware single-channel policy, 실제로 발생한 손익을 함께 제시해야 이 주장의 범위를 정할 수 있다. 이 다섯 원문은 해당 연구 질문을 위해 선정한 문헌이며 체계적 전수조사는 아니다. [Dendukuri & Joseph](https://joseph.research.mcgill.ca/publications/Methodological/conddep_diag.pdf) · [Li et al.](https://arxiv.org/abs/1802.10153) · [Acosta et al.](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf)
