# v59 · 같은 표면에서 한 센서를 reset하면 물리적 변형의 지속과 공통 센서 오류를 구별할 수 있는가?

**한 센서의 reset만으로는 두 원인을 구별할 수 없었다. 독립 high-speed camera를 함께 쓰면 구별할 수 있었지만, 8회 impact에서의 정확도는 55.81%였다.** Camera를 쓸 때 reset을 추가하면 이후 유리한 정책을 고르는 비율도 54.84%에서 55.23%로 올랐고 평균 held-out Brier loss가 0.976×10⁻⁶ 줄었다. 같은 비교에서 session loss p90의 개선은 확인되지 않았다. 이 결과는 24개 새 synthetic catalog를 사용한 서버 실험이다. Camera는 측정 오류를 제어한 sign model과 실제 grayscale frame을 생성해 추적하는 두 방식으로 모사했다. [원인 검증](PILOT_KO.md) · [정책 검증](SCORE_KO.md) · [영상 결과](frame59.json)

## 1. 한 표면에서 관측하는 것과 개입하는 것

각 pilot에서 같은 표면의 session 4개를 시작한다. 각 session에서 서로 다른 물체 두 개의 순서를 무작위로 정하고, 각 물체에 2.0 N reference와 impact를 한 번씩 가한다. **Mismatch**는 reference와 impact 때의 변형 방향이 달라진 사건이다. 균등하게 평가한 두 환경 A·B에서 실제 marginal mismatch는 모두 29%다. V56의 예측에 따라 이후 deployment에서 A는 네 번째 관측을 항상 impact로 사는 **fixed 3R+I**, B는 세 reference를 보고 네 번째 관측을 구입하는 **v54 gate**가 유리하다고 지정했다. v59의 새 표면 48개 각각에서 두 행동의 평균 Brier 순위가 재현됐다. [사전 채점 계획](SCORE_PLAN_KO.md) · [결과](score59.json)

두 환경 각각에 겉보기가 같은 두 원인을 만들었다. **P(physical persistence)**에서는 같은 session의 물체들이 실제 mismatch 확률을 공유한다. **C(common-mode error)**에서는 실제 물체들의 mismatch가 독립이지만 두 전자센서가 받는 upstream 신호가 때때로 똑같은 가짜 방향을 가리킨다. A에서 C의 공통 오류 빈도는 29.58%, B에서는 4.08%로 설정했다. 그래야 A와 B 각각에서 두 물체의 전자 신호 전체 분포가 P일 때와 같아진다. P/C는 pilot의 관측 원인이고, A/B는 이후 deployment의 기대 손익을 정한다. 각 원인과 환경을 독립적으로 조합했지만 C 안에서의 **오류 빈도는 의도적으로 A/B에 연결했다**. 이 선택을 풀었을 때의 결과도 아래에 보고한다. [생성 명세](PILOT_PLAN_KO.md) · [전이 검산](transfer59.json)

각 전자센서에는 별도로 자기 신호에만 생기는 **local latch**가 있다. Session당 26.58% 확률로 그 장치의 readout이 고정된다. 둘째 물체 직전 동전을 던져 한 센서를 골라 local latch를 지우고, 다른 센서·upstream 신호·물리 표면은 그대로 둔다. 대조군인 **sham**은 같은 센서 ID를 지정하되 전자 reset을 실행하지 않는다. Primary에서는 reset이 local latch를 완전히 해제한다고 가정했다. Camera를 켠 조건에서는 전자센서의 신호를 경유하지 않고 물체를 본다. 전자센서의 개별 sign error는 2%, sign-level camera는 active 물체의 80%에서 판독되며 개별 sign error는 3%다. 정책에는 유료 sign과 순서·reset 배정만 제공하고 실제 원인, 환경, 미래 outcome은 가렸다. [Frozen selector](route59.py) · [생성 코드](pilot59.py)

## 2. 무엇이 원인을 드러냈는가

환경 A에서 **선택한 전자센서**의 double mismatch는 reset 후 17.77%에서 13.07%로 줄었고, 선택하지 않은 센서는 그대로였다. 감소 폭은 P와 C에서 같았다. 즉 reset은 자기 장치의 latch를 지웠다는 증거를 남기지만 그보다 앞단의 오류와 실제 물리 상태가 만든 신호는 구별하지 못한다. Camera를 끄면 reset 유무 모두 P와 C가 가능한 모든 전자 관측에 같은 likelihood를 주므로, 균등 prior에서 원인 판별률의 이론값은 50%다. [계측 결과와 동등성 검산](PILOT_KO.md)

독립 camera가 읽은 double mismatch는 A에서 **P 14.56%, C 9.72%**, B에서 **P 10.43%, C 9.72%**였다. 전자센서에 같은 신호가 반복돼도 camera가 이를 함께 보는지는 실제 변형이 반복됐는지에 달렸다. 표면당 여덟 impact에서 camera와 reset을 모두 적용한 원인 판별률은 A에서 60.14%, B에서 51.49%였다. 전체 55.81%에서도 C의 65.18%를 P로 오판했다. 두 원인이 관측상 분리될 수 있다는 사실과 한 표면에서 확신을 갖고 진단할 수 있는지는 구분해야 한다. [조건별 판별률과 CI](diagnostics59.json)

| 4×2 MAP 조건 | 원인 P/C 판별률 | 정책 A/B 판별률 | 평균 held-out Brier ↓ | Session loss p90 ↓ |
|---|---:|---:|---:|---:|
| Sham | 50.000% | 53.164% | 0.160948655 | 0.192561001 |
| 한 센서 reset | 50.000% | 53.109% | 0.160948847 | 0.192561403 |
| Sham + 독립 camera | 55.168% | 54.843% | 0.160947990 | 0.192559555 |
| 한 센서 reset + 독립 camera | **55.813%** | **55.227%** | **0.160947014** | 0.192559406 |

원인과 정책 모두 균등 가중한 balanced accuracy다. 아래 그림 위쪽은 별도 영상 실험에서 사용한 한 물체의 실제 합성 frame이다. 위 표식은 변형을 따라가고 아래 표식은 camera 흔들림을 보정하는 fixture marker다. 아래 막대는 전체 sign-level pilot에서 측정했다. [Figure 코드](figure59.py) · [Primary 수치](score59.json)

![실제 합성한 reference와 impact 영상 및 원인과 정책 판별률](sensor_reset59.png)

**Reset 효과를 직접 추적한 검산.** 첫 물체의 camera와 둘째 물체의 선택된 전자센서 사이의 mismatch covariance는 reset 후 P에서 A +1.329 pp [+1.222, +1.441], B +0.177 pp [+0.080, +0.274]로 증가했다. C에서는 A +0.048 pp [−0.062, +0.153], B +0.002 pp [−0.098, +0.101]였다. 첫 물체의 독립 camera가 읽은 상태에 둘째 물체의 센서가 reset 후 더 가까워지는 현상은 실제 물리적 지속이 있을 때 나타났다. [음성 대조와 조건별 covariance](PILOT_KO.md)

**Camera는 같은 물체에서의 불일치와 물체 간 지속 신호를 모두 제공했다.** Reset + camera에서 두 물체의 dependence를 무시하고 같은 물체의 전자센서–camera agreement만 써도 원인 판별률은 **55.45%**였다. 전체 joint likelihood는 여기에 **0.359 pp** [95% CI +0.324, +0.398]를 더했다. Camera만 쓰면 **51.50%**였다. 따라서 원인 판별 이득 대부분은 직접 관찰한 물체에서 전자센서와 camera가 서로 다르게 읽었는지로 설명된다. 반면 A/B 정책 판별에는 물체 간 dependence를 살린 이득이 +1.156 pp였다. [탐색적 단서 제거 실험](cues59.json)

## 3. 판별과 실제 정책 손익

새 24개 catalog에 surface 2종과 기존 rate law 2종을 교차하고, 12,288개 held-out 물체를 같은 frozen continuous joint decoder로 평가했다. Positive event는 residual tangential speed ≤0.020 m/s인 stick이다. 평균 Brier는 새 물체에서 두 변형 방향을 각각 시뮬레이션한 binary outcome에 대해 조건부 기대값을 구했다. Session loss p90은 독립 binary deployment draw에서 session별 128개 물체의 평균 loss를 낸 뒤 구했다. Surface·원인·환경마다 2,048개로 반복한 pilot은 유한 관측의 변동을 적분하기 위한 것이다. 추론의 독립 단위는 24개 catalog이며, 주 비교는 paired bootstrap 5,000회와 endpoint별 4개 contrast의 동시 95% CI를 사용했다. [평가 명세](SCORE_PLAN_KO.md) · [원자료](score59_cells.npz)

| 비교 · MAP 앞 조건에서 뒤 조건을 뺌 | 정책 판별률 차이 (pp) | 평균 Brier 차이 (×10⁻⁶) | Session p90 차이 (×10⁻⁶) |
|---|---:|---:|---:|
| Reset − sham, camera on | **+0.385** [+0.331, +0.444] | **−0.976** [−1.269, −0.706] | −0.149 [−3.081, +1.934] |
| Reset − sham, camera off | −0.055 [−0.118, +0.009] | +0.191 [+0.089, +0.297] | +0.402 [−1.043, +1.535] |
| Camera on − off, reset | +2.119 [+2.027, +2.224] | −1.832 [−2.201, −1.538] | −1.997 [−5.048, 0.000] |
| Camera on − off, sham | +1.678 [+1.565, +1.809] | −0.665 [−0.791, −0.530] | −1.445 [−4.154, 0.000] |

Camera가 있을 때 reset은 A에서 fixed를 택하는 비율을 44.38%에서 41.59%로, B에서는 34.70%에서 31.13%로 낮췄다. A에서 fixed를 올바르게 쓸 때의 이득은 평균 41.006×10⁻⁶, B에서 잘못 쓸 때의 손해는 87.082×10⁻⁶이었다. B에서 비싼 오판을 줄인 쪽이 전체 손익에 더 크게 반영됐다. 독립 binary replay 두 번에서 평균 reset 효과는 −0.948, −1.018×10⁻⁶였고 같은 solver의 ultra resolution에서는 −1.026×10⁻⁶였다. [환경별 선택과 replay](SCORE_KO.md)

**강한 대조는 항상 v54 gate를 쓰는 정책이다.** 이 대조의 평균은 0.160942010으로 reset + camera MAP보다 5.004×10⁻⁶ 낮았다. 항상 fixed 3R+I는 0.160965048이었다. V56의 손익 비율로 threshold를 미리 고정한 reset + camera **loss-weighted** 정책만 gate보다 평균이 소폭 낮았다(0.160941708, 개선 0.302×10⁻⁶, pointwise 95% CI 0.152–0.448×10⁻⁶). 실제 환경을 안다면 회수할 수 있는 최대 이득 20.503×10⁻⁶의 약 1.47%다. [모든 정책의 결과](score59.json)

**장치 오류 빈도를 표면에서 분리하면 손익의 방향이 바뀌었다.** 위 결과에서는 A/B를 흉내 내도록 만든 common-mode 발생률 자체가 최적 행동의 단서다. 같은 synthetic pilot을 재배정해 C 조건의 장치 오류 빈도만 deployment 환경과 독립으로 만들면, reset + camera MAP의 정책 판별률은 55.23%에서 **51.30%**, 그 관계를 반전하면 **47.37%**였다. Loss-weighted의 gate 대비 평균 Brier 차이는 각각 **−0.302 → +0.680 → +1.661**(×10⁻⁶)이었다. 총 관측·terminal·결과는 고정했다. 이는 동일 자료의 사후 stress test이며, 실물에서 오류와 표면 마찰이 어느 방향으로 연관되는지를 측정한 결과는 아니다. [전이 대조와 CI](transfer59.json)

## 4. High-speed camera를 영상으로 구현했을 때

Camera 오차를 확률로만 가정한 데 따른 민감도를 확인하기 위해 같은 새 catalog의 surface별 사전 지정된 앞 64개 pilot을 따로 평가했다. 총 12,288개 8-impact pilot 조건, 98,304개 물체 관측, reference·impact 각각을 합쳐 196,608개 clip이다. 이 수에는 같은 specimen을 대응시킨 반사실 조건이 포함된다. 1,000 fps로 0–5 ms에 13×27 grayscale frame 6장을 만들었다. Exposure 내 두 시점의 영상, 부분 가림, subpixel shake와 pixel noise를 모사했다. Reader는 위·아래 marker의 brightness centroid 간 거리를 구하고 0.5 pixel 이상의 이동에서만 방향을 출력했다. Renderer만 실제 변형 부호를 사용하며 reader에는 pixel만 들어간다. Reset과 sham의 영상은 동일하다. Source likelihood와 정책 threshold를 다시 맞추지 않았다. [영상 사전 명세](FRAME_PLAN_KO.md) · [Renderer와 blind reader](frame59.py)

| 같은 영상 subset의 MAP | 원인 판별률 | 정책 판별률 | 평균 held-out Brier |
|---|---:|---:|---:|
| Sham + sign-level camera | 55.07% | 55.09% | 0.160947555 |
| Reset + sign-level camera | 55.84% | 55.57% | 0.160946515 |
| Sham + frame tracker | 55.55% | 55.38% | 0.160946324 |
| Reset + frame tracker | 56.28% | 55.75% | 0.160945198 |

영상 tracker에서도 reset의 증분은 원인 판별 **+0.732 pp** [+0.529, +0.928], 정책 판별 **+0.366 pp** [+0.081, +0.643], 평균 Brier **−1.126×10⁻⁶** [−1.566, −0.710]였다. 이 구간은 별도 paired catalog bootstrap의 pointwise 95% CI다. 49,572개 active 관측 중 두 부호를 모두 읽은 것은 33,128개(66.8%)였고, 판독한 쌍 중 잘못된 mismatch는 0개였다. 두 가지 marker만 있는 단순한 장면과 엄격한 판독 거절 기준이 유리하게 작용했다. 반사·복잡한 재질·실제 contact trajectory를 촬영한 결과로 볼 수 없다. [영상 집계](frame59.json)

**Trigger timing이 이 결론을 제한한다.** 사후 stress test로 impact clip의 20%를 2 ms 늦게 시작하면서 reader와 reference timing은 그대로 뒀다. 늦은 frame에서는 peak 이후 되돌아오는 움직임을 읽을 수 있다. 판독 가능한 30,951건 중 4,198건에서 mismatch를 잘못 읽었다. Reset + camera 원인 판별률은 **56.28% → 53.74%**, 정책 판별률은 **55.75% → 54.19%**, 평균 Brier는 **0.160945198 → 0.160949423**이 됐다. 같은 subset의 camera-off reset 평균은 0.160948734였다. 같은 contacts라도 trigger가 잘못 정렬되면 camera가 평균 손실을 늘릴 수 있다. 2 ms는 합성 pulse와 coupling한 시나리오이며 일반적인 장비 허용오차는 아니다. [Late-trigger 생성 코드](latetrigger59.py) · [민감도 수치](latetrigger59.json)

## 5. GPU p90과 sensing 비용

실제 사용량이 0 MiB, utilization 0%, compute process 0개인 것을 확인한 **node55의 NVIDIA RTX PRO 6000 Blackwell Server Edition 한 장**을 사용했다. 정책별 호출은 512개 held-out object를 처리한다. 첫 benchmark는 메모리에 있는 구입된 sign부터 policy·frozen decoder·GPU prediction까지 측정했다. 두 번째는 메모리에 있는 두 surface의 고속영상 32 clips에서 시작해 CPU marker tracking과 최종 GPU prediction까지 포함했다. 각 호출 양끝에서 CUDA를 동기화했고, 독립 replay와 최대 예측 차이는 0이었다. 카메라의 실제 노출 대기·기계 동작 시간은 별도다. [GPU 실행·재현 기록](GPU_WORLD_KO.md)

| 입력에서 최종 prediction까지 | Sham | 한 센서 reset | Reset − sham, ms [95% CI] |
|---|---:|---:|---:|
| 두 전자센서 sign, camera off | 489.684 ms | 491.465 ms | +1.780 [−371.506, +465.387] |
| 두 전자센서와 camera sign | 484.082 ms | 502.576 ms | +18.494 [−24.718, +512.944] |
| 두 전자센서 sign + 실제 합성 frame에서 tracker 실행 | 483.919 ms | 483.758 ms | −0.161 [−17.435, +4.410] |

Sign benchmark는 24개 catalog, 정책당 72회 측정했고 표의 구간은 사전 지정한 네 비교의 동시 95% CI다. 일부 catalog에서 모든 방법의 지연이 함께 커져 p90 차이의 불확실성이 컸다. Frame benchmark는 별도 crossover에서 정책당 48회, 단일 reset 대조의 paired 95% CI다. 두 benchmark는 실행 시점·표본이 달라 행 간 차이를 영상 처리 비용으로 해석할 수 없다. Camera와 reset이 GPU p90을 줄이거나 동일하게 유지한다는 통계적 근거는 확보되지 않았다. [Sign timing](timing59.json) · [Frame-to-terminal timing](frame_timing59.json)

모든 조건이 **8 impacts + 8 references, 16 mechanical resets**를 지불한다. 추가된 것은 reset 조건에서 네 번의 **electronic reset**, camera 조건에서 최대 16개 clip(96 frames)의 동시 촬영이다. 첫 deployment의 공통 calibration까지 포함하면 24 contacts·24 mechanical resets다. 장치 reset 1회당 겹치지 않는 시간이 R초, clip당 추가 시간이 K초이면 pilot을 H개 deployment에 재사용할 때 증분은 각각 4R/H와 16K/H초다. R과 K는 실측하지 않았다. Reference·impact에 reset 포함 각 1.04초를 가정하면 첫 deployment의 기계 시간은 모두 22.56초다. Impact가 2.08초면 공통 padded budget은 31.92초, padding 전 MAP의 reset + camera와 sham + camera는 각각 31.488초, 31.509초다. 이 차이는 다음 deployment에서 마지막 impact를 구매하는 비율 차이다. 항상 gate를 쓰는 정책은 실제 배포에서 pilot 자체를 생략할 수도 있다. [비용 장부](cost59.json)

## 6. 논문 기여의 경계와 재현

논문 결과로 채택할 만한 부분은 **무작위 개입이 실제로 작동했어도 개입 지점보다 상류에 있는 두 원인은 여전히 구별되지 않을 수 있다**는 점이다. 여기에 독립 광학 판독을 추가한 효과를 원인 판별과 행동 손익으로 분리했고, camera가 upstream 신호를 일부 또는 전부 공유하는 반례도 확인했다. Camera가 upstream을 공유하는 pilot 비중이 0%, 25%, 50%, 100%일 때 reset + camera의 원인 판별률은 각각 **55.81%, 54.36%, 52.91%, 50.00%**였다. 두 원인이 한꺼번에 섞인 실물 표면에서 각 기여량을 8번으로 추정할 수 있다는 결론은 나오지 않는다. [독립성 민감도](diagnostics59.json) · [단서 분석](cues59.json)

**배포 방법으로서의 기여는 아직 제한적이다.** 원인 식별과 달리 행동 손익은 어느 환경에서 무엇을 잘못 사는지에 달려 있다. Source의 fault 빈도 관계를 제거하거나 camera timing을 어긋나게 하면 이득이 사라졌다. 실물 확인의 우선순위는 (1) reset 성공률·settling time과 bystander sensor 무변화를 기록하고, (2) 독립 장착·clock의 camera에 별도 동기화 표식을 두며, (3) 장치 fault 주입률을 표면·날짜와 독립 무작위화한 새 session에서 같은 고정 정책을 평가하는 것이다. 이때 8회 impact와 독립 surface 수를 별도로 보고하고, 유료 policy를 상시 v54 gate와 reset·촬영 시간까지 포함해 대조해야 한다. [관련 원문 비교와 10개 후속 질문](LITERATURE_KO.md)

서버 폴더는 `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/sensor_reset_v59`다. 새 36,864개 parameter vector는 검사한 이전 159,744개와 중복되지 않았다. Fine·ultra에서 24,576개 stick endpoint 중 855개가 달랐다. 두 해상도는 같은 PyBullet solver를 사용하며 기존 contact-law·geometry family를 공유한다. 전체 정책 표, signed pilot, raw frame fixture와 각 시점의 readout, frozen likelihood, 예측·timing replay 및 checksum을 보관했다. [물체 감사](world59.json) · [채점 checksum](score_results59.sha256) · [GPU forecast와 solver checksum](deployment59.sha256) · [Pilot 생성 및 재현](PILOT_KO.md)


## 문헌 검토와 연구 심사

### 1·6. 논문별 질문, 방법, 표본, 결과와 한계

| 논문 | Research question과 방법 | 표본과 평가 단위 | 결과와 결론 | 이번 연구에 적용할 때의 한계 |
|---|---|---|---|---|
| Scott et al. (Automatica 2014), *Input design for guaranteed fault diagnosis using zonotopes* | 서로 다른 fault가 다른 출력을 내도록 어떤 입력을 가해야 하는가? Bounded uncertainty를 둔 linear model의 reachable output set을 구해 separating input을 최적화했다. | 수학적 정리, 수치 예제, 무작위 생성한 50-state system. 독립적인 실물 표면 표본은 없다. | 명시한 조건에서는 후보의 출력 집합을 분리하는 입력과 진단 가능성을 계산할 수 있다. | 센서의 내부 latch만 지우는 조작이 그보다 upstream의 상태를 구별해 준다는 보장은 없다. v59는 출력 분포가 일치하는 반례와 finite-sample accuracy를 다룬다. [원문](https://web.mit.edu/braatzgroup/input_design_for_guaranteed_fault_diagnosis_using_zonotopes.pdf) |
| Dendukuri & Joseph (Biometrics 2001), *Bayesian Approaches to Modeling the Conditional Dependence Between Multiple Diagnostic Tests* | 두 검사가 함께 틀릴 수 있을 때 latent truth를 추론할 수 있는가? Fixed와 random effects를 둔 Bayesian model을 비교했다. | 캄보디아 난민 162명의 serology와 stool test. 두 검사 양성은 각각 125명과 40명. | 해당 모형에서 두 개의 불완전한 검사만으로 모든 오차와 실제 prevalence를 식별하기 어려웠고 prior가 결론에 영향을 주었다. | 질환 판정에는 물체 순서, 물리 개입이나 deployment policy가 없다. 독립성 가정의 한계에만 적용했다. [원문](https://joseph.research.mcgill.ca/publications/Methodological/conddep_diag.pdf) |
| Rigi et al. (Sensors 2018), *A Novel Event-Based Incipient Slip Detection Using Dynamic Active-Pixel Vision Sensor (DAVIS)* | Event-based slip 경보를 외부 관측으로 검증할 수 있는가? 다른 시점에서 1,000 fps camera로 contact area를 재고 LED로 동기화했다. | Baxter robot, 5개 물체, 37회 실험. | 평균 감지 latency 44.1 ms를 보고했고, 서로 다른 각도와 조명·noise가 측정에 영향을 줄 수 있음을 다뤘다. | Contact area로 slip onset을 정한 실측이다. v59의 reference–impact 변형 부호 오류율·가림 확률을 이 값에서 추정할 수 없다. [원문 §4.1–4.4](https://www.mdpi.com/1424-8220/18/2/333) |
| Li, Dong & Adelson (ICRA 2018), *Slip Detection with Combined Tactile and Visual Information* | 촉각·외부 영상을 합쳐 slip을 검출할 수 있는가? GelSight와 camera 영상에 DNN을 학습했다. | 일상 물체 94개, 새 물체 10개에서 152 grasps 평가. | 결합 모델의 최고 slip detection accuracy는 88.03%였다. | 예측에 유용한 두 stream도 같은 fixture 오류를 공유할 수 있다. 단일 새 표면에서 원인 판별이나 reset 효과를 시험하지 않았다. [원문](https://arxiv.org/pdf/1802.10153) |
| Acosta, Yang & Posa (RA-L 2022), *Validating Robotics Simulators on Real-World Impacts* | 실측 충돌에서 보정된 simulator가 어느 dynamics까지 맞히는가? 실제 trajectory와 engine별 prediction을 비교했다. | 550개 cube toss, Cassie의 22개 jump. | Inelastic contact와 전체 운동이 맞는 경우에도 elasticity와 착지의 세부 오차가 남았다. | 이 자료에는 동일 surface의 무작위 sensor reset, 두 물체의 변형 부호, blind camera audit 및 이후 stick outcome이 함께 기록되지 않았다. [원문](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf) |

### 2–3. 주제별 종합과 knowledge map

**실제 변형 → 장치가 공유하는 upstream 신호 → 각 sensor의 local latch → 관측한 부호**가 전자센서의 측정 경로다. 독립 camera는 실제 변형에서 따로 갈라진다. Reset은 이 경로 중 한 장치의 latch에 작용한다. 동일한 전자 신호를 낼 수 있는 두 설명을 구별하려면 조작이나 관측이 두 설명에 다르게 반응해야 한다. 앞부분은 active fault diagnosis와 dependent measurement 연구의 개념이며, 이를 유료 물리 probe와 배포 정책에 연결한 부분이 v59의 가설이다. [Scott et al.](https://web.mit.edu/braatzgroup/input_design_for_guaranteed_fault_diagnosis_using_zonotopes.pdf) · [Dendukuri & Joseph](https://joseph.research.mcgill.ca/publications/Methodological/conddep_diag.pdf)

영상은 두 역할을 갖는다. Rigi et al.은 다른 각도의 고속영상을 외부 판독에 사용했고, Li et al.은 영상과 촉각을 합쳐 예측했다. 한쪽은 sensor를 평가하는 reference로서의 역할이고 다른 쪽은 의사결정에 추가하는 입력이다. v59에서는 동일 camera가 원인 추론과 정책 선택을 얼마나 바꾸는지 별도로 평가했다. [Rigi et al.](https://www.mdpi.com/1424-8220/18/2/333) · [Li et al.](https://arxiv.org/pdf/1802.10153)

### 4–5·7. 연구 공백, 다르게 보이는 결과, 증거의 범위

- **이론적으로 확립된 조건.** 서로 다른 후보가 가능한 모든 측정에 같은 분포를 주면 그 관측만으로 두 후보를 식별할 수 없다. 구별을 보장하는 입력을 구하는 연구는 입력이 도달하는 범위와 가능한 noise를 명시한다. 약한 개입이 항상 충분하다는 결론은 나오지 않는다. [Scott et al.](https://web.mit.edu/braatzgroup/input_design_for_guaranteed_fault_diagnosis_using_zonotopes.pdf) · [Dendukuri & Joseph](https://joseph.research.mcgill.ca/publications/Methodological/conddep_diag.pdf)
- **응용 근거와 불일치의 해석.** 외부 camera가 slip 예측에 도움이 된다는 실험과, 두 계측의 일치가 truth를 보장하지 않는다는 통계 결과는 endpoint와 독립 단위가 다르다. 88.03% grasp 판별률을 새로운 표면의 원인 판별률과 수치로 비교할 근거는 없다. [Li et al.](https://arxiv.org/pdf/1802.10153) · [Dendukuri & Joseph](https://joseph.research.mcgill.ca/publications/Methodological/conddep_diag.pdf)
- **검증이 남은 부분.** 실제로 sensor reset이 표면과 clock을 건드리는지, camera readout이 전자센서와 별개인지, 새로운 surface에서 camera·reset 비용을 회수할지는 원문들로 결정할 수 없다. 고속영상을 이용한 실측과 independent simulator 검증은 이 가정들을 시험하는 서로 다른 방법이다. [Rigi et al.](https://www.mdpi.com/1424-8220/18/2/333) · [Acosta et al.](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf)

### 8. 후속 연구 질문 열 가지

| 질문 | 다루는 공백과 의미 | 변수와 검증 방법 |
|---|---|---|
| 1. 실제 전자 reset은 latch를 얼마나 자주 지우는가? | 개입의 도달 범위가 원인 판별을 좌우한다. [Scott et al.](https://web.mit.edu/braatzgroup/input_design_for_guaranteed_fault_diagnosis_using_zonotopes.pdf) | 알려진 fault를 주입해 reset 전후 recovery·settling time을 잰다. |
| 2. Reset을 명령한 순간 표면도 미세하게 움직이는가? | Physical intervention과 electronic intervention의 혼동. [Acosta et al.](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf) | 무전원 sham reset, encoder·독립 광학 reference로 변위를 비교한다. |
| 3. Camera clock·mount의 공유가 추론에 주는 영향은? | Common-mode noise에서 반복 계측이 독립성을 잃는다. [Dendukuri & Joseph](https://joseph.research.mcgill.ca/publications/Methodological/conddep_diag.pdf) | 독립·공유 clock과 mount를 무작위 교차하고 cause accuracy를 잰다. |
| 4. 실제 impact duration에서 몇 fps와 어느 노출이 필요한가? | 빠른 방향 전환이 프레임 사이에서 사라질 수 있다. [Rigi et al.](https://www.mdpi.com/1424-8220/18/2/333) | 동일 촬영본을 exposure·sampling rate별로 downsample해 부호·누락을 평가한다. |
| 5. 어느 재질에서 camera missingness가 mismatch에 의존하는가? | 조명과 투과율이 contact 판독을 좌우한다. [Rigi et al.](https://www.mdpi.com/1424-8220/18/2/333) | 반사·불투명·연성 표면을 hold out하고 독립 gauge로 결측 trial까지 audit한다. |
| 6. 유료 8회 중 몇 회만 camera를 켜도 같은 risk를 달성하는가? | 센서 융합 정확도와 비용 회수는 별도 질문이다. [Li et al.](https://arxiv.org/pdf/1802.10153) | 매 session의 두 번째 물체만 촬영하는 대조 등과 사전 지정 utility를 비교한다. |
| 7. 다른 유형의 reset이 upstream common mode를 드러내는가? | 다른 가설을 분리하려면 다른 조작이 필요할 수 있다. [Scott et al.](https://web.mit.edu/braatzgroup/input_design_for_guaranteed_fault_diagnosis_using_zonotopes.pdf) | Sensor reset·amplifier reset·공통 calibration target 재측정을 같은 예산에서 비교한다. |
| 8. 표면의 실제 마모가 순서 무작위화 뒤에도 남는가? | 연속 contact의 독립성과 수치 solver의 한계. [Acosta et al.](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf) | 첫 물체의 충격 크기와 간격을 조절해 다음 object의 true deformation을 촬영한다. |
| 9. 물체별 연속 displacement와 binary sign의 정책 가치는 어떻게 다른가? | 서로 다른 modality의 예측 기여가 endpoint에 따라 달라진다. [Li et al.](https://arxiv.org/pdf/1802.10153) | 같은 영상에서 부호와 궤적을 추출해 고정 terminal과 동일 held-out session으로 평가한다. |
| 10. 두 원인이 함께 존재하며 강도가 연속일 때도 결론이 유지되는가? | 작은 후보 집합에서 얻은 분류율을 자유로운 latent mechanism으로 일반화하기 어렵다. [Dendukuri & Joseph](https://joseph.research.mcgill.ca/publications/Methodological/conddep_diag.pdf) | 알려지지 않은 mixture 비율로 새 material을 만들고 posterior calibration과 regret을 측정한다. |

### 9. 연구 심사

논문에서 가장 방어하기 쉬운 부분은 어떤 조작이 어느 측정 경로를 바꾸는지 명시하고, 동일하게 보이는 두 물리적 설명에 대해 **식별 가능한 관측·식별 불가능한 관측·실제 정책 손익**을 한 예산에서 함께 측정했다는 점이다. Active diagnosis와 외부 카메라 검증은 이미 연구됐다. 실제 센서 독립성, hardware reset 성공률 또는 새로운 contact dynamics에서의 보편적 성능은 이 문헌과 simulation으로 입증하지 못한다. 이 다섯 논문은 동일한 질문을 다루는 방법별 비교를 위해 선정했으며 체계적 전수조사는 아니다. [Scott et al.](https://web.mit.edu/braatzgroup/input_design_for_guaranteed_fault_diagnosis_using_zonotopes.pdf) · [Rigi et al.](https://www.mdpi.com/1424-8220/18/2/333) · [Acosta et al.](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf)
