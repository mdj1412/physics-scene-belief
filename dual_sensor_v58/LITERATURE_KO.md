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
