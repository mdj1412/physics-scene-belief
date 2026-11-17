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
