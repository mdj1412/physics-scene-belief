## 문헌과 논문 기여의 범위

이번 검토의 질문은 “같은 네 쌍을 쓸 때, 마지막 reference를 impact 관측으로 교환할 이유가 있는가?”다. 아래 네 원문을 선택했다. 첫 연구는 정보의 가치를 정의하고, 다음 세 연구는 실제 충돌과 접촉을 어떻게 측정하고 검증했는지 보여준다. 이 범위의 검토로 분야 전체에서 최초라는 주장을 뒷받침할 수는 없다.

### 1. 원 논문 비교

| 원 논문 | 질문·방법 | 표본과 평가 단위 | 주요 결과·결론 | v53에 적용할 때의 한계 |
|---|---|---|---|---|
| Goel & DeGroot, *Comparison of Experiments and Information Measures* | 고정한 prior에서 서로 다른 관측 실험의 정보량을 어떻게 비교하는가? Expected Bayes risk로 실험 순서를 정의하고 분포족에 대해 조건을 유도했다. | 수리적 분석. 실물 표본과 센서 시간은 없다. | 어느 실험이 모든 decision problem에서 더 유익하다는 주장은 특정 loss에서의 승리보다 강하다. | v53은 하나의 예측 문제, 특정 prior, 두 summary에서의 교환 가치만 측정한다. [원문](https://doi.org/10.1214/aos/1176344790) |
| Pfrommer et al., *ContactNets* | 관측된 자세에서 discontinuous contact dynamics를 학습할 수 있는가? Signed distance와 contact-frame Jacobian을 학습해 rollout했다. | 10 cm acrylic cube, 목재 표면, 148 Hz 카메라 세 대. 750회 중 고품질 toss 570회. | Contact geometry를 구조화한 모델은 적은 training toss에서도 end-to-end 모델과 비교해 특히 회전 예측이 좋았다. | Camera가 추적한 rigid pose와 v53이 요구하는 국소 탄성 변형 부호는 관측량이 다르다. [원문](https://proceedings.mlr.press/v155/pfrommer21a/pfrommer21a.pdf) |
| Acosta et al., *Validating Robotics Simulators on Real-World Impacts* | 실물 trajectory를 기준으로 solver와 contact parameter의 적합도를 비교했다. | Cube toss 550개, Cassie jump 22개. Cube는 세 solver, Cassie는 두 solver를 비교했다. | 비탄성 충돌과 큰 움직임은 재현해도 restitution과 일부 edge/corner impact에서 오차가 남았다. Cassie trajectory는 마찰 계수를 식별할 slip을 충분히 만들지 않았다. | Solver를 정교하게 하거나 시행 수를 늘려도 필요한 contact regime이 빠질 수 있다. 앞선 cube 계열 자료와 수집 계보가 겹쳐 독립적인 재질 반복으로 세기 어렵다. [원문](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf) |
| Dong et al., *Maintaining Grasps within Slipping Bound by Monitoring Incipient Slip* | GelSlim의 displacement field와 rigid motion의 차이로 incipient slip을 검출했다. | 일상 물체 10개 각 24회, 총 240회. 수동 push/rotation, 사후 사람 판독. | 평균 검출 정확도 86.25%. 접촉 신호가 약한 두 물체에 false negative가 집중됐다. | Grasp slip 검출은 짧은 impact에서의 변형 부호 판독과 다르다. 이 정확도를 v53 sensor의 noise rate로 옮길 수 없다. [원문](https://arxiv.org/pdf/1810.13381) |

### 2–3. 증거의 흐름과 지식 지도

관측 예산 → 가능한 두 실험 → 각 실험이 만드는 posterior → 같은 downstream decision의 손실, 이 연결은 Goel & DeGroot의 risk 관점에서 나온다. V53에 적용하면 세 reference에 impact를 **더하는** 효과와 네 번째 reference를 **교체하는** 효과에 각각 다른 comparator가 필요하다. [Goel & DeGroot](https://doi.org/10.1214/aos/1176344790)

측정 가능한 위치·자세 → contact model → 미래 trajectory는 ContactNets가 실물에서 다룬 경로다. 이에 비해 v53의 경로는 weak displacement → impact 순간의 국소 변형 방향 → 다른 물체들의 stick probability다. 두 중간 변수가 실제로 관측되는지는 별도의 문제다. 힘을 가한 상태에서 surface motion을 직접 측정한 Dong 등의 결과와 재료·충돌 상태에 따른 solver 오차를 측정한 Acosta 등의 결과가 이 간극을 드러낸다. [ContactNets](https://proceedings.mlr.press/v155/pfrommer21a/pfrommer21a.pdf) · [Dong et al.](https://arxiv.org/pdf/1810.13381) · [Acosta et al.](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf)

### 4–7. 연구 공백, 상충해 보이는 결과와 증거 강도

- **강한 근거**. 구조화된 contact model로 같은 cube의 일부 trajectory를 잘 예측할 수 있고, 일부 탄성·모서리 충돌은 계속 어렵다. 두 결과는 평가 지표, 충돌 조건, 비교 모델이 다르다. 평균 회전 오차가 줄었다는 결과는 모든 contact regime의 국소 변형을 맞힌다는 뜻으로 확대하기 어렵다. [ContactNets](https://proceedings.mlr.press/v155/pfrommer21a/pfrommer21a.pdf) · [Acosta et al.](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf)
- **제한적인 근거**. 접촉 신호의 세기에 따라 방향 관련 측정의 활용도가 달라질 수 있다. GelSlim 실험은 물체·하중에 따른 검출 차이를 보였지만 저하 원인을 실험 모든 물체에서 하나로 식별하지는 않았다. V53의 weak-to-impact 조건에 대한 물리적 인과 효과는 독립 측정이 필요하다. [Dong et al.](https://arxiv.org/pdf/1810.13381)
- **근거 부족**. 네 논문은 같은 교체 후 session에서 weak/reference와 weak/impact를 맞교환하고 다른 128개 물체의 held-out Brier 분포를 비교하지 않는다. 특히 calibration 대상을 희생하면서 생긴 surface wear와 impact 방향 readout 오류의 공동분포는 남은 공백이다. [Dong et al.](https://arxiv.org/pdf/1810.13381) · [Acosta et al.](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf)

### 8. 판별 가능한 후속 연구 질문 열 가지

| 질문 | 공백과 중요한 이유 | 독립 변수와 검증 방법 | 근거 |
|---|---|---|---|
| 1. 서로 다른 센서가 같은 impact 방향을 읽는가? | 단일 센서 오류가 정답에 섞일 수 있다. | 고속 측면 영상과 tactile field의 blind annotation을 active 물체에서 비교한다. | [Dong](https://arxiv.org/pdf/1810.13381) |
| 2. 방향 판독이 유효한 최소 exposure는 얼마인가? | 빠른 충돌은 pose track 사이에서 끝날 수 있다. | 같은 고속 영상을 시간적으로 downsample하고 부호와 무응답률을 계산한다. | [ContactNets](https://proceedings.mlr.press/v155/pfrommer21a/pfrommer21a.pdf) |
| 3. Inactive 물체를 피하도록 네 번째 대상을 고를 수 있는가? | 변형이 없으면 직접 관측도 방향을 주지 않는다. | 기존 세 쌍의 정보와 구입 전 영상만 이용해 사전 선택하고 무작위 선택과 비교한다. | [Dong](https://arxiv.org/pdf/1810.13381) |
| 4. Reference 오류가 session 안에서 군집하는가? | 공동 오차는 128개 예측의 tail을 바꾼다. | 날짜·표면별 반복을 모아 independent와 hierarchical noise를 비교한다. | [Acosta](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf) |
| 5. Impact를 가한 이력이 뒤의 마찰을 바꾸는가? | 측정 자체가 이후의 contact law를 바꿀 수 있다. | 새 표면을 짝짓고 impact와 reference 순서를 무작위화한 뒤 wear·slip을 기록한다. | [Acosta](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf) |
| 6. Query에서 떨어진 위치에서도 방향을 옮길 수 있는가? | 하나의 위치가 전체 물체를 대표하지 않을 수 있다. | Probe offset과 query height를 교차하고 새 표면에서 Brier를 비교한다. | [ContactNets](https://proceedings.mlr.press/v155/pfrommer21a/pfrommer21a.pdf) |
| 7. 약한 negative evidence를 무응답으로 처리하면 tail이 줄어드는가? | 낮은 contrast에서 false sign의 비용이 크다. | Source에서 abstention을 고정하고 held-out object에서 coverage와 p90을 함께 보고한다. | [Dong](https://arxiv.org/pdf/1810.13381) |
| 8. Reference–impact prior의 폭을 실물에서도 고정할 수 있는가? | 고정 prior에서의 실험 순위가 다른 배포 혼합에서 달라질 수 있다. | Source surface로 prior를 정하고 untouched surface/date holdout에서 두 probe를 다시 짝짓는다. | [Goel & DeGroot](https://doi.org/10.1214/aos/1176344790) |
| 9. 실제 reset 시간으로도 교체가 이득인가? | 같은 횟수의 contact가 같은 시간이나 손상을 보장하지 않는다. | Reset 실패·교체 포함 wall time으로 예산을 맞추고 전체 cycle의 gain을 측정한다. | [Goel & DeGroot](https://doi.org/10.1214/aos/1176344790) |
| 10. 더 강한 contact decoder에도 효과가 남는가? | 단순 decoder의 representation error가 보정 이득을 만들 수 있다. | 구조화한 learned decoder와 원 decoder에 같은 calibration·split을 교차 적용한다. | [ContactNets](https://proceedings.mlr.press/v155/pfrommer21a/pfrommer21a.pdf) |

### 9. 심사 관점

Bayesian averaging 자체를 새로운 기법으로 제시하기는 어렵다. 이 실험의 기여 후보는 *동일한 두 번의 접촉으로 reference 또는 impact 중 하나만 관측할 때, 그 선택이 평균 및 나쁜 session의 손실을 어떻게 다르게 바꾸는가*라는 검증 가능한 질문이다. 외부 타당성을 판단하려면 서로 다른 날과 미사용 surface specimen, 독립적인 고속 impact 측정, 실제 reset 시간이 필요하다. 특히 cube pose 데이터만으로는 local deformation sign이나 동일 specimen의 weak/reference 반응을 복원할 수 없다. [Goel & DeGroot](https://doi.org/10.1214/aos/1176344790) · [ContactNets](https://proceedings.mlr.press/v155/pfrommer21a/pfrommer21a.pdf) · [Acosta et al.](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf)
