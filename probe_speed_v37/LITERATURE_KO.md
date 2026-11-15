## 마지막 probe의 속도를 바꾸면 얻는 정보

속도 변화에 대한 실제 마찰 측정, 충돌에서의 parameter identifiability, 관측 선택의 목적함수, 확률 보정을 다룬 원문 7편을 검토했다. 결과의 단위는 각 연구가 직접 측정한 값으로 유지했다.

| 원문과 연구 질문 | 방법 및 표본 | 결과와 결론 | v37 적용 한계 |
|---|---|---|---|
| Yu et al. (IROS 2016). Pushing의 friction을 속도·방향에 무관한 상수로 놓을 수 있는가? | 로봇으로 steel object의 pose와 force를 기록했다. ABS, Delrin, plywood, polyurethane(PU) 네 지지면에서 speed scan을 10~500 mm/s의 10개 속도로 각 10회씩 진행했다. | Dynamic coefficient of friction은 ABS, Delrin, plywood에서는 속도에 따라 조금 변했고 PU에서는 고속에서 약 1.0까지 증가했다. 연구진은 넓은 속도 범위에서 재료별 검증이 필요하다고 결론지었다. | 지지면 위 평면 이동과 v37의 짧은 impact는 서로 다른 실험이다. 같은 물체의 반복 scan은 독립 재료 수에 포함되지 않는다. [원문, Table II·§V·Fig. 5][Y] |
| Tramsen et al. (J. R. Soc. Interface 2018). Sawtooth의 방향으로 높은 friction 방향을 예측할 수 있는가? | Aspect ratio, stiffness와 substrate roughness를 바꾼 96개 sample–substrate 조합에서 양방향 pulling force를 비교했다. | Geometry가 같아도 조합에 따라 높은 friction 방향이 바뀌었다. 저자들은 실제 접촉 면적과 mechanical interlocking의 상대적 역할로 설명했다. | 특정 방향의 quasi-static pulling coefficient로 다른 속도의 impact에서 stick할 확률을 정하기에는 추가 측정이 필요하다. [원문, §§2–4][T] |
| Sakuma et al. (PRL 2026). 같은 물질에서 sliding velocity에 대한 friction의 민감도가 유지되는가? | Muscovite mica를 22~200 °C, normal stress 100 MPa에서 실물 shear experiment로 측정했다. 네 sliding velocity는 0.01, 0.1, 1.0, 3.0 μm/s였다. | 22 °C에서는 coefficient가 log sliding velocity에 의존했고, 온도가 오르면서 의존성이 약해져 200 °C에서는 측정 범위에서 거의 일정했다. | 고압·고온 mica에서의 기전을 일상 접촉으로 옮기려면 별도 측정이 필요하다. Rate sensitivity와 방향에 따른 차이는 독립 축이다. [원문][M] |
| Fazeli et al. (공개 원고 2017). Planar rigid-body impact로 friction parameter를 식별하고 실제 impulse를 재현할 수 있는가? | 실제 drop 2,000회 중 tracking 등 품질 검사를 통과한 1,718건을 사용했다. 6개 모델에 friction과 restitution을 맞추고 허용되는 impulse의 범위를 비교했다. | 고정 restitution에서 coefficient가 sticking threshold를 넘으면 같은 impulse를 내 해당 사건에서 coefficient를 구별하기 어렵다. 실제 impulse가 각 모델의 예측 범위 안에 든 비율은 35.5~45.8%였다. | 이 비율은 모델이 표현할 수 있는 impulse의 범위를 나타낸다. 실제 slip 중에도 rigid-body family가 설명하지 못하는 충돌이 생길 수 있다. [원문, §3 식 (9)·Table 2][F] |
| Arslan Waltersson & Karayiannidis (IJRR 2025). 물체를 집어 올릴 때 slip을 일으켜 후속 조작에 쓸 물성을 추정할 수 있는가? | UR10에 parallel gripper를 장착하고 두 손가락에서 각각 force와 sliding velocity를 측정했다. 접촉 추정 실험은 wood, 안경집, plastic, cardboard 네 물체에 각각 10회, 5 N으로 수행했다. | Linear exploration으로 static·Coulomb·viscous friction을, 이어지는 rotational exploration으로 contact radius를 추정했다. 두 단계는 2초였고 cardboard와 plastic이 각각 가장 높고 낮은 friction을 보였다. | 동일 접촉의 짧은 time series에서 여러 값을 사용한다. v37처럼 네 번의 noisy velocity만으로 다른 물체의 충돌 확률을 맞히는 문제로 옮기려면 sensor와 budget을 다시 맞춰야 한다. [원문, Algorithm 1·Table 9][W] |
| Zhong et al. (공개 원고 v2 2025). 가장 많은 parameter 정보를 주는 sensor가 미래 quantity of interest에도 최적인가? | Nested Monte Carlo, MCMC, density estimation, Bayesian optimization으로 predictive information gain을 최적화했다. 합성 test problem과 convection–diffusion source inversion에서 1~3개 sensor 배치를 비교했다. | 미래 농도를 예측할 위치와 예측량이 달라지면 높은 utility의 sensor 위치가 달라졌다. Parameter와 그 nonlinear function에 대한 information gain의 관계도 도출했다. | 목적함수는 predictive distribution의 information gain이다. 제한된 선택지만 쓰는 hard gate의 Brier 개선은 별도로 측정해야 한다. [원문, §3·Figs. 14–19][Z] |
| Ranjan & Gneiting (JRSS B 2010). 확률 예측을 고정 비율로 결합하면 calibration이 유지되는가? | Linear opinion pool의 성질을 증명했다. Beta-transformed linear pool을 가상 예와 미국 본토 29개 도시의 강수 확률 예측으로 검토했다. | 서로 다른 calibrated forecasts의 양의 가중 평균은 결합 뒤 calibration을 잃을 수 있었다. 결합 후 재보정 방법을 제시하고 Brier와 구성항을 평가했다. | v37의 0.325는 log-odds correction 크기에 곱한다. Probability를 선형 결합한 이 논문과 수식이 다르다. 같은 holdout과 scoring rule에서 보정들을 대조하는 방법론이 연결된다. [원문, §§2–4][R] |

### 연구 흐름과 달라 보이는 결과

| 연구 사이의 연결 | 확립된 근거와 불확실성 |
|---|---|
| Commanded speed → contact state → 관측되는 velocity | 실제 impact와 rigid model에서 stick 중 일부 friction 값은 동일 impulse를 만든다. 이 구간을 떠나는 것이 식별에 필요한 경우가 있다. 서로 다른 두 slip response가 sensor noise에 비해 충분히 멀어지는지는 별도 문제다. [Fazeli et al.][F] |
| Sliding speed × 표면 조건 → friction response | PU에서는 큰 속도 의존성, 세 다른 지지면에서는 작은 의존성이 보고됐다. Mica는 같은 재료에서도 온도에 따라 의존성이 달라졌다. 재료·normal stress·온도·동작의 차이를 고려하면 보편 rate law를 정할 수 없다. [Yu et al.][Y] · [Sakuma et al.][M] |
| 표면 방향 → query 결과 → correction 선택 | 양방향 마찰 차이의 부호는 물체·표면 조합에 따라 바뀐다. 어떤 물성이 관측에 드러나도 query에 맞춘 utility는 따로 평가해야 한다. Weak와 null에서 같은 결정을 내리는 편이 손실상 최적일 가능성은 v37에서 시험하는 가설이다. [Tramsen et al.][T] · [Zhong et al.][Z] |
| Benefit score → binary gate 또는 shrinkage → held-out probability | Brier score는 확률 오차를 평가한다. 높은 material AUROC만으로 correction 크기의 적절성을 판단할 수 없다. 0.325의 우위 여부를 직접 다룬 증거는 이 7편에 없다. [Ranjan & Gneiting][R] · [Zhong et al.][Z] |

### 결과에서 확인할 주장과 공백

1. **관측성.** 같은 calibration 물체의 latent state와 sensor noise를 대응시켜 speed별 slip rate, weak–local-null AUROC, 두 regime의 velocity separation을 비교한다. Trial과 sensor draw를 독립 surface로 세면 interval이 지나치게 좁아진다. [Fazeli et al.][F] · [Yu et al.][Y]
2. **기전.** Speed를 올리면 slip 외에도 tangential demand, normal impulse와 velocity-dependent coefficient가 함께 달라진다. 같은 speed contrast로 식별할 수 있는 양은 속도의 총효과다. Slip만의 매개효과를 주장하려면 law를 고정한 반사실이나 독립적인 조작이 필요하다. [Yu et al.][Y] · [Sakuma et al.][M] · [Fazeli et al.][F]
3. **예측 이득.** 같은 speed에서 같은 base probability와 direction proposal을 사용하는 gate와 shrinkage를 비교한다. Speed 변경이 gate의 상대 이득까지 높였다는 주장은 두 손실 차이가 속도에 따라 얼마나 바뀌었는지로 평가한다. 결과에서 유리한 속도를 고른 경우 새로운 test panel이 필요하다. [Zhong et al.][Z] · [Ranjan & Gneiting][R]
4. **논문 범위.** Sticking 중 비식별성에는 선례가 있다. 어느 속도까지 weak/null 식별이 증가하고 그중 어느 부분이 제한된 budget에서 후속 충돌의 Brier를 낮추는지를 보여주는 실험이 기여 후보가 된다. 실물에서 시험하려면 probe의 outgoing motion과 다른 물체의 독립 query outcome을 짝지어야 한다. [Fazeli et al.][F] · [Tramsen et al.][T] · [Zhong et al.][Z]

### 문헌의 공백에서 나온 연구 질문 10개

| 질문 | 공백과 중요성 | 변수와 검증 설계 |
|---|---|---|
| 1. 같은 slip rate에서도 weak/null 식별이 달라지는가? | Stick 구간 밖이라는 사실만으로 impulse 차이의 크기는 정해지지 않는다. [Fazeli et al.][F] | 두 속도에서 모두 미끄러지는 동일 물체를 짝지어 velocity separation과 AUROC를 평가하고 하위집단 추정의 범위를 명시한다. |
| 2. Velocity-dependent law와 contact state 변화 중 어느 쪽이 정보를 증가시키는가? | 재료별 rate sensitivity가 다르다. [Yu et al.][Y] | Coefficient의 속도 항을 기준 속도 값으로 고정하고 incoming speed만 올린 조건을 원래 law와 짝짓는다. |
| 3. 높은 속도에서 sensor clipping이 증가하면 순위가 뒤집히는가? | 실물 impact와 측정·모델의 예측 범위에는 제한이 있다. [Fazeli et al.][F] | 장비의 속도별 noise covariance·saturation을 측정하고 동일 예산에서 AUROC와 Brier를 다시 계산한다. |
| 4. 새로운 표면 패치에서도 네 번의 읽기를 재사용할 수 있는가? | 실제 지지면에서 friction의 공간적 변동이 관찰됐다. [Yu et al.][Y] | Patch 거리와 probe–query 지연을 교차하고 object·surface session을 함께 hold out한다. |
| 5. 방향과 함께 재료별 속도 의존성도 학습하면 query 예측이 좋아지는가? | 양방향 마찰 부호와 속도 민감성은 서로 다른 조건에 의존한다. [Tramsen et al.][T] · [Sakuma et al.][M] | 방향 부호·속도 기울기를 독립 생성하고 joint posterior와 frozen cue를 비교한다. |
| 6. 네 번째 probe의 고속 방향에 반대되는 query에도 효과가 있는가? | 방향을 바꾸면 anisotropy 부호의 의미가 달라진다. [Tramsen et al.][T] | Calibration 방향과 deployment query 방향을 독립 배정하고 signed 및 orientation-invariant decoder를 비교한다. |
| 7. Material AUROC 최적 speed와 후속 Brier 최적 speed는 언제 다른가? | Parameter 정보와 미래 quantity of interest 정보는 서로 다를 수 있다. [Zhong et al.][Z] | Training environments에서 두 기준으로 각자 하나의 speed를 선택·고정하고 새 panel에서 paired regret을 측정한다. |
| 8. Hard gate와 continuous shrinkage의 차이는 어느 예측에서 발생하는가? | 확률 결합은 값의 척도와 calibration에 의존한다. [Ranjan & Gneiting][R] | 같은 source labels로 multiplier를 학습하고 calibration, resolution 및 held-out Brier를 boundary별로 측정한다. |
| 9. 어떤 물체를 골라 접촉할지까지 허용하면 추가 이득이 남는가? | 관측 위치의 가치가 예측 대상에 따라 달라진다. [Zhong et al.][Z] | Factual geometry만으로 informative fourth object를 사전 선택하고 무작위 물체에 같은 속도로 접촉한 경우와 비교한다. |
| 10. 실물에서 높은 속도의 이득이 운동·sensor reset 비용을 감수할 만큼 큰가? | 실물 benchmark에서는 force·pose 기록과 반복 reset이 필요하다. [Yu et al.][Y] · [Arslan Waltersson & Karayiannidis][W] | 이전 gripper 실험의 2초 exploration을 참고해 physical cycle time, 에너지, GPU p90과 Brier를 각각 측정하고 동일 장비의 Pareto frontier를 추정한다. |

[Y]: https://arxiv.org/pdf/1604.04038
[T]: https://findresearcher.sdu.dk/ws/files/151776983/rsif.2017.0629.pdf
[M]: https://journals.aps.org/prl/pdf/10.1103/y3jy-nqcd
[F]: https://arxiv.org/pdf/1710.04979
[Z]: https://arxiv.org/pdf/2403.18072
[R]: https://academic.oup.com/jrsssb/article/72/1/71/7076442

[W]: https://research.chalmers.se/publication/549631/file/549631_Fulltext.pdf
