## 어떤 접촉이 이후 충돌의 예측에 도움이 되는가

v36의 질문은 네 번째 접촉의 높이와 속도를 바꾸면, 이미 가진 방향 보정을 **어떤 환경에서 적용할지** 더 잘 결정할 수 있는가이다. 아래 표는 측정 동작, 추론 목표, 실제 task의 평가가 연결되는 여덟 연구를 목적 표집한 결과다. 각 논문의 원문에서 관측 대상과 평가 단위를 확인했다.

| 연구와 질문 | 방법과 표본 | 주요 결과와 저자의 결론 | v36에 남는 한계와 근거 |
|---|---|---|---|
| Fishel & Loeb (2012). 어떤 접촉 동작이 표면을 구분하는가? | BioTac으로 117개 texture를 탐색했다. Normal force와 sliding velocity 조합을 바꾸고 traction·roughness·fineness를 측정했다. Bayesian exploration이 다음 동작을 선택했다. | 이미 수집한 117개 후보의 texture identification은 중앙값 5회 탐색으로 95.4%에 도달했다. 비슷한 두 후보만 구별할 때는 99.6%였다. | 후보 집합과 평가 목표가 고정돼 있다. 처음 보는 contact law에서 다른 물체의 stick probability를 예측하려면 별도의 평가가 필요하다. [원문, Abstract·Results][F] |
| Tramsen et al. (2018). 비대칭 돌기에서 어느 방향에 friction이 더 큰가? | 인공 sawtooth와 substrate를 조합하여 양방향으로 당겼다. 상대 stiffness, aspect ratio, roughness를 바꾸고 pulling force를 기록했다. | 재료와 형상 조합에 따라 높은 friction을 보이는 방향이 반전됐다. 저자들은 접촉 면적과 mechanical interlocking의 상대적 역할로 설명했다. | 측정량은 pulling force다. 같은 재료의 impact에서 방향 효과의 크기, 또는 네 번의 접촉 뒤 Brier gain은 따로 검증해야 한다. [원문, Figs. 2–5][T] |
| Fazeli, Kolbert, Tedrake & Rodriguez (IJRR 2017). 운동과 외력을 어느 정도 알아야 contact dynamics를 식별하는가? | Planar rigid contact의 complementarity formulation에서 식별 가능한 parameter 조합을 유도했다. 실물 정사각형과 타원형 물체에 drop test 총 280회, 각 drop의 처음 세 bounce를 사용했다. | 알려진 외력이 없으면 inertial parameter와 contact force의 일부 비율이 식별된다. 타원의 mass/inertia 비율 추정치는 587 m⁻², 참값은 535 m⁻²였다. Noise가 커지면 추정 오차가 증가했다. | 실물 검증의 주대상은 inertia와 contact force다. Contact mode와 sensor noise가 식별 범위를 제한하는 선례로 읽을 수 있다. [원문, §§2·5.3–5.4·7][I] |
| Song & Boularias (RSS 2020). 몇 번의 push로 얻은 물성 추정이 다른 조작을 예측하는가? | Differentiable physics로 mass와 friction의 공간 분포를 맞췄다. 실물 5개, simulation 물체 8개를 사용했다. 실제 trajectory를 절반씩 나누는 25개 split으로 평가했다. | Hammer를 table edge로 보내 잡는 실험에서 friction 상한을 둔 방법은 16/16회, uniform mass는 12/16회, friction 상한을 없앤 방법은 13/16회 성공했다. 탁자 위 운동을 잘 맞춘 model이 상판 지지가 사라지면 실패할 수 있었다. | 실물 mass와 friction 참값을 알 수 없어 held-out pose와 task success로 평가했다. 한 hammer의 16회 시도는 16종의 독립 물체가 아니다. [원문, 실험·Final Remark][S] |
| Le et al. (2021 원고). 일부 표면의 접촉으로 다른 위치의 friction을 추정할 수 있는가? | Panda와 force–torque sensor, Kinect point cloud를 사용했다. 실물 15개 중 4개는 단일 재질이고 나머지는 2~4개 재질이었다. 국소 haptic 측정과 visual prior를 결합했다. | 여러 재질에서 위치별 friction 차이를 복원했다. 컵 두 종류를 사용한 grasp case study에서 추정된 friction이 높은 위치로 grasp candidates를 좁혔다. | 표면 내 visual similarity를 통해 측정값을 전달한다. 같은 재질에서 속도가 달라졌을 때의 friction law와 binary impact outcome은 별개의 변수다. [원문, §§III–V][L] |
| Attia, Alexanderian & Saibaba (2018). 예측할 quantity of interest가 달라지면 sensor 위치도 달라지는가? | Linear Gaussian inverse problem에서 목표 예측의 posterior covariance를 최적화했다. Advection–diffusion simulation에는 후보 위치 22개가 있다. Fig. 5의 thresholded design은 5개 위치를 사용한다. | 예측하려는 contaminant concentration의 범위와 차원이 달라지면 같은 후보 중에서도 유리한 sensor 배치가 달라졌다. | 선형 모델과 알려진 noise에서 계산한 design이다. 같은 표면에서 stick/slip transition이 만드는 비선형성과 model mismatch는 별도로 다뤄야 한다. [원문, §5][A] |
| Zhong et al. (SIAM/ASA JUQ, 2026). 비선형 predictive quantity에 적합한 실험을 어떻게 선택하는가? | Nested Monte Carlo, MCMC, kernel density estimation, Bayesian optimization을 결합했다. Convection–diffusion 예제는 source 위치 2,000개로 surrogate를 만들고 미래 농도를 예측할 네 지점을 바꿨다. | Parameter와 미래 농도에 대한 expected information gain을 각각 최적화하자 sensor 위치가 달라졌다. 일부 조건에서는 유체가 이동하는 방향을 고려해 목표 지점보다 상류에서 관측하는 가치가 나타났다. | 정확히 같은 위치나 조건이 항상 최적인 것은 실험으로 보장되지 않는다. 이 연구의 목적함수는 predictive information gain이고 v36의 목적함수는 downstream Brier loss다. [원문, §§4.3–4.4][Z] |
| Mozannar & Sontag (ICML 2020). 모델과 다른 예측자 중 누구에게 맡길지 어떤 loss로 학습하는가? | Cost-sensitive surrogate로 classifier와 rejector를 학습했다. CIFAR10H에서는 영상당 50개 annotator 응답에서 simulated expert를 만들고, 주석 있는 자료를 반으로 나눈 split을 10회 반복했다. | Table 2에서 imputation을 쓴 모델의 system accuracy는 96.29%, confidence baseline은 95.09%였다. Coverage는 각각 51.67%, 79.48%였다. | 두 방법이 다른 비율로 결정을 넘겼다. 이 논문의 classification accuracy와 v36의 연속 확률 Brier는 목적함수가 다르다. [원문, §6.3·Table 2][M] |

### 관측에서 결정까지의 연구 흐름과 증거

| 관계 | 근거 수준과 아직 식별되지 않은 부분 |
|---|---|
| 접촉 조건 → sensor response → 가능한 물성 | Force와 velocity를 바꿔 texture 판별력이 달라진 실물 측정, 재료 조합에 따른 friction 방향 반전, 여러 재질에서 위치별 friction 차이가 관측됐다. 재질·측정 방식·정답은 세 연구마다 다르다. 약한 impact anisotropy의 식별 한계에는 직접적인 수치가 없다. [Fishel & Loeb][F] · [Tramsen et al.][T] · [Le et al.][L] |
| 물성에 대한 적합도 → 새 intervention의 예측 | Fazeli et al.이 관측 가능한 contact dynamics의 조합을 수식과 실물 낙하로 검증했다. 이 연구를 인용하는 Song & Boularias의 hammer 실험에서 기존 동작과 잘 맞았던 물성이 변경된 지지 조건에 옮겨가지 못했다. Attia et al.과 Zhong et al.은 목표 예측을 직접 design에 넣으면 최적 측정 위치가 달라지는 수치 예를 보였다. 물리적 실패 사례와 design 원리는 서로를 보완한다. 네 번째 probe를 어느 높이에 둘지는 해당 contact model로 검증해야 한다. [Fazeli et al.][I] · [Song & Boularias][S] · [Attia et al.][A] · [Zhong et al.][Z] |
| Paid evidence → 보정 여부 → held-out loss | Mozannar & Sontag에서는 예측할 주체를 고르는 학습 목표가 system performance에 영향을 줬다. v36에 대응하는 질문은 관측된 접촉으로 두 decoder의 상대적인 downstream loss를 예측할 수 있는가이다. 논문에 제시된 수치만으로 새 표면의 correction gain을 산출할 수 없다. [Mozannar & Sontag][M] |

**서로 다르게 보이는 결과의 원인.** Fishel & Loeb의 높은 성공률은 미리 경험한 재질 중 정체를 맞히는 과제에서 나왔다. Song & Boularias의 일부 실패는 같은 물체에서 지지 조건까지 바꾼 뒤 나타났다. 표면을 구별하는 accuracy, 같은 조건의 pose error, 변경된 intervention의 task success는 서로 다른 측정량이다. Attia et al.과 Zhong et al.의 결과를 함께 보면 task와 관측 사이의 전달 관계를 알려주는 측정이 유리하다. 기하학적 거리가 가까운 동작에 대한 보편적인 우위까지 입증하는 자료는 없다. [Fishel & Loeb][F] · [Song & Boularias][S] · [Attia et al.][A] · [Zhong et al.][Z]

### 무엇을 측정해야 마지막 probe에 기여를 돌릴 수 있는가

아래 판정식은 v36의 paired 실험에 대한 분석 제안이다. 환경 e에서 같은 관측을 사용했을 때 보정한 Brier와 원래 Brier의 차이를 Δₑ, 해당 환경에서 보정을 켰으면 Gₑ=1이라고 두면 gate의 평균 변화는 E[GΔ]다. 같은 비율로 무작위로 켜는 policy의 기대 변화는 E[G]E[Δ]다. 두 값의 차이 Cov(G,Δ)가 음수이면 보정이 상대적으로 유리한 환경에 보정을 더 자주 적용했다는 뜻이다. 표면별 평균이 다른 경우 pooled covariance에는 표면군 구분의 효과가 포함된다. 각 군 안에서 계산하면 같은 surface regime 내의 구분 능력을 점검할 수 있다.

| 남은 공백 | 결과를 해석하거나 확장할 때 필요한 비교 |
|---|---|
| 보정을 덜 쓰는 것만으로 weak/isotropic 평균 loss가 내려갈 수 있다. Coverage와 accuracy가 함께 달라졌던 deferral 선례가 있다. [Mozannar & Sontag][M] | 항상 적용·항상 생략과 함께, 같은 적용률을 가진 randomized gate와 실제 gate의 차이를 보고한다. 모든 환경에서 correction이 꺼졌다면 약한 표면의 손실을 피한 효과는 확인할 수 있고 환경별 선택 효과는 0이다. |
| 네 번째 행동과 terminal decoder를 동시에 바꾸면 두 효과가 합쳐진다. 같은 물성 추정도 후속 과제에 따라 효용이 달랐다. [Song & Boularias][S] · [Zhong et al.][Z] | 같은 첫 세 접촉을 공유하고, 네 번째 접촉의 입력을 가린 gate 및 두 행동에서 같은 방식으로 학습한 gate를 함께 비교한다. 각 행동에서 강제 on·off를 같은 관측으로 채점한다. |
| Source에서 많이 보던 표면에 유리한 gate가 target mixture에서도 유리할 수 있다. 재료 조합에 따라 방향이 뒤집힐 수 있다. [Tramsen et al.][T] | Strong·weak·isotropic에서 보정 적용률과 paired Brier를 각각 계산한다. Source와 달라진 환경 비율에서의 pooled loss는 해당 비율을 명시한다. |
| Query와 가까운 동작이 friction에 민감한 동작인지는 contact state에 따라 달라진다. 선행연구에서는 force·속도 조합이나 upstream 위치가 측정의 가치를 바꿨다. [Fishel & Loeb][F] · [Zhong et al.][Z] | Friction cone 제약을 적용하면 sticking 상태에서 tangential response가 마찰 허용 범위 안에 있으면 여러 friction coefficient가 같은 측정을 낼 수 있다. Probe가 slip을 유발한 비율, plausible noise 변화, 별개의 deployment 물체에서의 gain을 함께 살핀다. [Fazeli et al., §2][I] |

### 후속 연구 질문 10개

| 질문 | 공백·중요성 | 조작할 변수와 검증 방법 |
|---|---|---|
| 1. 어느 높이와 속도에서 약한 방향성이 드러나는가? | 예측할 위치와 정보가 많은 측정 위치가 다를 수 있다. [Zhong et al.][Z] | 높이와 속도를 독립 교차하고 held-out query를 고정해 Brier와 강제 on의 상대 이득을 계산한다. |
| 2. 네 번째 접촉이 stick이면 무엇을 알 수 있는가? | 접촉 동작별로 잘 드러나는 물성이 다르고, 접촉 제약은 식별 가능한 물성 조합을 제한한다. [Fishel & Loeb][F] · [Fazeli et al.][I] | Normal impulse와 tangential speed를 독립 조작하고 probe stick/slip에 따른 측정 분포와 conditional Brier를 비교한다. |
| 3. Directional amplitude가 어느 범위에서 gate가 바뀌는가? | 재료 조합에 따라 방향과 크기가 변한다. [Tramsen et al.][T] | 연속 amplitude sweep에 방향이 없는 조건을 포함하고 적용률 및 oracle 대비 regret을 추정한다. |
| 4. Target에 isotropic 표면이 많아져도 gate가 probe의 정보를 활용하는가? | 예측자에게 맡기는 비율이 달라지면 평균 성능도 달라진다. [Mozannar & Sontag][M] | 같은 결과에 target prevalence를 다시 가중하고 같은 coverage의 무작위 대조와 비교한다. |
| 5. 같은 그림으로 보이는 두 접촉면에서 gate가 유지되는가? | Visuo-haptic friction 예측은 visual similarity와 물성의 관계에 의존한다. [Le et al.][L] | 표면 외관을 맞추고 force response만 바꾼 object–surface pair를 독립 holdout한다. |
| 6. Geometry와 friction이 서로 보상하면 방향 보정이 유리한가? | 서로 다른 물성이 같은 운동을 설명한 실물 사례가 있다. [Song & Boularias][S] | Curvature와 directional amplitude를 독립 조작하고 관측이 유사한 환경 쌍에서 다음 충돌의 확률을 평가한다. |
| 7. 단일 query에 맞춘 probe가 다른 query에도 유용한가? | Quantity of interest가 달라질 때 sensor design이 달라졌다. [Attia et al.][A] · [Zhong et al.][Z] | Calibration 후 query 높이·속도 범위를 넓힌 holdout을 사용해 query별 Brier와 worst-group loss를 계산한다. |
| 8. 약한 방향성을 binary gate와 연속 shrinkage 중 어느 쪽이 잘 다루는가? | Classification에서의 최적 routing 규칙은 연속 확률 예측의 손실에 그대로 대응하지 않는다. [Mozannar & Sontag][M] | Source에서 hard gate와 연속 interpolation weight를 따로 고정하고 동일 target에서 paired Brier를 비교한다. |
| 9. 같은 네 번으로 friction coefficient의 연속 posterior를 추정하는 방법과 차이가 있는가? | 실물에서 연속 물성 추정이 후속 manipulation에 쓰였다. [Le et al.][L] · [Song & Boularias][S] | Isotropic mass를 포함한 directional physics model과 frozen correction에 동일한 관측·계산 예산을 부여한다. |
| 10. 실제 robot에서 probe 교체에 드는 비용까지 포함하면 어느 방법이 유리한가? | 다른 motion의 force·속도와 과제 성공을 실물에서 측정한 선례가 있다. [Fishel & Loeb][F] · [Song & Boularias][S] | 물체와 surface session을 동시 holdout하고 reset·motion·sensor acquisition 포함 wall time, GPU p90, binary impact Brier를 기록한다. |

논문에서는 독립 deployment 물체의 event label, 네 번째 접촉을 가린 대조, 동일한 적용률의 대조가 각기 다른 주장을 뒷받침한다. 첫 번째는 최종 확률의 예측 성능, 두 번째는 추가 측정의 가치, 세 번째는 환경에 맞춰 보정을 선택한 가치를 검증한다. 새 동작이 잘 작동하는 물리적 이유를 주장하려면 별도 측정한 anisotropy나 contact state와 sensor response 사이의 대응도 필요하다. 실물 자료가 외부 타당성을 지지하는 범위는 각 문헌이 측정한 동작과 정답까지다. [Tramsen et al.][T] · [Le et al.][L] · [Song & Boularias][S]

[F]: https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2012.00004/full
[T]: https://findresearcher.sdu.dk/ws/files/151776983/rsif.2017.0629.pdf
[S]: https://www.roboticsproceedings.org/rss16/p099.pdf
[L]: https://arxiv.org/pdf/2010.08277
[A]: https://arxiv.org/pdf/1802.06517
[Z]: https://arxiv.org/pdf/2403.18072
[M]: https://proceedings.mlr.press/v119/mozannar20b/mozannar20b.pdf
[I]: https://groups.csail.mit.edu/robotics-center/public_papers/Fazeli17a.pdf
