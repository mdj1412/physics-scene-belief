## 선행연구에서 무엇을 가져왔는가

아래 비교는 접촉 역학, 관측 불변성, 판단의 외부 검증을 다룬 원문을 대상으로 한다. v40의 성능 수치는 자체 실험에서 산출했다. 문헌에서 다른 hardware나 평가 지표로 측정한 값을 v40의 성능 근거로 사용하지 않았다.

| 원문과 연구 질문 | 방법·표본 | 결과와 결론 | v40에 적용할 때의 한계 |
|---|---|---|---|
| Yu et al. 실제 마찰은 재료·위치·속도·방향에 따라 얼마나 변하는가? | 로봇의 planar pushing에서 재료, 형상, 접촉 위치, 방향, 속도, 가속도를 바꿨다. 네 재료의 마찰을 따로 측정했다. | 세 재료의 속도 변화는 작았고 polyurethane은 고속에서 큰 변화와 방향 차이를 보였다. | 실제 마찰 이질성의 근거다. 새 속도에서 방향성이 반전되는 v40의 crossover 식을 실측한 자료는 아니다. [원문](https://arxiv.org/pdf/1604.04038) |
| Datta et al. 물체를 미는 방향과 속도에 따라 friction response가 함께 달라지는가? | 약 57° 기울어진 titania nanorod에서 force microscopy로 네 속도와 두 배치의 마찰을 측정했다. | 속도가 커질 때 마찰이 감소하는 정도가 측정 축에 따라 달랐다. | 특정 microscale sliding 재료의 실측이다. v40 crossover의 기준 속도나 충돌에서의 방향 반전은 별도 가정이다. [원문](https://arxiv.org/abs/2602.13343) |
| Fazeli et al. 실제 충돌에서 모델 parameter를 식별할 수 있는가? | 로봇이 떨어뜨린 planar object의 2,000회 drop 중 품질 검사를 통과한 1,718회에 6개 rigid-body model을 비교했다. | Impulse의 표현 범위와 parameter 식별에 제약이 있었고, sticking impact에서는 friction에 하한만 주어졌다. | 같은 calibration cue가 다른 query로 전달되는 가설을 시험할 근거가 된다. 실물 결과는 이번 analytic law의 수치를 뒷받침하지 않는다. [원문](https://arxiv.org/pdf/1710.04979) |
| Zhong et al., ActivePusher. 적은 실행으로 pushing dynamics와 planning을 개선할 수 있는가? | Residual physics와 uncertainty로 실험과 계획을 선택했다. Simulation 4종, 실물 UR10 2종의 물체에서 skill learning을 평가하고 물체당 독립 test sample 1,000개를 사용했다. | 적극적으로 고른 학습 관측과 residual model로 data efficiency 및 planning success가 개선됐다. | 접촉 후 다른 물체의 stick 확률을 네 번의 측정으로 맞히는 비교와는 표본·목표가 다르다. [원문](https://arxiv.org/pdf/2506.04646) |
| Simard et al., Tangent Prop. 알려진 입력 변형에 둔감하게 만들면 일반화가 좋아지는가? | 문자 영상의 작은 이동·회전 등 알려진 변환을 tangent vector로 표현해 학습에 반영했다. | 입력을 어느 방향으로 바꿀 때 출력이 유지되어야 하는지를 지정할 수 있음을 보였다. | 여기서 tangent vector는 **영상 변환 방향**이다. v40의 물리적 local tangent와 다르다. Known nuisance direction을 제거한다는 아이디어 자체의 독창성은 낮다. [원문](https://proceedings.neurips.cc/paper/1991/file/65658fde58ab3c2b6e5132a39fae7cb9-Paper.pdf) |
| Koh et al., WILDS. 기존 성능이 자연적인 distribution shift에서도 유지되는가? | 병원·지역·시기 등을 달리한 10개 dataset에서 평가했다. | 표준 학습은 분포 밖에서 성능이 떨어졌고 여러 대응 방법에도 차이가 남았다. | 자연 집단의 넓은 대표성이 강점이다. 같은 물체에서 nuisance만 바꾸는 효과는 v40의 paired simulator가 더 직접적으로 측정한다. [원문](https://proceedings.mlr.press/v139/koh21a.html) |
| Yao et al. 각 조건에서 다른 predictive model이 유리할 때 어떻게 결합하는가? | Bayesian hierarchical stacking에서 input-dependent weight를 partial pooling하고 이론 및 여러 응용 자료로 검증했다. | 예측 성능의 이질성을 반영하는 weight를 추정할 수 있다. | 새 물리 law에서 어느 보정량이 이길지 선험적으로 알 수 있다는 정리는 아니다. v40은 동결한 weight의 transfer를 직접 채점한다. [원문](https://arxiv.org/abs/2101.08954) |
| Brier. 확률로 표현된 예보를 어떻게 검증하는가? | 기상 예보의 평가 문제에서 각 사건의 예측 확률과 실제 발생을 제곱 오차로 비교했다. 방법론 논문으로 독립 물체 표본 수는 적용되지 않는다. | 관측과 예측의 거리를 정량화하는 scoring rule을 제시했다. | 본 보고서는 이진 사건의 양성 class 한 항 (p−y)²를 사용한다. 예측 개선을 작업 성공률로 변환하려면 별도 행동과 비용을 정해야 한다. [원문](https://journals.ametsoc.org/view/journals/mwre/78/1/1520-0493_1950_078_0001_vofeit_2_0_co_2.xml) |
| Rossa et al., Action-BED. Task loss를 줄이는 관측을 학습할 수 있는가? | Joint model에서 expected future loss를 사용했다. Source location 예제는 30단계 정책을 2,048개 rollout으로 평가했다. | 측정 목적을 downstream loss에 맞춘 sequential design을 구성하고 검증했다. | v40에서 두 방법은 같은 probe를 구입한다. 결과를 probe 자체의 가치나 최적 실험 횟수로 해석할 수 없다. [원문](https://arxiv.org/html/2606.23662v2) |

**증거의 연결.** 실물 자료에는 접촉 조건에 따라 마찰이 바뀌고 stick 상태에서 동역학을 유일하게 알아내기 어려운 사례가 있다. 이 두 관찰은 단일 contact model의 transfer를 검증할 이유를 제공한다. 입력 변형에 대한 불변성과 조건별 weight 결합은 존재하는 방법이다. v40의 open question은 주변 contour가 유효한 stick 정보를 얼마나 제공하는지, 또 source에서 추정한 보정의 순위를 law가 달라져도 target label 없이 예고할 수 있는지다. [Yu et al.](https://arxiv.org/pdf/1604.04038) · [Fazeli et al.](https://arxiv.org/pdf/1710.04979) · [Simard et al.](https://proceedings.neurips.cc/paper/1991/file/65658fde58ab3c2b6e5132a39fae7cb9-Paper.pdf) · [Yao et al.](https://arxiv.org/abs/2101.08954)

**결과가 달라 보이는 이유.** 같은 실물 연구에서도 재료별 speed dependence가 다르다. ActivePusher가 보고한 planning 개선과 rigid impact 연구가 보고한 friction 식별 한계도 논리적으로 양립한다. 전자는 residual dynamics와 action sequence의 성공을, 후자는 충돌 하나가 parameter를 얼마나 제약하는지를 측정했다. 실물에 대한 근거는 이 범위에 있고, v40의 analytic law·고정 높이·한 방향의 시각 변형을 넘어서는 증거는 부족하다. [실제 재료별 측정](https://arxiv.org/pdf/1604.04038) · [로봇 planning](https://arxiv.org/pdf/2506.04646) · [Impact 식별](https://arxiv.org/pdf/1710.04979)

| 다음 연구 질문 | 근거와 중요성 | 변수와 검증 |
|---|---|---|
| 1. 실제 재료에 같은 속도 반전이 존재하는가? | [Yu et al.](https://arxiv.org/pdf/1604.04038)의 재료별 속도·방향 차이는 반전 여부까지 결정하지 않는다. | 양방향 속도 sweep으로 material별 마찰 곡선을 동결한 후 holdout 충돌을 검증한다. |
| 2. 접촉점의 높이가 움직여도 형상 효과가 남는가? | [Fazeli et al.](https://arxiv.org/pdf/1710.04979)에서 국소 운동 상태가 예측 정확도에 영향을 준다. | Height jitter의 크기를 정해 local normal과 label이 바뀌는 비율을 함께 측정한다. |
| 3. 카메라에서 nuisance 방향을 알아낼 수 있는가? | [Tangent Prop](https://proceedings.neurips.cc/paper/1991/file/65658fde58ab3c2b6e5132a39fae7cb9-Paper.pdf)은 알려진 변형을 출발점으로 삼는다. | 정답 없는 paired image로 방향을 추정하고 synthetic oracle projection과 Brier를 비교한다. |
| 4. 여러 nuisance 방향을 제거하면 어디서 정보 손실이 생기는가? | 같은 [변환 불변성 연구](https://proceedings.neurips.cc/paper/1991/file/65658fde58ab3c2b6e5132a39fae7cb9-Paper.pdf)에서 불변으로 지정할 방향은 사전 지식이다. | Rank 1~3의 projection과 무변형 control에서 손익을 측정한다. |
| 5. 특정 probe 속도만으로 새 law를 구별할 수 있는가? | [Fazeli et al.](https://arxiv.org/pdf/1710.04979)의 stick 식별 한계로 같은 관측이 여러 해를 허용한다. | 속도 crossing의 양옆에서 1회씩 probe해 held-out law discrimination과 Brier를 잰다. |
| 6. 추가 접촉의 Brier 감소가 접촉 비용을 상쇄하는가? | [Action-BED](https://arxiv.org/html/2606.23662v2)는 downstream loss를 실험 목적으로 삼는다. | Fixed 3/4 정책에서 측정·reset 시간과 slip 비용을 같은 utility로 정의한다. |
| 7. 조건 비중이 달라도 최선의 보정이 유지되는가? | [WILDS](https://proceedings.mlr.press/v139/koh21a.html)에서 분포 구성의 변화가 성능에 영향을 준다. | 별도 material cohort에서 prevalence를 얻어 동일 정책을 재평가한다. |
| 8. 항상 source 평균 Δ만 쓰는 예측보다 나은가? | [Hierarchical stacking](https://arxiv.org/abs/2101.08954)은 이질성 활용에 자료가 필요함을 보인다. | 완전히 제외한 다수의 law에서 pooled·law-only·full forecast의 MAE를 비교한다. |
| 9. 확률 정확도 차이가 실제 manipulation 성공률로 이어지는가? | [ActivePusher](https://arxiv.org/pdf/2506.04646)는 planning까지 평가한다. | 미리 정한 행동 threshold에서 slip·오선택 비용·cycle time을 함께 기록한다. |
| 10. 하나의 접촉 simulator에서 찾은 손해 조건이 다른 solver에서도 재현되는가? | [Fazeli et al.](https://arxiv.org/pdf/1710.04979)에서 6개 contact model은 impulse를 다르게 설명한다. | 다른 solver를 블라인드 target으로 두고 동일한 forecast와 matched geometry를 재검증한다. |
