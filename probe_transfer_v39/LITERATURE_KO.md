## 문헌에서 본 v39의 위치

이 검토는 예측 가중치의 일반화, 실험 정보의 이용, 접촉의 관측성이라는 세 질문을 대상으로 했다. 논문별 평가 단위가 다르므로 각 연구가 직접 측정한 결과만 비교했다.

| 원문과 질문 | 방법과 표본 | 결과 및 결론 | v39에 적용할 때 남는 한계 |
|---|---|---|---|
| Yao et al., *Bayesian Hierarchical Stacking*. 두 모델의 우열이 입력에 따라 달라질 때 어떻게 결합하는가? | 여러 predictive distribution의 input-dependent weights를 partial pooling으로 추정하고 이론과 응용 자료에서 평가했다. | 예측 성능이 이질적인 영역에서 가중치가 변하도록 모델링할 수 있음을 보였다. | 원문은 predictive density의 결합을 다룬다. v39는 방향 단서를 log-odds로 얼마나 보정할지 고른다. 연속 가중치를 고른다는 사실만으로 독창성을 주장하기 어렵다. [원문](https://arxiv.org/abs/2101.08954) |
| Koh et al., *WILDS* (ICML 2021). 같은 방법이 다른 수집 환경에서도 작동하는가? | 병원, camera trap, 지역·시기 등 자연적으로 발생하는 shift를 반영한 10개 dataset에서 평가했다. | 통상적인 학습이 distribution shift에서 일반적으로 성능 저하를 보인다는 결과를 보고했다. | v39에서 독립 catalog는 synthetic population의 반복이다. 두 형상·두 law 실험에서 얻은 interval로 실제 재료 분포를 대표할 수 없다. [원문](https://proceedings.mlr.press/v139/koh21a.html) |
| Rossa et al., *ACTION-BED* (NeurIPS 2026, 저자 공개본). 어떤 실험이 이후 손실을 낮추는가? | Joint model에서 생성한 sample과 expected future loss로 design policy와 action policy를 학습하고 source-location(두 source, 30단계, 2,048개 평가 rollout)·dynamics 예제 등에서 평가했다. | Parameter uncertainty 감소 외에도 downstream loss에 직접 맞춘 실험 설계를 구성할 수 있음을 보였다. | v39는 네 번의 동작을 고정한다. 검증 대상은 source에서 학습된 보정의 효과가 어떤 target에서 유지되는가이다. [원문](https://arxiv.org/pdf/2606.23662) |
| Saha et al., *New Evidence, Same Choice* (2026 공개 원고). 물리 증거가 달라지면 다음 선택도 달라지는가? | 6개 VLM, 144개 parameter set의 sliding·bouncing·spring 문제에서 추가 측정을 고르는 선택과 최종 답을 따로 평가했다. | 정답 행동이 바뀐 pair에서도 direct response는 자주 같은 행동을 반복했다. | 선택 변경이 필요한 사례를 만든 선례다. v39에서는 행동 변경률 대신 새 물체의 실제 충돌을 Brier로 채점한다. [원문](https://arxiv.org/html/2609.11022v1) |
| Fazeli et al. 물리식과 실제 충돌이 어느 범위에서 일치하는가? | 실물 planar drop 2,000회 중 품질 검사를 통과한 1,718회에서 6개 rigid-body model에 parameter를 맞추고 post-impact impulse를 비교했다. | 서로 다른 parameter가 비슷한 impact를 설명할 수 있고, 모델이 표현할 수 있는 impulse에도 제약이 있었다. | 두 높이에서 local slope가 가까워도 미지 물체의 correction error까지 가까워지는 물리적 보장은 없다. [원문](https://arxiv.org/pdf/1710.04979) |
| Zhong et al., *ActivePusher* (ICRA 2026). 예측을 보정하고 불확실한 push를 고르면 실제 manipulation이 향상되는가? | Residual physics와 uncertainty로 학습용 push 및 실행할 동작을 고르고 simulation·실제 UR10에서 평가했다. | 더 적은 상호작용으로 모델을 학습하고 여러 단계의 planning success를 개선한 결과를 보고했다. | 네 번의 측정 후 별도 물체의 사건확률을 맞힌 v39와 실물 성공률의 평가 단위가 다르다. [저자 공개 자료와 원문](https://elpislab.org/projects/activepusher/) |
| Gneiting & Raftery. 불확실한 binary outcome의 확률을 어떻게 비교하는가? | Proper scoring rules의 이론을 정리하고 forecast 사례를 분석했다. | True distribution에 최적인 expected score의 조건을 정리했다. | Brier는 다음 충돌에 대한 확률의 품질을 측정한다. 장비 손상이나 slip 대응 행동에 따르는 실제 비용은 별도 효용을 정의해야 한다. [원문](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf) |
| Steyerberg et al. 다른 집단으로 예측기를 옮길 때 복잡한 재학습이 필요한가? | TIMI-II의 3,339명에서 얻은 모델을 GUSTO-I 40,830명 자료에서 평가하며, 크기가 다른 fitting sample로 intercept 또는 slope 보정과 더 큰 수정을 비교했다. | 검증 표본을 200명부터 10,000명까지 바꿔 model updating의 표본 의존성을 평가했다. | 의학적 결과를 충돌에 적용할 근거는 아니다. 보정 이득이 사실상 전체 발생확률의 편향인지 검토하는 single-intercept 대조의 방법론적 선례다. [원문](https://onlinelibrary.wiley.com/doi/10.1002/sim.1844) |

**연구 관계와 해석.** Contact geometry와 friction은 주어진 속도에서 어떤 운동을 관측할지 결정한다. 관측은 source model의 표면 posterior와 signed cue를 바꾸고, 보정량은 두 endpoint 사이에서 예측 확률을 이동시킨다. 그 확률 이동이 실제 잔차의 방향에 맞아야 Brier가 줄어든다. 앞의 물리적 관측성과 끝의 scoring 원리는 근거가 강하지만 **두 local slope만으로 새 환경에서 그 정렬까지 예측할 수 있는가**는 미확정 상태였다. 네 개의 사전 예측은 이 연결에 대한 좁은 검증이다. [Impact model](https://arxiv.org/pdf/1710.04979) · [Input-dependent weighting](https://arxiv.org/abs/2101.08954) · [Proper scores](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf)

**충돌하는 것처럼 보이는 결과.** Partial pooling 연구에서 input-dependent weighting이 도움이 될 수 있다는 결과가 있어도, 같은 source weight가 모든 target shape에서 유리할 필요는 없다. 동일 환경에서 예측기가 상호 보완적인지, 빠져 있던 새 환경에 기존 weight mapping이 옮겨가는지는 다른 가정이다. 실제 로봇의 push planning에서 얻은 개선도 네 번의 접촉과 terminal Brier에 그대로 대응하지 않는다. 이러한 차이를 감안해 네 cell의 성패를 따로 기록하고, 사후 설명과 사전 예측을 구분했다. [Hierarchical stacking](https://arxiv.org/abs/2101.08954) · [WILDS](https://proceedings.mlr.press/v139/koh21a.html) · [ActivePusher](https://elpislab.org/projects/activepusher/)

### 문헌이 남기는 검증 가능한 질문 10개

| 질문과 근거 | 왜 필요한가 | 변수와 후속 검증 |
|---|---|---|
| 1. 두 높이의 slope가 비슷하지만 나머지 contour가 다르면 부호가 유지되는가? [Fazeli et al.](https://arxiv.org/pdf/1710.04979) | 관측이 같은 물리 상태를 유일하게 정하지 않을 수 있다. | Slope-matched shape 여러 개를 생성하고 전체 형상을 단위로 hold out한다. |
| 2. Slope를 연속적으로 바꾸면 손익분기점이 재현되는가? [Fazeli et al.](https://arxiv.org/pdf/1710.04979) | 두 형상만으로 transition curve를 정할 수 없다. | 한 축을 sweep하고 별도 곡선 family에서 crossing을 예측한다. |
| 3. 새 contact law에서도 기하에 따른 부호가 유지되는가? [WILDS](https://proceedings.mlr.press/v139/koh21a.html) | Catalog shift와 mechanism shift는 다르다. | 기존 두 law를 학습에만 두고 rate dependence가 다른 law 전체를 제외해 평가한다. |
| 4. 실물 sensor noise를 대입하면 효과가 줄어드는가? [ActivePusher](https://elpislab.org/projects/activepusher/) | 실제 행동의 불확실성은 동작에 따라 달라질 수 있다. | 속도별 noise·clipping을 측정해 동결한 예측의 Brier를 다시 잰다. |
| 5. 표면 이름 없이 예상 이득이 큰 환경을 고를 수 있는가? [Yao et al.](https://arxiv.org/abs/2101.08954) | Cell 이름을 쓰는 규칙은 미지 표면의 실시간 정책과 다르다. | 허용된 접촉 cue만으로 선택하고 law-blind holdout으로 검증한다. |
| 6. 세 번째 관측에서 이미 네 번째 보정의 성패를 예측할 수 있는가? [Saha et al.](https://arxiv.org/html/2609.11022v1) | 접촉한 뒤의 판단과 접촉할지의 판단은 다른 문제다. | Prefix-3 scorer와 prefix-4 scorer를 별도 모델로 고정하고 비용까지 계산한다. |
| 7. Source gain의 최적값을 축소하면 손해 cell의 악화를 줄이는가? [Hierarchical stacking](https://arxiv.org/abs/2101.08954) | 한 source에서 맞춘 가중치는 prediction error를 옮길 수 있다. | 보정량 shrinkage를 source 안에서 결정하고 독립 catalog의 worst-cell Brier를 측정한다. |
| 8. 어느 sensor가 이미 충분한 정보를 제공하는가? [ACTION-BED](https://arxiv.org/pdf/2606.23662) | 추가 관측의 가치는 이미 얻은 factual evidence에 의존한다. | Deployment query와 probe를 무작위 배정해 correction과의 interaction을 측정한다. |
| 9. 네 cell의 현실적 빈도를 적용하면 정책 순위가 달라지는가? [WILDS](https://proceedings.mlr.press/v139/koh21a.html) | 균등 평균은 배포 시 발생률과 다를 수 있다. | 독립 material cohort에서 prevalence를 구해 같은 frozen 결과를 재가중한다. |
| 10. 작은 Brier 개선이 로봇 행동을 바꾸는가? [Proper scoring rules](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf) · [ActivePusher](https://elpislab.org/projects/activepusher/) | 확률 지표와 실패 비용의 단위가 다르다. | Slip 실패 비용과 동작 threshold를 사전에 정하고 regret·cycle time을 실측한다. |
