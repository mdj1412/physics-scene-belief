### 이 대조에서 알 수 있는 식별 한계

위 reverse 조건에는 모델을 바꿔도 넘을 수 없는 한계가 하나 있다. 두 실험실이 같은 물체에 완전히 같은 네 번의 probe를 했고, 같은 factual data와 같은 deployment observation을 전달했다고 하자. 한쪽 실험실에서만 **질문한 충돌에서의** directional friction 부호가 바뀐다. 이 실험의 데이터 생성 방식에서는 예측 직전까지 두 실험실을 구분하는 값이 하나도 없다. 어떤 reader를 쓰더라도 그 입력만으로 ‘어느 실험실인가’를 맞힐 수 없다. 실제 재료에서 반드시 이런 상황이 발생한다는 주장이 아니라, 이 설계에서 같은 evidence가 서로 다른 query 결과와 양립한다는 뜻이다.

두 regime을 같은 확률로 고른다고 명시하면 한계의 크기도 계산할 수 있다. 두 정답이 다른 물체에서 공통 evidence에 하나의 확률 u를 내야 할 때 평균 Brier는 [u²+(1−u)²]/2 = (u−0.5)²+0.25다. 따라서 이런 물체는 어떤 u라도 0.25 이상을 지불한다. Latent θ를 추가로 알려주더라도 regime을 알려주지 않으면 같은 하한이 남는다. 전체 하한은 **부호 역전으로 정답이 바뀐 비율 × 0.25**다. 이는 이 인위적 two-regime mixture의 절대 loss 하한이며, primary 평균에서 관측한 오차 전체가 식별 불가능하다는 뜻도, 모든 물리 조건의 minimax bound라는 뜻도 아니다. 이 한계는 우리 보정식에만 생기는 현상이 아니다.

따라서 실패 후 모델의 parameter 수만 늘리는 것보다 정보가 어디에서 들어올 수 있는지 먼저 묻는 편이 좋다. 다음 비교에서는 같은 네 번의 budget 중 하나만 query에 가까운 높이·속도로 옮긴 **query-matched probe**를 배정하고, 그 probe의 response가 두 regime에서 실제로 다른지 실측해야 한다. 차이가 없다면 prediction을 과감하게 이동시키는 대신 base model로 복귀하거나 두 regime에 대한 predictive mixture를 내는 전략이 더 타당하다.
