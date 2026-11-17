## 예측·이력·label을 교차하는 진단

확인 실험은 두 law를 각각 calibration, factual, paid probe 및 terminal impact에 적용한다. 보조 검산으로 동일 catalog·환경·prototype·noise repeat의 고정된 예측과 두 엔진의 terminal label을 모두 교차한다. v67 예측은 두 기존 관측 law에서 각각 loss를 계산한 뒤 평균한다. 새로운 prescribed/compliant 예측은 (a) 네 calibration을 소비하는 source-frozen log-odds update, (b) calibration 보정 생략, (c) 같은 세계·접촉·noise에서 다른 law로 얻은 calibration shift만 바꾼 대조를 채점한다. (c)는 배포 정책이 아니라 calibration 개입의 진단이다. 차이에는 catalog별 pairing을 사용하고 secondary contrast는 탐색용으로 표시한다.

공통 prediction에 실제 label을 교체했을 때 두 정책의 Brier 차이 변화는 −2E[(p₁−p₀)(y₂−y₁)]와 일치해야 한다. 기존 이력/기존 label → 새 이력/새 label의 총 차이를 각 교체 순서에 대해 양분한 평균으로 분해한다. 같은 probability contract로 계산한 결과를 v67 및 v68의 독립 primary scoring과 대조한다. External Bullet subset은 별도 사전계획을 따른다.
