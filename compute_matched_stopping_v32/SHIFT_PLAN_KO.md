# 물리 법칙 변경 시 평가 대상 자체의 차이

Held-out Brier 비교를 해석하기 위해 각 law의 실제 query stick label과 같은 latent world에 nominal Coulomb을 적용한 label의 불일치율을 계산한다. 새 simulator의 factual·query view·probe 값을 정책이 사용한 보고된 source 기준과 구별한다. 이 진단에는 지상 정답을 사용하며 stopping의 입력으로 쓰지 않는다.

총 Brier 차이를 source label과 동일한 물체, label이 바뀐 물체의 두 부분으로 기술 분해한다. 각 부분은 **모든 물체 수로 나눈 질량 가중 기여**다. 두 부분을 더하면 실제 전체 policy 차이와 같아야 한다. 이는 선택 또는 decoder 오류의 원인·매개효과 추정이 아니다. 두 subset은 물성·관측·action 이득 분포가 다를 수 있다. 구간은 law별 block을 단위로 계산한다.
