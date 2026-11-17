# v65 · 실물 표본 수를 위한 계산 계획

분모는 같은 surface에서의 **새 8-impact calibration pilot와 별도 deployment query 1개의 paired 구매 결정**이다. A와 B의 frozen policy가 같은 deployment prefix에서 다른 R/I 구매를 제안하면 한 건이다. 한 impact, lamp 한 개, frame 한 장과 calibration 내부의 4개 session은 별도 표본으로 세지 않는다. 각 반복은 표면의 새 설치·접촉 위치·관측으로 이루어지고 deployment query도 새로 뽑는 것으로 설계한다. 동일 bank를 여러 query에 재사용하는 실험은 이 표와 다른 cluster 구조를 갖는다.

v64 전체 평균과 crossed catalog×board bootstrap 95% 양끝을 planning scenario로 둔다. 이는 field prior나 field lower bound가 아니다. 실제 확률 0도 열려 있다. 새 표면을 독립 cluster로 가정하고 표면당 세션 1, 4, 8, 16, 32회, ICC 0·0.01·0.05·0.10을 비교한다. Beta-binomial model과 정확한 convolution으로 (i) 전환 ≥1건을 볼 확률 95%, (ii) ≥10건을 볼 확률 90%, (iii) ≥10개 서로 다른 표면에서 전환을 볼 확률 90%를 만족하는 최소 정수 surface 수를 구한다. ICC는 설계 민감도이고 표면마다 공통된 외부 board·operator·day 효과는 해당 가정 밖이다. 고정한 소형 pilot에서 0건 가능성, ≥1건 가능성, 고정 예산에서 더 많은 표면과 더 많은 반복의 차이도 보고한다.

정밀도는 (a) 독립 paired session에서 equal-tailed 95% Clopper–Pearson interval의 총 폭이 p 또는 p/2 이내일 확률 90%를 만족하는 첫 100-session grid 지점, (b) cluster 설계에서 목표 25% 또는 15% relative standard error를 만족하는 정확한 분산에 따른 표본 수로 구분한다. Brier 차이를 검정할 표본 수는 구매 전환율 한 개로 정하지 않는다. 누락 민감도는 완전한 paired session이 90%인 예에서 독립 결측 가정을 명시한다. Exact 계산은 simulation과 iid 경계, 최소 표면 수 바로 전 지점에서 검산한다.
