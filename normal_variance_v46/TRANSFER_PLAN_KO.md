# Source 규칙 전이가 실패하는 경우까지 해석하기 위한 보조 진단

배포 평가는 항상 source에서 동결한 rule로 수행한다. 별도 분석에서 target catalog를 하나씩 완전히 제외하고, 나머지 11개 catalog의 label로 같은 여섯 x bin 및 네 variance bin의 Brier 차이를 다시 추정한다. Bin 경계는 source 것 그대로, shrinkage는 25% 그대로 둔다. 빈 cell은 해당 x의 평균을 쓴다. 같은 11개로 refit한 x-only와 x+variance를 제외된 catalog에서 비교한다. Predicted·analytic geometry 각각에 적용한다. 이는 같은 두 target family를 미리 관찰할 수 있을 때의 진단이며 새 family로 zero-shot deployment한 결과와 구분한다. Source gate와 target conditional loss의 부호가 어디서 달라지는지, 해당 cell의 분포 비중과 함께 보고한다. 12개 fold의 training 표본이 겹치므로 구간은 기술적으로만 해석한다.
