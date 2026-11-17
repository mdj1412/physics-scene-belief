# v57 · 표본 공개 전 추가 진단

Pilot session은 e∈{0,.25,.5}를 한 번 뽑고 별도 물체의 active mismatch를 e에 따라 독립적으로 생성한다. 환경은 A=(.42,0,.58), B=(0,.84,.16), 같은 prior probability다. 여덟 번의 시도에서 활성 확률 .5의 이상적 부호 관측을 가정해 모든 session별 count의 확률을 열거한다. 8×1과 4×2의 최대 balanced accuracy, v56의 A/B별 Brier 차이로 비용을 정한 risk rule이 새 평균 risk에서 회수하는 가치, usable pair가 0/1/2개 이상인 빈도를 산출한다. Fresh target gap을 아는 post hoc two-environment oracle도 upper bound 진단으로만 쓴다. 2×4와 이론상 모든 물체가 활성인 경우는 동일한 8회에서의 model-only 참고값이다.

공유 원인의 진단으로 두 번째 물체가 첫 번째 물체와 같은 e를 공유하는 확률 rho를 0,.25,.5,.75,1로 바꾸고 나머지는 같은 모집단의 독립 e에서 뽑는다. 원래 rho=1에서 고정된 rule을 그대로 쓰는 경우와 count 분포를 아는 이상적 판별을 계산한다. 이는 추가 hardware 실측이나 primary endpoint가 아니다.

같은 session의 동조된 측정 오류가 환경 식별을 모방할 수 있는지도 확인한다. B에서 두 번째 mismatch indicator가 첫 번째를 복사할 확률 h를 두고 그 외에는 원래 conditional draw를 유지한다. A와 B의 주변 불일치율을 .29로 유지하면서 두 mismatch 동시확률까지 같아지는 h를 구한다. 결론 범위는 독립된 두 물체가 같은 latent e를 공유하고, pilot noise가 그 e에 조건부 독립인 설계로 한정한다. 실제 sensor error rate나 surface prevalence로 해석하지 않는다.
