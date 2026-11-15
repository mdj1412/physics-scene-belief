# Secondary diagnostics — main outcome table 열람 전 추가

Primary protocol/settings는 변경하지 않는다. Public factual posterior에서 event probability p와 maximum LMMSE score를 이용해 다음 subgroup을 구분한다: ambiguous .2<=p<=.8; low-linear-gain은 p(1-p)>.05 AND maximum LMMSE gain/[p(1-p)]<.2. Low-linear-gain subset은 variance가 남는데 covariance acquisition이 낮은 조건이다. Selection 후 target으로 subset을 정하지 않는다. Subgroups는 exploratory diagnostics이며 full-population primary와 구분한다.

같은 nominal action choices를 QMC64 decoder로 재평가하여 finite-bank decoder error를 확인한다. QMC16/64 actual-observation ESS도 공개한다. 이 비교가 exact posterior error의 측정은 아니다.
