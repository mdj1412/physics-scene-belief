# 동일 factual pool의 continuation 재표집 · v31 보조 분석 계획

주 분석은 paid prefix 뒤에 새로운 calibration 물체를 iid로 더한다. 보조 panel은 v30처럼 같은 1,024개의 factual pool 안에서 paid prefix에 이미 들어간 ID를 제외하고, 각 future에 대해 독립적으로 무복원 추출한 2개/4개를 사용한다. 서로 다른 future의 추출 사이에는 중복이 허용되며 새로운 probe noise를 사용한다. 따라서 보조 panel은 같은 숨은 표면·물성의 finite pool에 조건부다. 두 panel 모두 동일한 paid prefix, 동일 4개 deployment batches, 동일 K256 및 common decoder를 사용하며 추가 probe 횟수도 같다. 기존 pool 데이터를 보유한 환경과 새 물체를 만날 때 처음 factual/probe를 함께 받는 환경의 차이다.

결과를 열기 전 보조 panel에도 주 분석과 같은 crossed method-of-moments, stage별 repeatability, block별 추정과 leave-one-block sensitivity를 적용하도록 정했다. 같은 block을 두 panel에서 함께 재표집하는 law-stratified paired bootstrap 10,000회로 pool − fresh 차이를 계산한다. Stage마다 (1) mean Brier gain, (2) batch main-effect variance − future main-effect variance, (3) 두 축을 모두 바꾼 1×1 순서 일치율, (4) 독립적인 2×2 평균끼리의 순서 일치율을 본다. 8개 pointwise 95% interval과 두 stage의 8개 대비 전체에 대한 simultaneous 95% interval을 함께 산출한다. 두 panel 차이의 CI가 0을 포함해도 동등하다고 결론짓지 않는다. 절대 재현성은 각 panel에서 별도로 판단한다.

Deployment records와 paid inputs를 공유하는 것은 두 panel을 짝짓는 근거다. Fresh future의 새 calibration worlds와 pool-panel continuation에는 서로 다른 독립 난수를 사용하므로 한 물체의 probe response를 한쪽 regime에서 다른 쪽으로 대체한 효과를 식별하는 실험은 아니다. Grid의 측정 반복·환경 쌍을 independent block 수로 세지 않는다. 통계 기록, 판단, 해시를 각각 별도 results_pool 경로에 쓴다.
