# 보조 continuous comparison

Primary 결과를 읽기 전, 이미 계획한 posterior-mean continuous correction과 별도로 v37 source expected-gain table 위에서 0≤w≤1의 Brier gain을 최대로 하는 보정량을 평가한다. Table은 z축 방향으로 선형 보간되므로 최댓값은 0, z 또는 두 끝점 사이의 knot에 있다. 동률이면 32.5%에 가장 가까운 최적 w를 고른다. 이득을 추가 source·target fitting 없이 계산할 수 있다. Primary 판정은 원래 두 대조를 유지한다. 이 보조 방법과 3-action을 같은 GPU·batch·warmup 조건에서 순서를 섞어 다시 측정한다. 세 번째 자리 이하의 두 timing session 차이를 방법 효과로 해석하지 않는다. 측정된 raw Brier와 해당 정책의 prefix3–4 Brier gain을 진단한다.
