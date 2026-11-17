# v55 · 정답 열람 전 고정한 확인 실험

검증할 정책은 v54의 policy54.select54(weak3, ref3, mode='robust')다. Fourier profile, material ID, law·catalog 이름, 교체 전 관측, 네 번째 weak 신호나 두 potential fourth readout, deployment label은 selector에 제공하지 않는다. 모든 정책은 v53의 frozen nominal continuous joint decoder를 사용한다. 같은 catalog, 동일한 surface와 3쌍의 prefix, 두 potential fourth readout, 동일 deployment 물체와 noise를 공유한다. v54의 simulator 설계와 terminal decoder를 보존해 독립적으로 새로 그린 surface와 session에서 재현을 측정한다.

## Endpoint와 판정

Primary는 평균 held-out Brier loss에서 gate−fixed 3R+I와 gate−rate-matched random, 두 paired contrast다. v54와 같은 목표 분포를 사용한다. e=(0,.25,.5), g=(0,.25,.5,1), 교체 전 방향 유지율 k=(0,.25,.5,.75,1)을 각각 균등 가중하고, swap이 실제 방향을 뒤집는 event는 확률 g다. 생성기가 각 조건을 층화한 경우 그 조건에서 동등 가중한다. 단일 realization으로 뽑을 경우 동일 설계 확률을 따른다. 평균은 deployment orientation과 가능한 contact/solver 결과에 대해 분석적으로 적분한 Brier를 우선 사용한다. Session p90은 128개 deployment object에서 실현된 Brier의 session 평균이 어느 값에서 전체 session의 90%를 포함하는지로 정의한다. 확률 혼합의 p90은 각 arm의 loss 분포 CDF를 혼합한 뒤 역함수로 계산한다. 평균 예측을 혼합해 채점하거나 서로 다른 p90을 평균하지 않는다.

Random policy의 구매율은 새 평가 prefix에 frozen gate를 적용해 얻은 marginal frequency다. Session과 무관한 Bernoulli action의 두 potential loss를 그 확률로 정확히 적분한다. Label을 열지 않고 구매율을 정할 수 있다. 고정된 session pool의 구매 횟수를 정확히 일치시키고 무작위로 slot을 치환해도 각 slot의 주변확률은 같다. 향후 deployment에서 realized counts가 정확히 같은 것은 같은 개수의 ticket을 미리 섞은 방식이고, 독립 Bernoulli에서는 기댓값만 같다.

Catalog를 동일 가중 paired cluster로 5,000회 재추출한다. 각 bootstrap에서 target gate 구매율과 random CDF를 다시 계산한다. 두 primary mean 비교에 대해 Bonferroni simultaneous 95% percentile interval을 제시한다. Lower is better이며 두 upper bound가 모두 0 미만일 때 방향 재현을 선언한다. 0.00001 absolute Brier를 개발 단계의 크기 참고선으로 사전 지정한다. 이전 두 mean gain(약 0.000018, 0.000023)의 대략 절반을 유지하는지 묻기 위한 수치다. 실물 실용성의 기준은 아직 없다. 같은 크기의 support를 선언하려면 두 upper bound가 모두 −0.00001 미만이어야 한다.

두 session p90 대조는 별도의 secondary family에서 같은 catalog resampling, 두 비교 Bonferroni simultaneous 95% interval을 사용한다. Primary와 family를 합친 전체 동시 보장은 주장하지 않는다. 다른 deployment draw에서 p90 direction, surface별 평균차이·catalog win count, mismatch별 차이, impact 구매율, fixed 4R 및 같은 catalog 내부 구매율로 한 번 더 맞춘 random은 sensitivity다. 변동 분석의 단위는 catalog이며 같은 prefix나 여러 e, g 조건을 독립 surface로 세지 않는다. 새로운 catalog만 쓰고 v54와 합쳐 significance를 계산하지 않는다. Frozen policy를 수정하지 않으며 결과에 따라 표본을 늘리지 않는다.
