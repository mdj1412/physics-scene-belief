# 조건부 wire-grid diagnostic

같은 Wilson ball의 별도 normal impact만 분석한다. Cross의 author manuscript §2 (Fig. 4b)에서 7.7±0.1 m/s, 접촉 시간 4.72±0.01 ms, R=33.0 mm를 입력으로 사용한다. Radius가 5·10·15·20 mm가 되는 첫 시점과 최대 직경 42±1 mm를 측정값으로 비교한다. 기준 e=0.75, 민감도 e=0.65, 0.85를 사전에 고정한다. 연구팀이 이미 공개된 4개 시점을 열람한 상태의 exploratory check이다.

Compliant solver의 normal oscillator에 spherical-cap readout을 붙여 계산한다. Loading과 unloading의 angular frequency 비는 1:e다. Normal contact duration으로 두 quarter-cycle의 합을 고정하고, predicted indentation으로 radius를 읽는다. Constrained CPU Slurm step에서 analytic solution을 계산한다. Rigid PyBullet이 반환하는 점접촉에는 대응되는 면적 time series가 없다. 별도 oblique impact 자료나 두 solver의 Brier loss와 합산하지 않는다.
