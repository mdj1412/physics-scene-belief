# Calibration witness로 방향 유지율을 추정하는 과정

κ는 active인 물체에서 **같은 위치의 weak displacement와 reference full-load displacement의 부호가 일치할 확률**이다. Inert 상태에서는 부호를 정의하지 않는다. Calibration object 하나를 0.4 N으로 누르고 reset한 뒤 같은 위치의 기준 2.0 N 변형을 별도로 측정한다. 한 쌍마다 contact 2회·reset 2회가 들며, 실험에 쓰는 full-load reference는 signed deformation을 읽는 별도 관측이다. Stick/slip outcome이나 PyBullet의 impact force를 읽지 않는다. Reference deformation이 향후 impact deformation과 같은 polarity law를 따른다는 것이 핵심 가정이다.

매 catalog·surface/law environment에서 128개 calibration specimen으로 구성된 별도 prototype support에서 매 session에 최대 8개를 비복원 추출했다. 독립적인 256개 session realization과 5개 κ를 비교했다. 각 environment의 4 family는 32개씩이고 접촉 높이는 19/31 mm 중에서 뽑았다. 같은 K 비교에는 앞에서부터 동일한 측정 prefix를 썼다. 다른 simulated session 사이에서는 prototype을 다시 뽑을 수 있다. 따라서 독립 실물 물체가 수백만 개라는 의미는 없다. κ는 동일 environment 안에서 calibration과 deployment가 공유한다. 각 specimen의 weak/full 측정은 같은 형상과 고정된 object gain을 공유하며 meter noise는 독립적이다. Weak meter의 station-level gain과 bias는 deployment와 공유한다. Full reference channel에는 별도 gain·bias와 SD 0.060 mm의 noise를 더했다.

Estimator 입력은 두 noisy scalar뿐이다. 세 latent state에 prior (0.5, 0.25, 0.25)를 두고, 각 측정에 대한 nominal Gaussian likelihood로 상태를 적분한다. Weak/full nominal active amplitude 0.084/0.420 mm와 SD 0.035/0.075 mm를 고정했다. Weak 절댓값이 0.018 mm 미만이거나 full 절댓값이 0.120 mm 미만인 쌍은 정보를 보태지 않는다. 회당 likelihood는 κ·L(같은 방향)+(1−κ)·L(반대 방향)이다. Uniform prior로부터 최대 K=8쌍의 posterior mean을 Gauss–Legendre quadrature로 계산한다. Hard policy는 mean <0.5일 때만 부호를 바꾸고 증거가 없는 tie에서는 원래 부호를 유지한다. 이 prior, threshold, amplitude를 target label로 fit하지 않았다. [Estimator](estimator50.py) · [Generator](calibration50.py)

| 실제 κ | K=4 posterior mean | K=4 MAE | K=4 RMSE | Flip 확률 K=1 | K=4 | K=8 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.272 | 0.272 | 0.284 | 51.05% | 93.77% | 99.61% |
| 0.25 | 0.385 | 0.148 | 0.201 | 38.41% | 73.78% | 84.97% |
| 0.50 | 0.499 | 0.134 | 0.166 | 25.90% | 47.66% | 50.48% |
| 0.75 | 0.612 | 0.151 | 0.204 | 13.66% | 21.61% | 16.11% |
| 1 | 0.728 | 0.272 | 0.284 | 0.88% | 0.41% | 0.09% |

K=4면 사용 가능한 쌍은 평균 2.04개였다. κ=0과 κ=1에서 posterior mean을 각각 0.272와 0.728로 추정했다. 어느 방향을 선택할지는 자주 맞혀도 유지율의 크기는 prior 쪽으로 상당히 축소한다. K=1의 tie는 48.14%, K=4에서는 5.85%였다. κ=.5에서는 어느 부호도 정답으로 지정할 수 없다. [전체 bias·MAE·RMSE와 선택률](calibration50.json)

**Pairing 대조.** 같은 environment·κ에서 full reference만 다음 독립 session으로 순환시켜 weak와 full의 개체 대응을 깼다. K=4에서 추정 평균은 κ=0,1일 때 각각 0.498, 0.502로 수렴해 구별력이 사라졌다. Weak 측정은 모든 κ에서 bitwise 동일하다. 따라서 이 생성식과 대칭 prior에서 weak 관측만으로 κ를 식별할 수 없다. 충분한 paired witness가 있어도 sensor polarity 자체가 반대로 장착되거나 calibration 뒤 material이 바뀌면 새로운 식별 문제가 된다. [대조 결과](calibration50.json)
