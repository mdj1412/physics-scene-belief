## Numerical checks와 남은 근사 오차
Main acquisition은 float64 Gauss–Legendre order41, actual decoder는 order61이다. Independent test의 first32 worlds에서 모든 action과32 noise repeats를 order81과101로 다시 계산했다. Main choices와 main results는 덮어쓰지 않았다.
| Decoder | 최대 velocity component 차이:61→101 | 최대 차이:81→101 |
| --- | --- | --- |
| joint | 5.6360899e-05 m/s | 3.6035233e-05 m/s |
| plugin | 9.6342286e-05 m/s | 4.6430449e-05 m/s |
| mixture | 5.7272116e-05 m/s | 3.6942816e-05 m/s |
| factorized | 3.8829515e-05 m/s | 2.0240891e-05 m/s |

| Policy/native decoder | 선택 action의 평균 MSE 변화:61→101 | 변화:81→101 |
| --- | --- | --- |
| joint | +4.0776644e-08 | +1.97077554e-08 |
| plugin_aware | +1.7453943e-08 | +1.58218851e-08 |
| mixture_aware | +3.92881347e-08 | +1.86491258e-08 |
| factorized | +1.59928832e-08 | +9.37684682e-09 |
| rule_best | +3.46241464e-08 | +1.96745799e-08 |
| lmmse | +4.07986491e-08 | +1.97171118e-08 |

이는 first32 worlds의 audit이며 전체384 test/stress worlds에 대한 uniform numerical error bound는 아니다. Hard plug-in은 discrete model의 argmax가 바뀌면 prediction이 불연속적으로 변할 수 있다. 매우 작은 method 차이에 대해서는 numerical error와 statistical uncertainty를 함께 봐야 한다.
Physics validation에서는 최대 energy excess=-0.047324566, finite-difference slope error=1.367878e-11, factorized marginal preservation error=1.3877788e-17였다.
