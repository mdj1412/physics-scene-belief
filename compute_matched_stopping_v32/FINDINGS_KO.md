**같은 학습 시간에서 2×2 평균 label로 학습해도 전체 stopping policy의 Brier는 실용적으로 개선되지 않았다. 1×1, fixed 4, uncertainty-only와의 비교 모두에서 0.003 개선은 simultaneous interval 밖이었다.**

첫 seed에서 scalar gain을 한 번 측정한 1×1은 597개 환경, 2개 futures와 2개 batches를 교차한 2×2는 255개 환경을 봤다. 실제 label 생성·fit 시간은 539.34초와 537.57초였다. 두 정책, fixed 4, uncertainty-only의 held-out Brier는 각각 0.13868, 0.13891, 0.13877, 0.13887였다. 2×2 − 1×1 차이는 +0.00023 [-0.00034, +0.00080]였다.

같은 prefix에서 2×2 평균으로 label을 만들면 conditional noise RMS는 2→4 stage에서 1×1의 61.6%, 4→8 stage에서 60.9%가 됐다. 별도 source data로 처음부터 세 번씩 학습했을 때 repeated − broad Brier는 +0.00023, -0.00023, +0.00034였고, 평균은 +0.00011였다.

여덟 환경의 전체 stopping·acquisition·decoding을 모두 실행한 GPU p90은 1×1 1491 ms, 2×2 1490 ms, fixed 4 1418 ms, uncertainty-only 1489 ms였다. 네 방법 모두 한 block에서 calibration 32회와 deployment probe 104회를 지불했다.
