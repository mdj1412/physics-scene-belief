| 정책 | Independent fault·균등 delay GPU p90 (ms/512 objects) | 세 fault 관계·균등 delay GPU p90 (ms/512 objects) |
|---|---:|---:|
| Corrected loss-weighted reset | 480.029 | 480.649 |
| Corrected loss-weighted sham | 480.451 | 481.030 |
| 항상 v54 gate | 479.928 | 480.407 |
| Uncorrected loss-weighted reset | 480.173 | 480.326 |
| Uncorrected loss-weighted sham | 485.086 | 487.596 |
| Corrected MAP reset | 493.251 | 488.975 |

| 같은 모집단에서 corrected reset − 대조 | Independent fault 차이 [동시 95% CI], ms | 전체 차이 [동시 95% CI], ms |
|---|---:|---:|
| Corrected loss-weighted sham | -0.422 [-18.078, +0.145] | -0.381 [-14.982, +7.828] |
| 항상 v54 gate | +0.100 [-3.842, +0.828] | +0.242 [-15.062, +8.050] |

데이터는 independent 모집단 정책당 24회, 전체 72회다. 24개 catalog를 함께 16,000회 복원추출해 각 모집단에서 두 primary contrast에 대한 동시 95% CI를 계산했다. 실행 순서 및 직전 정책의 각 조합은 전체에서 각각 12회 나타났다. 관측된 최대 forecast replay 오차는 0였다. Cell별 8개 관측의 p90은 작은 표본의 기술통계다. [Raw pixel-to-terminal 결과](timing60.json) · [원자료 예시](timing60/panel_00.npz)

별도의 CPU microbenchmark에서는 동일 32개 영상의 marker 검출·pixel tracking·policy routing만 따로 실행했다(정책당 432회, compute node56). p90은 corrected reset 0.3474 ms, corrected sham 0.3462 ms, uncorrected reset 0.3451 ms, 항상 gate 0.0012 ms였다. 두 reset 조건의 preprocessing 차이는 0.00235 ms였고, 전체 GPU p90의 차이는 이 값에서 단순히 계산할 수 없다. [Pixel-to-routing 비용](reader_cost60.json) · [측정 코드](reader_cost60.py)
