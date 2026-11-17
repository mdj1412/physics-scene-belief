## 방향 제거가 무엇을 바꿨는가

같은 물체에서 base·hollow·bulge의 실제 stick event가 같기 때문에, 세 contour에서의 평균 Brier가 두 항으로 정확하게 나뉜다. 첫 항은 세 예측 확률의 분산이고, 둘째 항은 세 예측의 평균과 실제 event 사이의 제곱 오차다. 이는 contour에 따른 흔들림을 얼마나 줄였는지와, 흔들림을 없앤 확률의 중심이 얼마나 맞았는지를 구분한다. Fixed 32.5%에서 계산했다.

| 항목 | Original | Oracle | Estimated | Estimated−original, 개별 95% CI |
|---|---:|---:|---:|---:|
| Contour 간 예측 분산 | 0.005295 | 0.000000 | 0.000217 | -0.005078 [-0.005214, -0.004941] |
| Contour 평균 예측의 Brier | 0.120025 | 0.121816 | 0.121499 | +0.001474 [+0.001202, +0.001746] |
| Base만 사용한 Brier | 0.120149 | 0.121816 | 0.121569 | +0.001420 [+0.001170, +0.001671] |
| Hollow·bulge 평균 Brier | 0.127905 | 0.121816 | 0.121790 | -0.006115 [-0.006386, -0.005844] |

세 contour의 비율은 1:1:1이다. 같은 query view 가지에서 base와 두 변형 사이의 보정 전 절대 확률 변화는 original 0.056790, oracle 5.10e-16, estimated 0.008569였다. 같은 물체의 probe 가지는 세 decoder에서 동일했다. [재계산](contour_breakdown41.json)
Base 확률을 1−q로, 나머지 두 변형 확률을 각각 q/2로 놓고 이 실험의 strata를 그대로 재가중하면 estimated와 original의 평균 Brier 순위가 바뀌는 q는 18.8%다. 실세계 prevalence와 그 값의 신뢰구간은 측정하지 않았다.

### 물체 대응 관계와 저렴한 대안

이 보조 비교는 각 panel마다 홀짝으로 지정한 한 block, 32회 중 고정된 noise 0·16만 사용했다. 24개 cell과 12개 panel, 구매한 관측, source bank와 contact label은 모두 같고 추론은 서버 CPU에서 계산했다. GPU p90 비교에는 위 주 실험의 9개 방법과 별도 crossover만 포함한다.

| 같은 물체와 noise에서의 방법 | Group당 추가 profile | Fixed 32.5% Brier |
|---|---:|---:|
| Original | 0 | 0.125342 |
| Oracle projection | oracle 정보 | 0.121875 |
| Estimated projection (matched) | 64 | 0.121743 |
| Unpaired projection (sham) | 64 | 0.199663 |
| Local window (중앙 ±2 mm 세 좌표) | 0 | 0.129383 |
| Finite-variance marginalization (matched) | 64 | 0.121621 |

Estimated−unpaired는 **-0.077920 [-0.084147, -0.071693]**였다. 같은 profile을 유지하고 두 번째 관측의 물체 대응만 한 칸 순환 이동했다. Estimated−local window는 **-0.007641 [-0.008132, -0.007149]**였다. Local window는 contact height와 가까운 세 좌표만 likelihood에 넣는다. [Sham 결과](sham41.json) · [Local window 결과](local_window41.json)

Finite-variance 방법은 pair difference의 첫 eigenvalue에서 sensor noise를 뺀 뒤, 해당 방향의 residual variance를 유한한 값으로 적분했다. 같은 û를 사용해 정보를 부분적으로 남기는 방법이다. Fixed에서 finite-variance−hard projection은 **-0.000122 [-0.000296, +0.000052]**였다. Finite-variance 안에서 selector−fixed는 -0.001124 [-0.001432, -0.000817]였다. Local window와 finite-variance는 주 실험의 평가 도중 별도로 명세한 보조 분석이며 실용 후보 간 순위를 넓혀 보기 위한 것이다. [방법과 결과](marginal41.json)

이 보조 분석의 구간들은 12개 panel의 개별 95% 구간이다. 보조 표본의 평균 Brier를 576개 block의 주 표와 직접 빼지 않았다.
