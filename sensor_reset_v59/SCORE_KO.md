# v59의 정책 판별과 held-out 손실

네 session에서 서로 다른 두 물체에 한 번씩 impact를 주었다. 두 번째 물체 직전에 무작위로 지정한 local sensor 하나를 reset했을 때, 독립 camera도 사용한 조건에서는 최적 정책 판별률이 **54.843%에서 55.227%**로 올랐고 평균 held-out Brier는 **0.976×10⁻⁶** 줄었다. Camera가 없는 조건에서는 reset 후 판별률 차이의 신뢰구간이 0을 포함했고 평균 Brier가 0.191×10⁻⁶ 늘었다. Camera와 reset을 함께 적용한 MAP도 상시 v54 gate보다 평균 Brier가 5.004×10⁻⁶ 높았다. [사전 평가 설계](SCORE_PLAN_KO.md) · [서버 결과](score59.json)

## 두 종류의 판별

표면의 배포환경 A와 B는 평균 deformation mismatch가 29%로 같다. A에서는 fixed 3R+I, B에서는 v54 gate의 기대 Brier가 낮다는 사전 지정이 새 synthetic surface 48개 각각에서 재현됐다. Gate에서 fixed를 뺀 평균 손익은 A에서 +41.006×10⁻⁶, B에서 −87.082×10⁻⁶이다. 이 두 행동 중 맞게 고른 balanced accuracy를 정책 판별률로 사용한다. 별도로 paired mismatch의 원인이 물리적 지속(P)인지 두 local sensor에 공유되는 upstream error(C)인지 맞춘 비율을 측정했다. 같은 A에서도 원인은 P 또는 C일 수 있다. [Deployment 결과](score59_cells.npz)

Source에서 알려진 네 가설(P/C × A/B)의 likelihood를 정확히 합산했다. 각 가설에서 group별 실제 mismatch, upstream readout, local sensor의 latch, randomization target, camera의 sign error를 포함한다. 각 local sensor에서 reference와 impact의 sign error는 각각 2%, camera에서는 3%다. Relative sign의 net error는 3.92%, 5.82%다. 네 group에 걸쳐 동일한 원인과 환경을 가정한다. 그 안에서 물리 state와 오류 state는 session별로 새로 뽑힌다. 개별 group마다 환경을 따로 정한 뒤 투표하는 과정과는 다른 likelihood다. MAP는 posterior odds가 1보다 크면 fixed로 전환한다. Loss-weighted는 v56에서 고정한 비용 비율 2.016을 사용한다. 가려진 camera readout은 적분해 없앴다. Policy에 실제 cause·regime·deployment outcome이나 임의의 simulator state를 입력하지 않았다. [동결한 selector](route59.py)

| Pilot policy | A/B 판별률 | P/C 판별률 | 평균 Brier ↓ | Session p90 ↓ |
|---|---:|---:|---:|---:|
| Reset + camera MAP | 55.227% | 55.813% | 0.160947014 | 0.192559406 |
| Camera MAP | 54.843% | 55.168% | 0.160947990 | 0.192559555 |
| Reset MAP | 53.109% | 50.000% | 0.160948847 | 0.192561403 |
| 두 개입 없이 MAP | 53.164% | 50.000% | 0.160948655 | 0.192561001 |
| Reset + camera loss-weighted | 51.585% | 동일 관측 | 0.160941708 | 0.192560509 |
| Camera loss-weighted | 51.188% | 동일 관측 | 0.160941846 | 0.192560852 |
| Reset loss-weighted | 50.072% | 동일 관측 | 0.160942008 | 0.192561552 |
| 두 개입 없이 loss-weighted | 50.066% | 동일 관측 | 0.160942007 | 0.192561552 |
| Always v54 gate | 50.000% | — | 0.160942010 | 0.192561552 |
| Always fixed 3R+I | 50.000% | — | 0.160965048 | 0.192567423 |
| Privileged A/B | 100.000% | — | 0.160921507 | 0.192524746 |

A/B·P/C를 균등하게 가중했다. 동일한 24 catalog, 48 surface, 두 rate law에서 12,288개 독립 held-out 물체의 Brier를 사용한다. Mean은 deformation 방향을 실제로 뒤집었을 때의 새 binary stick outcome을 해당 환경의 확률에 대해 적분한 값이다. Session p90은 독립 binary deployment draw에서 물체 128개당 평균 Brier 분포의 90번째 백분위수다. 같은 catalog의 여러 물체·sensor·2,048개 pilot replicate를 독립 표본 수로 간주하지 않았다. 아래 구간은 24개 catalog를 5,000회 짝지어 resample한 결과이며 endpoint별 4개 contrast에 대해 동시 95% CI를 적용했다. [채점 구현](score59.py)

| MAP contrast | A/B 판별률 차이 (pp) | 평균 Brier 차이 (×10⁻⁶) | Session p90 차이 (×10⁻⁶) |
|---|---:|---:|---:|
| Camera 있을 때 reset | +0.385 [+0.331, +0.444] | −0.976 [−1.269, −0.706] | −0.149 [−3.081, +1.934] |
| Camera 없을 때 reset | −0.055 [−0.118, +0.009] | +0.191 [+0.089, +0.297] | +0.402 [−1.043, +1.535] |
| Reset 있을 때 camera | +2.119 [+2.027, +2.224] | −1.832 [−2.201, −1.538] | −1.997 [−5.048, 0.000] |
| Reset 없을 때 camera | +1.678 [+1.565, +1.809] | −0.665 [−0.791, −0.530] | −1.445 [−4.154, 0.000] |

Camera가 있을 때 reset으로 인해 fixed 구매가 **A에서 44.38%에서 41.59%, B에서 34.70%에서 31.13%**로 줄었다. B에서 잘못 fixed를 고를 때의 비용이 A에서 fixed를 선택할 때 얻는 이득의 두 배여서 평균 손실이 내려갔다. 두 독립 binary replay에서 평균 reset 효과는 −0.948, −1.018×10⁻⁶이었고 ultra solver에서는 −1.026×10⁻⁶이었다. Primary expected mean 감소는 independent binary label에서도 같은 방향이었다. 4개 primary p90 contrast는 어느 것도 엄격한 감소를 보이지 않았다. [모든 paired contrast](score59.json)

원인이 P인 subset에서는 camera가 있을 때 reset의 MAP 판별률 차이가 −0.062 pp, C인 subset에서는 +0.831 pp였다. 두 subset에서 평균 Brier 차이는 각각 −0.823, −1.129×10⁻⁶이었다. 손실을 고려해 설계한 reset+camera의 평균은 상시 gate보다 0.302×10⁻⁶ 낮았다(pointwise 95% CI 0.152~0.448×10⁻⁶ 개선). 환경을 완벽히 알 때 상시 gate보다 낮출 수 있는 20.503×10⁻⁶의 약 1.47%다. 같은 비교에서 session p90 차이는 −1.043×10⁻⁶ [−3.651, 0.000]이었다. 해당 결과만으로 추가 sensor reset이나 camera의 실측 운영 비용을 정당화하기 어렵다.

## Sensor hazard가 새 표면에 전이되는가

C에서는 upstream fault가 A와 B 각각의 second moment를 모방한다. 따라서 두 환경 사이의 fault 빈도 차이 자체가 정책에 단서를 제공한다. 이 단서가 사라질 때의 결과를 확인하려고 추가 impact 없이 이미 얻은 C pilot을 같은 표면·동일 예산에서 재배정했다. A/B target에 관계없이 두 hazard를 절반씩 배정하거나 source와 반대로 배정했다. P 사례, target 물체·label, likelihood, cutoff를 고정했다. 이 분석은 primary에 미리 지정하지 않은 exploratory stress test다. [재배정 코드](transfer59.py) · [전이 결과](transfer59.json)

| C branch hazard와 배포환경 관계 | Reset+camera MAP의 C branch 판별률 | 전체 정책 판별률 | 전체 평균 Brier | Loss-weighted 평균 Brier − gate (×10⁻⁶), pointwise 95% CI |
|---|---:|---:|---:|---:|
| Source와 일치 | 57.862% | 55.227% | 0.160947014 | −0.302 [−0.443, −0.147] |
| 서로 독립 | 50.000% | 51.296% | 0.160949532 | +0.680 [+0.487, +0.883] |
| Source와 반대 | 42.138% | 47.366% | 0.160952049 | +1.661 [+1.405, +1.931] |

메커니즘 P/C를 구별할 수 있어도 어떤 action이 유리한지 알려면 그 메커니즘과 실제 deployment 손익 사이의 연결이 필요하다. Source와 일치한 설정에서의 작은 개선이 hazard를 재배정하면 손해로 전환된 점이 이 연구의 중요한 경계다. 위 결과는 같은 synthetic pilot과 결과의 반사실적 재배정이므로, 새 실험 장비의 fault와 마찰이 실제로 독립 또는 반대라는 증거는 아니다. [Downstream 결과와 checksum](score_results59.sha256)
