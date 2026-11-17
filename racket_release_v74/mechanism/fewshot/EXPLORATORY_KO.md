# Paid calibration으로 반전 경계를 찾을 수 있는가?

같은 시편의 많은 관측에서 효과가 있었던 $|s_{in}|$ threshold 형태를 본 뒤 설계한 **탐색 분석**이다. 앞선 진단에서 얻은 6.592 등의 cutoff 수치는 이 정책에 넣지 않았다. 실제로 지불한 4/8회 중 fast이고 $|s'|>0.20$인 사례에서 반전과 유지가 모두 관측되면 calibration error가 가장 작은 경계를 고른다. 두 class 중 하나라도 관측되지 않으면 항상 반전한다. 동률 순서와 projection 규칙을 코드 실행 전 고정했다. [고정한 규약](FROZEN_KO.md) · [SHA-256](DESIGN.sha256)

| Split · paid K | 판독된 fast / 전체 fast | Threshold / 항상 반전 적중률 | Δ 적중률, pp [percentile 95% 구간] | Threshold velocity / spin | 항상 반전 velocity / spin |
|---|---:|---:|---:|---:|---:|
| Official · 4 | 22/37 | 86.43 / 86.36% | +0.07 [−1.35, +1.60] | 0.367 / 11.86 | 0.367 / 11.84 |
| Official · 8 | 22/37 | 87.00 / 86.36% | +0.64 [−3.98, +5.70] | 0.331 / 11.48 | 0.337 / 11.77 |
| Secondary · 4 | 33/68 | 74.21 / 72.73% | +1.48 [+0.004, +2.96] | 0.397 / 14.11 | 0.399 / 14.12 |
| Secondary · 8 | 33/68 | 78.04 / 72.73% | +5.31 [+0.55, +10.12] | 0.366 / 13.66 | 0.371 / 13.75 |

Velocity는 m/s, spin은 rad/s다. 두 규칙은 같은 elastic 예측에 동일한 impulse-preserving projection을 적용한다. 각 impact에서 256개 draw의 적중률 차이를 먼저 평균한 뒤, 고유한 판독 가능 held impact를 4,096회 paired bootstrap했다. 구간은 같은 시편과 관측된 calibration bank에 조건부이며 secondary K4 구간의 하한은 사실상 0이다. Official과 secondary의 calibration 자료에는 중복이 있다. [수치와 분모](results.json) · [분석 코드](evaluate.py)

양쪽 class가 관측된 draw는 official에서 K4 24/256과 K8 84/256, secondary에서 24/256과 77/256이었다. K8에서 그중 실제로 유한한 단일 cutoff로 calibration label을 완전 분리한 draw는 각각 83개와 75개였다. 그 draw에서도 reversal과 retain 사이에 허용되는 cutoff 구간의 폭 중앙값은 3.20, 2.46 m/s였다. 두 class를 읽었다고 정밀한 경계를 얻은 것은 아니다. 양쪽 class가 관측된 K8 draw에서만 계산하면 official은 88.31%, secondary는 90.40%였다. 나머지 draw는 fallback으로 항상 반전과 같아졌다. 이 조건부 적중률은 전체 budget 성능과 별도로 해석해야 한다. [Draw별 관측과 cutoff](draws.csv)

같은 시편의 secondary 8회에서는 일부 draw에서 얻은 경계가 단순 반전에 추가 이득을 보였다. Official에서는 단순 반전 대비 추가 효과의 범위가 영점을 포함했다. Threshold 가설을 형성하는 단계에서 이미 같은 시편의 label 구조를 확인했으므로, 새 물체에서의 성능 주장을 세우려면 cutoff 학습·fallback을 고정하고 독립 시편 또는 이후에 수집한 세션에서 평가해야 한다.
