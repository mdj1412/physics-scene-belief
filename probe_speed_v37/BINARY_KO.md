## 두 surface만을 후보로 가정했을 때의 판별력

일반 posterior mean strength는 strong, weak, local-null, global-null을 모두 고려한다. 이 요약값이 weak와 local-null의 작은 차이를 희석하는지 검사하기 위해 후보를 둘로 좁힌 source likelihood ratio를 계산했다. 각 class에서 네 source family와 양쪽 orientation을 적분했다. 어떤 표면인지 정답은 사용하지 않지만 가능한 상태가 둘뿐이라는 정보는 추가로 제공한다. 두 점수는 같은 paid observations를 보았다.

| Speed | Posterior mean strength AUROC | Weak/null log likelihood ratio AUROC | 두 값의 차이 [95%] |
|---|---:|---:|---:|
| 0.50 | 0.5058 | 0.5093 | +0.0035 [-0.0115, +0.0185] |
| 0.75 | 0.5099 | 0.5137 | +0.0038 [-0.0139, +0.0215] |
| 1.10 | 0.5215 | 0.5301 | +0.0086 [-0.0030, +0.0201] |

두 후보 진단에서 0.50 대비 AUROC 변화는 0.75에서 +0.0044 [-0.0113, +0.0201], 1.10에서 +0.0208 [+0.0051, +0.0365]였다. Object–surface·첫 방향을 층화한 paired bootstrap으로 32개 seed block을 12,000회 재표집했고, 두 speed change에 simultaneous interval을 적용했다. 이 진단에서는 correction action과 GPU p90을 재측정하지 않았다. [계산 및 구간](binary37.json) · [분석 코드](binary37.py)
