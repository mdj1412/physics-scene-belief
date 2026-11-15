## 같은 gate에 conventional fourth와 query-matched fourth를 각각 제공한 결과

원래 v35의 conventional reader와 새 near gate의 차이에는 방향 추정 방식의 변화도 들어간다. 여기서는 두 gate가 같은 frozen source model, 처음 세 접촉의 signed proposal, 같은 β를 쓴다. 네 번째 conventional response는 실제 부호에 맞는 action likelihood로 갱신하고, near response는 query 부근 action의 likelihood로 갱신했다. 표에서 on 비율의 단위는 환경이다.

| 평가 범위 | Gate on, prefix3 | Gate on, conventional4 | Gate on, matched4 | 두 fourth의 결정이 달랐던 비율 | Far gate4 Brier | Near gate4 Brier |
|---|---:|---:|---:|---:|---:|---:|
| 세 local 균등 | 6.8% | 16.0% | 35.1% | 33.9% | 0.11289 | 0.11164 |
| 강한 방향성 | 6.8% | 16.0% | 37.3% | 35.0% | 0.12007 | 0.11767 |
| 약한 방향성 | 6.8% | 16.0% | 35.0% | 33.8% | 0.10957 | 0.10854 |
| query 부근 방향성 없음 | 6.8% | 16.0% | 33.0% | 33.0% | 0.10903 | 0.10872 |
| 전체 표면 방향성 없음 | 21.9% | 23.8% | 47.7% | 44.9% | 0.10640 | 0.10617 |
| 네 regime 균등 | 10.6% | 18.0% | 38.2% | 36.7% | 0.11127 | 0.11027 |

| 평가 범위 | Near4 − far4, 각자의 decoder | 네 번째 측정의 difference-in-differences | Near4 − far4, 같은 near decoder | Near4 − far4, 같은 far decoder |
|---|---:|---:|---:|---:|
| 세 local 균등 | -0.00125 [-0.00211, -0.00039] | -0.00109 [-0.00163, -0.00054] | -0.00112 [-0.00168, -0.00057] | -0.00109 [-0.00162, -0.00056] |
| 강한 방향성 | -0.00240 [-0.00354, -0.00127] | -0.00258 [-0.00328, -0.00187] | -0.00260 [-0.00331, -0.00189] | -0.00256 [-0.00320, -0.00191] |
| 약한 방향성 | -0.00103 [-0.00195, -0.00010] | -0.00075 [-0.00134, -0.00016] | -0.00079 [-0.00140, -0.00018] | -0.00074 [-0.00134, -0.00015] |
| query 부근 방향성 없음 | -0.00031 [-0.00113, +0.00050] | +0.00007 [-0.00060, +0.00073] | +0.00003 [-0.00065, +0.00071] | +0.00004 [-0.00063, +0.00070] |
| 전체 표면 방향성 없음 | -0.00023 [-0.00104, +0.00058] | -0.00012 [-0.00066, +0.00042] | -0.00013 [-0.00067, +0.00041] | -0.00010 [-0.00064, +0.00045] |
| 네 regime 균등 | -0.00099 [-0.00176, -0.00023] | -0.00085 [-0.00135, -0.00034] | -0.00087 [-0.00139, -0.00036] | -0.00084 [-0.00134, -0.00034] |

같은 near probabilities에 적용했을 때 세 local regime 평균은 -0.00112 [-0.00168, -0.00057]로 matched가 유리했다. Difference-in-differences는 -0.00109 [-0.00163, -0.00054]로 matched가 유리했다. 양쪽에서 네 번째 probe를 제공했을 때의 변화량을 각각 구했으므로, 이 지표는 처음 세 번만 읽은 정책에 대한 단순 우위를 넘어 어느 네 번째 동작이 더 유용했는지 평가한다.

세 local regime은 conventional 접촉 지점의 law를 공유한다. 실제로 같은 orientation·block에서 conventional fourth까지 사용한 gate decision이 세 regime 모두에서 같음을 모든 archive에서 확인했다. 그럼에도 conventional fourth가 첫 세 번의 방향 추정을 보완하거나 global-null을 구별하는 데 쓰일 수 있다. 각자의 decoder를 사용한 Brier에는 deployment action이 달라진 효과도 포함된다. 같은 decoder의 두 열에서는 같은 물체의 같은 sensor realization과 같은 proposal을 사용한다.

각 범위의 네 contrast에는 object × surface와 시작 부호를 층화한 paired bootstrap 12,000회의 simultaneous 95% interval을 사용했다. 새 conventional gate는 GPU가 저장한 관측과 prediction으로 CPU에서 평가했으며, 이 방법만의 GPU p90은 미측정이다. [보조 분석 계획](SUPPLEMENT_PLAN_KO.md) · [계산 코드](supplement36.py) · [수치](supplement36.json)
