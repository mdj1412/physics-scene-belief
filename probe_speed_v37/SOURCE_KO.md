## 속도별 likelihood와 고정한 gate

0.50의 네 번째 접촉에는 v36의 동결 파일을 그대로 읽었다. 0.75와 1.10에는 높이 0.023 m에서 해당 속도로 움직였을 때의 joint observation likelihood만 새로 계산했다. 같은 v33 source family 4종, friction orientation 두 개와 amplitude regime 네 개에 같은 사전확률을 두고, v36 source seed 360021776에서 생성한 64개 component별 384개 latent draws를 사용했다. 각 component는 세 개의 noisy contour projection, factual velocity 두 성분과 probe velocity 두 성분에 Gaussian을 맞춘다. 고정된 simulator response로부터 component moments를 추정하고 sensor noise covariance를 더했다.

첫 세 접촉의 likelihood, signed proposal z, Brier gain lookup, 분포를 평균하는 방법, β=0.8289760825845313과 apply threshold 0은 v36의 값을 사용했다. 다른 속도에서 표면의 실제 regime나 target outcomes를 이용해 gate를 다시 fitting하지 않았다. 0.50에서 다시 생성한 모든 component의 mean·precision·log normalization은 원본과 원소별로 일치했다. v36 archive 한 개에서 최종 확률에 쓰는 z와 apply 결정, strength posterior도 동일했고 다른 두 속도에서 첫 세 접촉의 출력도 일치했다. [동결 기록](freeze37.json) · [구현](gate37.py) · [동결 코드](freeze37.py)

독립 source prefix 512개를 표면별로 대응시키고, 같은 물체와 noise에서 마지막 속도만 바꿨다. 이 source 진단은 v36의 source validation split을 그대로 사용한다. 강한 표면에 대한 식별력과 작은 weak/null 차이에 대한 식별력은 서로 달랐다.

| 마지막 speed | Local regimes의 slip 비율 | Strong 대 weak/null AUROC | Weak 대 null AUROC |
|---|---:|---:|---:|
| 0.50 | 58.9% | 0.588 | 0.528 |
| 0.75 | 80.1% | 0.611 | 0.535 |
| 1.10 | 94.2% | 0.643 | 0.542 |

Source에서 속도를 높이면 slip은 크게 늘었지만, weak/null의 strength ranking은 1.10에서도 0.542였다. Source model은 near strength가 1, 0.3, 0인 세 local regime와 globally isotropic인 경우를 가정한다. Source의 Brier gain lookup은 v34 conventional deployment probabilities에 query label을 교체해서 얻었으므로 global-null의 deployment sensor 변화는 이 표에서 근사한다. Held-out target을 직접 채점할 때는 각 regime의 실제 sensor와 query를 함께 생성한다. [Source 결과](source37.json) · [계산](source37.py) · [v36 gain lookup 설명](../query_matched_probe_v36/GATE_KO.md)
