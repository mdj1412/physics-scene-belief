## 논문 기여 가능성

이 실험이 직접 평가한 양은 네 번이라는 고정 budget에서 **속도를 바꾸어 더 식별 가능한 physical response를 만들었을 때, 그 정보가 downstream decision loss로 얼마나 전달되는가**이다. Sticking 상태에서 일부 friction coefficient가 동일한 impulse를 만든다는 관측성 한계는 선행연구에 이미 제시돼 있다. 더 빠른 속도에서 slip이 늘었다는 결과만으로 새 방법의 기여를 주장하기는 어렵다. [Fazeli et al., §3](https://arxiv.org/pdf/1710.04979)

**+0.75에서 속도 변경 자체의 효과와 baseline 우위를 구분해야 한다.** Gate의 Brier가 +0.50에서 +0.75로 바뀐 양은 -0.00012 [-0.00072, +0.00048]였다. 같은 속도에서 32.5%를 대조한 이점이 +0.50보다 커졌는지 보는 차이의 차이는 -0.00019 [-0.00058, +0.00019]다. +1.10과 +0.50의 차이의 차이는 -0.00106 [-0.00151, -0.00061]였다. 이 값은 재질 순위의 AUROC와 독립적으로 확인할 수 있는 decision utility다. 원래 v36의 +0.50에서 Gate−32.5%는 약 +0.00037이었으며, 이번 독립 panel의 +0.50에서는 +0.00067였다. [v36 결과](../query_matched_probe_v36/REPORT_KO.md)

**어떤 결정이 어려운지도 분리할 수 있다.** Gate는 원래 확률과 전체 보정 확률 중 하나를 고르고, shrinkage는 물체마다 그 사이의 별도 확률을 낸다. 표면을 정확히 분류하더라도 두 endpoint 중 하나가 squared error를 가장 작게 만든다는 보장은 없다. 이 차이를 확인하기 위해 환경별 64개 deployment 물체의 label로 binary action을 고른 후 다른 64개로 채점했다. 이때 고정 32.5% 대비 Brier 차이는 속도별 -0.00150, -0.00151, -0.00155였다. 이 진단과 실제 gate의 차이에는 관측 가능한 정보, source–target 차이, 유한한 deployment label 수가 함께 들어간다. 다른 물체의 정답을 사용한 진단값을 배포 가능한 성능으로 쓸 수 없다.

고속에서의 보조 실험은 다음 가설을 구체화한다. 같은 frozen gain table이 0%, 32.5%, 100% 중 하나를 고르게 하면 +1.10의 Brier가 0.11107이고, 고정 32.5%보다 −0.00080 [−0.00122, −0.00037] 낮았다. 기존 binary gate와 비교한 차이도 −0.00040 [−0.00068, −0.00012]였다. +0.75에서는 고정값보다 낮다는 증거가 없었다. +1.10에서의 이득은 0.003보다 작고, 이 3-action 대조는 같은 archive에서 추가한 offline 분석이며 p90을 측정하지 않았다. [세 행동의 선택률과 paired interval](TERNARY_KO.md)

+1.10에서 강한 방향과 weak/null의 구별은 AUROC 0.571까지 좋아졌으나 weak 대 local null은 0.522였다. 같은 실험에서 약한 두 표면만 후보로 제한한 별도 likelihood ratio도 AUROC 0.530이었다. 세 regime 사이에서 gate 결정을 무작위로 다시 배정했을 때보다 실제 배정의 Brier가 약 0.00031 낮았고, 같은 세 표면에서 77.0%는 같은 행동을 골랐다. 높은 속도에서 얻은 추가 정보의 일부가 유용했으나 그 효과의 절대 크기는 작았다. [선택 효과 분해](ROUTING_KO.md) · [두 후보 판별](BINARY_KO.md)

+0.50에서 +1.10으로 갈 때 새로 slip한 36.1%가 전체 Gate−shrink 차이의 변화에 −0.00021을, 이미 slip이었던 60.1%가 −0.00081을 기여했다. 반면 weak와 local null의 물리적 구별 상한 증가는 주로 적어도 한 후보가 stick에서 slip으로 바뀐 물체에서 나왔다. 두 계산은 각각 실제 policy loss와 privileged sensor separation을 다루며 집단을 묶는 기준도 다르다. 정보 상한 증가율을 Brier 개선으로 그대로 읽을 수 없는 이유다. [집단별 손실 변화](SLIP_KO.md) · [속도 요인 분해](PHYSICS_FACTOR_KO.md)

**주장의 범위.** 네 조합에서 독립 seed를 사용했고 source contour·friction 함수와 target 함수가 다르지만, 세 local regime과 global-null이라는 후보 자체는 source와 target이 공유한다. Tangential velocity는 고정 부호의 세 수준만 시험했다. Late query의 방향과 sensor noise는 고정했으며, reference decoder는 population proxy coupling을 알고 있다. 속도 변경에 따른 slip 변화만의 causal effect를 실험으로 분리하지 않았다. GPU p90은 한 장치와 batch size에서 실측했다. 실제 표면에서의 최적 속도나 장비에서의 cycle time에 대한 증거는 아직 없다.

**다음 검증.** 동일 표면·물체에서 명령 속도를 무작위화하고 noisy velocity와 독립적인 slip 판정, 새 물체 충돌의 stick/slip을 함께 기록할 수 있는 실물 실험이 필요하다. 같은 네 번의 비용에서 현재 binary gate와 32.5%를 고정하고, 이번 보조 분석에서 얻은 3-action 정책을 미리 주방법으로 고정해 별도 물체·표면에서 재현한다. 직접 추정한 연속 friction posterior와 비교하고 3-action의 전체 p90도 측정한다. 속도가 달라지면 실제 sensor noise도 달라지는지 측정하고, 같은 재료 내 반복과 새 material session을 구분해 Brier interval을 낸다. 실물 자료 전까지는 이 실험을 속도와 정보, 예측 손실을 분리한 simulation 연구로 제시하는 것이 타당하다.
