# 접촉 위치를 고정한 pressure area 변화의 검증

12개 새 catalog에서 geometry·표면·속도·scalar normal impulse와 최초 접촉 위치를 짝지었다. 같은 radius 안에서도 위치의 local surface slope에 따라 실제 surface area가 달라지므로, 각 물체에서 곡면의 effective pressure area도 적분했다. Fold의 narrow/broad 평균은 **0.18095 / 11.65051 mm²**, ripple은 **0.18060 / 11.55776 mm²**였다. 두 면적비의 개별 물체 범위는 **61.35–68.89배**다. 두 arm에서 총 scalar normal impulse는 정확히 같은 배열(0.9097–2.0478 N·s)을 사용했고 pressure centroid는 매 순간 동일했다. Point control의 면적은 0이다. 총하중을 고정했으므로 분포 너비와 peak pressure는 함께 변한다. [개입 명세](PRESSURE_PROTOCOL_KO.md) · [면적·충격량 검산](pressure_audit45.json)

## 같은 비율로 event가 달라져도 Brier 차이는 다르다

D는 Brier(source-axis projection)−Brier(original)이다. 같은 물체에서 narrow→broad로 바꿔도 두 예측 확률은 그대로여서 Brier 효과의 변화는 −2(p_projected−p_original)(y_broad−y_narrow)의 평균과 정확히 일치한다. 따라서 어느 방향으로 event가 바뀐 항목이 D에 기여하는지도 직접 계산했다. 아래 기여도는 전체 물체를 분모로 계산해 두 열을 더하면 broad−narrow D가 된다.

| Shape | Stick→slip | Slip→stick | 전체 switch | Stick→slip의 D 기여 | Slip→stick의 D 기여 | 합계 |
|---|---:|---:|---:|---:|---:|---:|
| Fold | 2.010% | 1.473% | 3.483% | +0.002323 | +0.001415 | **+0.003738** |
| Ripple | 1.559% | 2.022% | 3.581% | +0.000563 | −0.000605 | **−0.000042** |

Fold에서는 두 전환 방향 모두 projection의 상대 손해를 키웠다. Ripple에서는 서로 다른 두 방향의 손익이 평균에서 상쇄됐다. Ripple의 평균 변화가 작아도 접촉 결과 자체가 같은 것은 아니다. Privileged-axis 대조에서 broad−narrow D는 fold +0.002102, ripple −0.000056이었다. [동일 물체에서의 분해 및 구간](pressure_audit45.json)

## 어떤 경로가 작용했나

사후 분석에서 pressure를 가중하는 위치별 normal과 material law를 각각 중앙값으로 고정하거나 지역값을 유지했다. Normal을 고정하면 local incoming speed도 그 normal에 맞춘 중앙값으로 쓴다. Material을 고정하면 중앙 위치와 중앙 접근 속도에서 한 번 정한 coefficient를 patch 전체에 적용한다. 각 항목은 동일 물체와 incident trajectory를 사용했다.

| 사용한 field | Fold broad−narrow D | Ripple broad−narrow D |
|---|---:|---:|
| 중심 normal·중심 coefficient | 0.000000 | 0.000000 |
| 중심 normal·위치별 coefficient | +0.000009 | +0.000001 |
| 위치별 normal·중심 coefficient | +0.003730 | −0.000059 |
| 위치별 normal·위치별 coefficient | +0.003738 | −0.000042 |

Fold에서는 patch가 넓어지면서 서로 다른 normal을 평균하는 과정만 남겨도 효과 대부분이 재현됐다. 이 모형이 근사하는 물리 경로에서는 friction field의 공간 변화만으로 같은 효과를 재현하기 어려웠다. 동일한 결론을 실물 friction law의 원인으로 확장하려면 contact pressure와 local normal을 독립적으로 측정해야 한다. [사후 factorial ablation 및 구간](pressure_mechanism45.json)

## 정밀도와 stick 정의

동일한 2,654,208개 물리 조건에서 quadrature를 12×6에서 24×12로 높였을 때 label은 956개(0.0360%) 바뀌었다. 변경 수는 point/narrow/broad 순으로 0/80/876이었다. 접선 속도 차이의 최댓값은 0.00387 m/s였다. Fold의 broad−narrow D는 정밀도를 높인 경우 +0.003650이었고, stick threshold를 0.010 또는 0.040 m/s로 바꾼 경우도 각각 +0.003866과 +0.003872였다. Ripple은 고정한 0.020 m/s에서 약 −0.00004였지만, 더 엄격한 0.010에서는 **−0.000138 [−0.000239, −0.000036]**, 0.040에서는 **+0.000247 [+0.000113, +0.000381]**였다. 두 구간은 여섯 비교에 대한 동시 95% 구간이다. 따라서 ripple에서 면적 효과가 항상 0이라고 해석할 근거는 없다. [수치 및 threshold 감도](pressure_mechanism45.json)

물리 모형은 prescribed pressure와 하나의 translational slip mode를 가정한다. 실제 탄성체의 모양과 compliance가 만나는 과정에서 형성되는 유동적인 압력이나 local partial slip을 직접 측정하지 않는다. 독립적으로 설계한 평면 및 마찰 변화 fixture와 전체 archive의 검산을 함께 보관했다. [Neutral control](pressure_neutral45.json) · [Archive checksum](pressure_outputs45.sha256)
