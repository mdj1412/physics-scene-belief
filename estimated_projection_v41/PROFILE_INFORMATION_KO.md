# v41 projection이 깨끗한 local tangent 추정에 주는 비용

이 보조 계산에는 label이나 decoder output을 쓰지 않았다. 물체별 local profile을 세 계수 (offset, local tangent, curvature)의 quadratic으로 두고, 일곱 높이 좌표에서 독립이고 분산이 같은 Gaussian sensor noise를 가정했다. 높이의 단위는 2 mm다. 기본 곡선의 결정적 깊이는 빼거나 이미 알고 있다고 가정한다. 원래 설계 행렬을 X, rank-one nuisance projection을 P라 할 때, local tangent의 분산 배율은 (XᵀPX)⁻¹ 및 (XᵀX)⁻¹의 tangent 대각 원소 비율이다. 다른 두 계수도 함께 추정한다. 이 배율은 원래 geometry로부터 얻는 정보의 손실을 뜻한다. 변형이 존재할 때 생길 수 있는 bias 감소는 별도다.

| Shape | Height | Direction | Panel 중앙 tangent 분산 배율 [최소, 최대] | 평균 잔류 변형 길이 |
|---|---:|---|---:|---:|
| flare | 19 mm | oracle | 1.540× [1.540, 1.540] | 0.0% |
| flare | 19 mm | paired | 1.592× [1.465, 2.135] | 21.5% |
| flare | 19 mm | unpaired | 131.530× [34.728, 1752.619] | 98.4% |
| flare | 31 mm | oracle | 1.861× [1.861, 1.861] | 0.0% |
| flare | 31 mm | paired | 1.718× [1.143, 2.349] | 23.7% |
| flare | 31 mm | unpaired | 799.221× [232.277, 3700.259] | 50.8% |
| roll | 19 mm | oracle | 2.828× [2.828, 2.828] | 0.0% |
| roll | 19 mm | paired | 2.341× [1.601, 3.654] | 29.1% |
| roll | 19 mm | unpaired | 246.878× [96.654, 1700.968] | 96.8% |
| roll | 31 mm | oracle | 4.536× [4.536, 4.536] | 0.0% |
| roll | 31 mm | paired | 3.891× [2.598, 5.608] | 18.4% |
| roll | 31 mm | unpaired | 632.648× [264.665, 3557.023] | 100.0% |

계산은 geometry-only 센서 모델의 정밀도를 설명한다. 실제 Brier loss에는 factual evidence, contact law, query/probe 선택, model mismatch와 보정도 관여한다. 표의 비율로 Brier loss 변화를 추론할 수는 없다.

## Oracle 없이 추정의 재현성을 확인하는 방법

같은 target group에서 최초 16쌍과 나머지 16쌍의 PCA 방향을 독립적으로 계산했다. 추정한 방향에 반대편 16쌍의 paired differences를 투영하여 계산한 sample variance를 sensor noise만 있을 때의 분산 2σ²로 나누고, 두 fold를 평균했다. 1보다 크면 그 축에 noise 이상의 측정 변화가 있다는 뜻이다. 같은 변형 모드가 실제 동역학에 무관한지는 이 지표만으로 판정할 수 없다.

| Shape | Height | 평균 16-vs-16 split angle | Held-out variance / noise variance |
|---|---:|---:|---:|
| flare | 19 mm | 25.71° | 6.24× |
| flare | 31 mm | 24.07° | 5.17× |
| roll | 19 mm | 38.19° | 3.43× |
| roll | 31 mm | 19.75° | 6.71× |

[각 panel의 계수별 결과](profile_information41.json)
