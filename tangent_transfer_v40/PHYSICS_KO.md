# v40 · 같은 local tangent를 공유하는 contour와 unseen contact law

두 물체의 contact normal과 friction이 같을 때 센서가 보는 주변 contour만 달라져도 고정 32.5% shrinkage와 frozen optimizer의 순위가 달라질 수 있는지를 검증한다. v39의 shelf와 ridge를 각각 기본 곡선으로 삼고, 각 family에서 세 contour가 같은 latent 물체를 공유하도록 짝지었다. Base는 이전 형상 식을 재사용하는 control이다. Hollow와 bulge는 새 형상이다. 두 contact law의 식은 이 연구를 위해 설정한 synthetic law이며 측정된 재료 법칙으로 보정한 값은 없다.

| 설계 요소 | 정의와 검증 대상 |
|---|---|
| Contact 높이 | 첫 세 calibration probe와 deployment probe는 0 mm, 마지막 probe는 23 mm, 예측할 query는 25 mm에 닿는다. |
| Tangent family | 기본 shelf와 ridge는 서로 다른 local tangent를 갖는다. 같은 family 안에서는 세 contour의 위치와 tangent가 세 접촉 높이 모두에서 일치한다. |
| 주변 contour | Base의 추가 변위는 0, bulge는 +0.9 mm, hollow는 −0.9 mm의 두 quartic lobe를 각각 더한다. 첫 lobe는 17~23 mm, 둘째는 25~33 mm에 있다. |
| 관측 차이 | 19, 21, 23, 25, 27, 29, 31 mm의 query view에서 bulge − base 변위는 0.7111, 0.7111, 0, 0, 0.50625, 0.900, 0.50625 mm다. RMS는 0.5774 mm(센서 σ의 1.925배)다. Factual profile은 −10~+10 mm에 있어 추가 변위가 정확히 0이다. |
| Catalog | 하나의 catalog ID와 여덟 α, calibration/deployment split별로 2,048개 prototype을 독립 생성한다. 같은 catalog의 12개 조건은 같은 latent support, 물체 추출, sensor noise, QMC bank 및 probe 순서를 공유한다. 각 split의 support seed는 2,040,100,000 + 10,000 × catalog ID + 100 × α index + lane이다. Calibration은 lane 10, deployment는 lane 20이다. |
| 검증 단위 | 같은 family 및 law에서 contour 교체로 query stick label·paid velocity·factual evidence·선택 센서가 같고, 선택된 query view 값만 달라져야 한다. 독립 표본의 단위는 세 contour나 열두 조건을 모두 포함한 새 catalog panel이다. |

각 lobe에서 x = (y−left)/(right−left)를 [0,1]로 자른다. 추가 변위는 amplitude × 16x²(1−x)²이고, 높이 방향 도함수는 amplitude × 32x(1−x)(1−2x)/(right−left)다. 양 끝에서 값과 도함수가 0이므로 곡선이 C¹로 이어진다. 같은 family에서 물체 하나의 contact normal은 유지되지만 관측에 사용하는 주변 점들의 위치와 curvature가 바뀐다. 이 설계에서 contour의 효과는 off-contact vision과 decoder를 통한 효과다. 자유로운 접촉점 선택이나 여러 점이 동시에 닿는 rigid-body dynamics는 이 simulator의 범위 밖이다.

## 두 unseen contact law

Incoming tangential velocity를 v, s=|v|로 놓는다. 두 법칙 모두 static friction과 kinetic friction을 별도로 사용한다. 기본 마찰계수 μ와 숨은 방향 d∈{−1,+1}에 대해 공통 배율은 μ[1+d e A(v)]다. Height envelope e는 0 mm에서 1, 23 mm 이상의 strong·weak·local null에서 각각 1·0.3·0이며 그 사이에 cosine으로 이어진다. Factual contact의 global null은 모든 높이에서 0이다. 위 배율에 아래 각 계수를 곱해 static threshold와 sliding impulse를 얻는다.

| Law | Direction contrast A(v) | Static multiplier | Kinetic multiplier |
|---|---|---|---|
| Saturating | 0.18 tanh³(v/0.62) | 1.015 + 0.025 tanh²(s/0.58) | 0.88 + 0.075 tanh²(s/0.58) |
| Crossover | 0.20 tanh(v/0.32) tanh[(s−0.70)/0.16] | 1.035 − 0.020 exp[−((s−0.68)/0.24)²] | 0.94 − 0.055 exp[−((s−0.68)/0.24)²] |

양의 tangential velocity를 예로 들어 v=0.425, 1.10에서 saturating의 방향 대비는 +0.03793, +0.15145다. Crossover에서는 −0.16294, +0.19692다. Crossover는 0.70에서 방향이 뒤집히고 saturating은 원점에서 cubic으로 증가한다. 과거 weakening·strengthening과 달라지는 구체적인 조건이다. 실제 tangential velocity는 각 물체의 tangent에 따라 변하므로 이 수치는 형상별 Brier를 계산한 결과가 아니다. [생성 코드](physics40.py)
