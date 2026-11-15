# v40에서 target label을 생성하기 전에 기록한 예측

Δ는 같은 물체·관측으로 계산한 **frozen v38 optimizer의 Brier − fixed 32.5%의 Brier**다. Δ<0은 optimizer의 이득, Δ>0은 고정값의 이득이다. 아래 부호는 독립 target catalog 전체의 조건별 평균에 대한 예측이다.

| Tangent / contour / law | 사전 Δ | 예상 우세 정책 | Historical 95% interval |
|---|---:|---|---|
| shelf / base / saturating | +0.000394 | fixed 32.5% | [+0.000194, +0.000616] |
| shelf / base / crossover | +0.001200 | fixed 32.5% | [+0.000943, +0.001465] |
| shelf / hollow / saturating | -0.000667 | optimizer | [-0.000923, -0.000391] |
| shelf / hollow / crossover | +0.000207 | fixed 32.5% | [-0.000068, +0.000489] |
| shelf / bulge / saturating | +0.001214 | fixed 32.5% | [+0.000992, +0.001461] |
| shelf / bulge / crossover | +0.001939 | fixed 32.5% | [+0.001637, +0.002239] |
| ridge / base / saturating | -0.000271 | optimizer | [-0.000488, -0.000064] |
| ridge / base / crossover | +0.000468 | fixed 32.5% | [+0.000219, +0.000722] |
| ridge / hollow / saturating | +0.000302 | fixed 32.5% | [+0.000138, +0.000459] |
| ridge / hollow / crossover | +0.000963 | fixed 32.5% | [+0.000762, +0.001171] |
| ridge / bulge / saturating | -0.001160 | optimizer | [-0.001477, -0.000858] |
| ridge / bulge / crossover | -0.000362 | optimizer | [-0.000708, -0.000028] |

두 법칙에서 모두 **shelf/bulge와 ridge/hollow는 고정값**, **ridge/bulge는 optimizer**를 예상한다. Shelf/hollow와 ridge/base는 saturating일 때 optimizer, crossover일 때 고정값으로 순위가 바뀔 것이라 예측한다. Shelf/hollow/crossover는 historical interval도 0을 포함하므로 특히 불확실하다. 12개 모두 방향 확인·반증·미확정의 분모에 남긴다.

### 어떤 근거로 contour와 contact law를 연결했는가

v39에서 수집한 4개 shape × law, 조건당 독립 catalog 12개와 2개 block의 관측·정답만 사용했다. 새 tangent family의 기본 곡선은 각각 v39 shelf·ridge와 같다. Hollow·bulge를 만들면 기존 factual profile과 실제 접촉 위치·기울기가 일정해 기존 label, 첫 세 접촉, 네 번째 접촉을 그대로 쓸 수 있다. Query view의 7개 측정값에 더해지는 변위만 약 ±[0.711, 0.711, 0, 0, 0.506, 0.90, 0.506] mm다. Historical law 아래 이 변위가 미치는 영향을 frozen physics decoder에 직접 replay했다. Base 대비 contour 차이는 각 historical block에서 128개 중 일정 간격으로 택한 32개 물체, 32개 noise realization 중 2개에 대응시켜 추정했다. 이들 중 query view가 배정된 물체만 확률이 달라진다. Historical base loss 자체는 전체 archive로 계산했다. 모든 재생에서 무변위 확률이 저장된 archive의 해당 확률과 일치하는지도 확인했다.

새 law에서 물체의 정답을 새로 만들지 않고, v39에서 이미 관측한 한 물체의 두 방향에 대한 stick label만 사용했다. 동일 물체·동일 표면 강도에서 방향을 유지한 관측 label을 Y₊, 반대 방향의 관측 label을 Y₋로 표기하자. Nominal query의 tangential velocity +0.425에서 새 law와 과거 law의 signed directional friction amplitude 비 ρ를 [−1,1]로 제한했다. Historical replay의 기대 Δ는 기존 방향의 loss 차이와 반대 방향 label에 대한 loss 차이를 각각 (1+ρ)/2 및 (1−ρ)/2로 가중한 값이다. Binary outcome의 두 예측 간 Brier 차이가 label에 대해 선형이기 때문에 두 실제 loss를 혼합해도 같은 Δ가 된다. V39의 두 contact law를 같은 비중으로 평균했다. Saturating은 query에서 방향성이 약해지고, crossover는 방향이 뒤집힌다는 명세가 이 추정에 들어갔다. 추정식이 새로운 충돌 결과를 결정하지는 않는다.

이 수치 예측에서 가장 취약한 가정은 source에서 측정한 세 접촉의 cue와 네 번째 접촉 후 선택한 보정량이 새 law에서도 대표성을 갖는다는 것이다. 실제 local tangent에 따른 tangential velocity 분산, static·kinetic multiplier 변화, 새 접촉 뒤 optimizer가 달리 고를 보정량은 반영하지 않았다. Source 95% 구간은 두 historical law 안에서 catalog를 독립 재표집해 얻었다. 새로운 law에 대한 prediction interval로 사용하면 불확실성을 과소평가한다. Direction의 사전 확신은 historical interval이 0을 가로지르는지와 cross-law 외삽의 한계를 함께 판단해야 한다.

**비교 기준.** 모든 조건에 같은 v39 전체 평균, shape의 v39 평균만 쓰는 tangent-only forecast, contour를 무시하고 law 방향만 이전한 forecast를 함께 동결했다. 네 방법의 12개 조건 동일 비중 MAE를 비교한다. 상세 forecast가 가장 유리한 baseline을 이기는지, 그리고 승리·패배를 각각 어느 조건에서 확인하는지가 주 질문이다. [사전 수치와 과거 catalog별 추정치](forecast40.json) · [계산 코드](forecast40.py)
