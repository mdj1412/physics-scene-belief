# v47 사전 분석 계획

Target outcome 생성 전에 새 family, frozen rules, 비교 집합, 통계 단위, 판정 기준을 고정한다. Target label은 simulator가 별도 생성한다. v45, v46의 catalog는 정책 개발 자료로 허용한다.

## 실험과 가설

- 동일 물체의 7점 paid query profile에 같은 크기의 변형을 가하고 source projection axis와의 squared cosine만 0.1 또는 0.9로 바꾼다. 동일한 pre-impact geometry에서 별도로 loaded contact의 변형을 꺼 두거나 활성화한다. Optical intervention과 mechanical intervention의 네 조합을 전부 만든다. Physical change는 유한 pressure 및 tangential velocity로 stick/slip을 계산한다. Bin label을 직접 뒤집지 않는다.
- Keel, crown, channel, bead 네 synthetic contour는 local contact 주변을 맞추고 바깥 geometry를 바꾼다. 같은 catalog·물체·height·offset·variant·noise를 네 factorial cell에서 짝짓는다. Pre-impact x 및 normal-angle variance의 실제 차이, 진짜 optical axis의 squared cosine, inert/active에서 hollow↔bulge physical label flip을 각각 감사한다. Matched physical neighborhood로 인한 중복은 별도 family replicate로 집계하지 않는다.
- **예측 1.** Mechanically inert 조건에서는 source axis에 더 잘 정렬된 optical deformation을 제거할 때 저정렬 조건보다 상대 Brier가 낮을 것으로 예상한다. Sign, magnitude, family별 결과를 함께 본다.
- **예측 2.** 같은 visual evidence와 paid profile에서 mechanical coupling만 바꾸면 정책 선택이 동일하다. 그 사실만으로 두 조건에서 최적 행동이 다르다고 판단할 수 없다. 관측이 같은 조건에서 상대 Brier의 부호 또는 크기가 어떻게 달라지는지 측정한다. Pooled interaction 부호를 사전 가정하지 않는다.

## 평가 예산과 판정

- 12개 fresh catalog, 각 4 environment × 128 draws, 4 synthetic family, 2 sensor alignment, 2 height, 2 signed variant, 3 offset, 2 finite pressure radius, 4 paid-noise 반복. 2 surface × 2 law를 environment에 배정하고 catalog마다 순환한다. 비교 모델은 같은 factual evidence, deployment acquisition, 선택된 sensor와 source decoder axis를 사용한다. Target label, family·law·coupling 이름, true axis, deformation pair와 구매하지 않은 measurement는 deployable policy에 제공하지 않는다.
- Primary는 held-out Brier loss, negative difference가 새 정책에 유리하다. 각 catalog 안의 factorial cell을 동일 가중하고 12개 catalog를 동일 가중한다. 새 observation-only rule − always-original, 새 rule − frozen v46의 두 paired 차이에 Bonferroni simultaneous 95% t interval을 쓴다. 두 상한 모두 0 미만이면 두 baseline에서 개선됐다고 판단한다. 어느 구간이든 0을 포함하면 해당 비교에서 이득은 미확인이다. 실용적 효과 크기는 실물 비용으로 검증되지 않았으므로 별도 단위를 만들지 않는다.
- Mechanism은 forced projection−always-original의 네 cell과 (high−low)의 active−inert difference-in-differences, coupling별 symbol flip을 기록한다. 각 cell의 4개 CI를 함께 보정하고 interaction은 별도 진단으로 표시한다. Family는 고정한 네 조건이다. Artificially balanced coupling prevalence에 조건부인 값이며 자연계 prevalence를 뜻하지 않는다. Target label을 사용한 group oracle은 추가 진단 전용이고 정책 점수에서 제외한다.
- Secondary는 GPU p90. 같은 할당 GPU 한 장에서 warm, randomized paired order, 같은 512-object batch, 동기화, 실제 single-route decoder/selection 비용을 포함한다. 12개 catalog/block을 paired bootstrap으로 재표집한다. Offline rendering, simulator와 실물 sensing/reset은 별도다. GPU latency와 Brier를 다른 평가 단위로 보고한다.
- 새 방법이 v46보다 좋아도 always-original로 돌아가는 효과일 수 있다. Projection rate, v46와 route disagreement, original을 골라 절감한 loss와 새로 projection해 변한 loss를 분해한다. 가능하면 같은 projection 빈도로 무작위 배정한 기대 loss를 계산한다. 추가 분석은 exploratory로 표시한다.
