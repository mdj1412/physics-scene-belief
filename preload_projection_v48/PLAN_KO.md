# v48 · Target outcome 생성 전 분석 계획

## 질문과 배포 정책

V47의 같은 pre-impact evidence로 구분할 수 없었던 unloaded/loaded deformation에 대해, query height 근처에서 1회 약한 preload를 가해 얻은 signed displacement가 projection의 상대 Brier를 예측하는가? Force는 제어 입력으로 고정하고 두 위치의 displacement 차이를 한 scalar로 읽는다. 같은 trajectory에서 동시 측정하는 두 fiducial은 두 번의 touch로 세지 않는다.

- Always-original은 공통 factual/paid observation을 받고 original posterior를 유지한다. 추가 preload는 0회다.
- Fixed probe는 source에서 유리한 fixed station 하나를 고른 뒤 모든 물체에 같은 힘으로 한 번 측정한다. 높이·sensor·budget·모델 용량을 맞춘 상태에서 query-station으로 위치를 맞추는 가치를 평가한다.
- Query-site preload는 query의 명목 contact height에서 같은 측정을 한 번 한다. Fixed와 query-site 모두 모든 물체에 1 touch와 1 reset을 지불한다. 비교 가능한 no-extra learned router를 source에서 함께 학습해 observation을 사는 순증분을 확인한다.
- 공통 decoder와 bank는 v47에서 고정한 original/projected 두 개다. 모든 방법에 같은 factual evidence와 K256의 구매 결과를 전달한다. 모델은 구매하지 않은 profile/probe, 상대 정책의 preload, unloaded/loaded variant pair, law/family 이름, true mechanical state, target label을 사용하지 않는다. 실제 비용은 필요한 sensor와 한 번의 선택된 decoding으로 측정한다.
- Source는 v47의 12개 catalog와 기존 4개 family다. Optical variant와 별도로 mechanical state를 0, 음, 양으로 교차하며 확률은 0.5, 0.25, 0.25이다. Signed signal과 outcome의 관계는 source에서만 학습한다. Target support 800–811, 새 ogee/cove/reed/flute 4개 contour, 19/31 mm, 2개 optical alignment, 2개 optical variant, 3개 offset, 2개 pressure radius, 4개 environment와 4개 noise를 평가한다. 19/31 station 사이에 imperfect mechanical correlation을 두며, downstream 결과는 이 prior에 조건부다.
- 예측은 query-station 신호가 fixed station과 같은 위치에서는 같은 물리를 보고, 다른 위치에서는 coupling·mechanical direction에 대한 정보를 더 제공한다는 것이다. 더 높은 식별력이 반드시 더 낮은 Brier를 주는지는 독립적으로 판정한다.

## 통계, 비용, 대안 설명

1. Lower-is-better held-out Brier를 같은 simulation case에서 직접 제곱오차로 계산한다. 각 catalog에서 mechanical state만 위의 비율로 가중하고 나머지 factorial dimension은 등가중한다. 새 policy−always-original과 새 policy−fixed probe를 primary 두 비교로 하여 12개 독립 catalog의 Bonferroni simultaneous 95% paired t interval을 사용한다. 두 상한 모두 0 아래여야 두 baseline을 이겼다고 판정한다. Family는 고정 평가 대상이다.
2. 같은 training set과 decoder 후보에서 no-extra router 대비 loss를 별도 paired contrast로 보고한다. 추가 관측 자체의 가치를 더 직접적으로 확인하기 위해 각 물체/기존 evidence/광학 noise는 유지하고, gate가 사용한 preload와 평가할 true mechanical state의 대응만 끊은 기대 Brier도 계산한다. Fixed site on/off query station, state·family·alignment별 loss와 route frequency를 별도 표시한다. Same-site readings가 같아도 두 모델을 별도 fit한 경우 policy 차이가 있을 수 있다.
3. Coupling(활성/비활성)과 mechanical direction(+/−, active에 조건부)을 별도 평가한다. Ground truth는 진단단에서만 쓰고 threshold가 필요하면 source에서 고정한다. 쉬운 binary 분류 결과가 실제 routing의 추가 가치라는 해석을 피한다. 관측기에 부여한 감도, 약한 하중과 본 충돌 시 변형의 연관성 및 station 간 transfer는 simulator의 가정이다.
4. Warm GPU p90은 공통 512-object batch의 acquisition과 실제 한 decoder 경로를 포함하여 같은 할당 GPU에서 randomized paired order로 재고 catalog를 재표집한다. 실물 장비가 없어 sensing·reset wall time은 측정 불가능하다. 실제 action 횟수와 가정한 force·actuator cycle 및 reset 시간으로 명시적 비용을 보고한다. GPU batch p90을 단일 물체의 지연시간으로 나누지 않는다. Probe당 허용 가능한 최대 Brier-equivalent 비용은 해당 비교의 실제 loss 감소로 계산하며, 같은 예산 둘 사이의 action 비용은 동일 동작 조건에서 상쇄된다.
5. 같은 위치에서의 비용, label quadrature, raw probability/route replay, source/target split, 동일 관측 예산을 검산한다. Target 결과를 읽은 뒤의 보조 분석은 별도 표시한다. 실물·cell별 일반화나 distribution-free 식별성 주장으로 확대하지 않는다.

부호 전달 가정에 대한 사전 민감도 분석으로, weak preload에서 관측한 ± state와 full-load collision state의 대응을 역전하는 matched replay도 한다. 같은 물체·noise에서 이미 계산한 두 gate를 교환하므로 재학습하지 않는다. 정방향/역방향 혼합 비율을 바꾸며 no-extra 및 fixed 대비 손익의 break-even을 구한다. 이 가정은 실제 부호 역전율의 추정치가 아니다.
