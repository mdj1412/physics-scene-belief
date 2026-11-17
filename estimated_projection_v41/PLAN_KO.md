# v41 사전 검증 계획

새 물체에서 서로 대응되는 noisy profile만으로 추정한 contour 변형 방향을 query decoder에서 제거한다. 그 뒤에도 과거 자료로만 만든 selector가 각 환경에서 v38 continuous gain optimizer와 fixed 32.5% 중 유리한 쪽을 고를 수 있는지 시험한다. Primary outcome은 실제 held-out stick event에 대한 binary Brier loss (p−Y)²다. 분석 단위는 결과를 본 뒤 변경하지 않는다.

**모집단과 정보.** V38~v40에 사용하지 않은 두 기본 곡선 flare·roll과 두 deployment query 높이 19·31 mm를 교차한다. 각 곡선에서는 같은 물체의 3개 변형(base, hollow, bulge)이 접촉 높이에서 같은 tangent를 갖는다. 기존 saturating·crossover contact law, 표면 방향, strong·weak·null near-field condition을 포함한다. V38의 보정 계수, source gain table, 네 번의 calibration과 마지막 +1.10/23 mm probe는 동결한다. Query 높이 변경에 맞춰 16,384개 inner bank 물리 가설의 관측 template과 예측 대상 사건은 해당 높이에서 계산한다. 각 조건에서 source acquisition은 25 mm 기준의 K256 정책을 공통으로 사용한다. 세 decoder는 동일하게 구입한 115개 query view와 13개 deployment probe를 이용한다. 같은 배치에서 fixed, 항상 optimizer, 관측으로 두 방법 중 하나를 고르는 selector를 평가한다.

12개 독립 catalog panel마다 같은 물체·noise를 모든 조건에 대응시키고 시작 방향이 다른 두 block을 둔다. 24개 cell(2 shape × 2 height × 3 contour × 2 law), 환경 8개, deployment 물체 128개, orientation·regime·noise 반복으로 집계한다. 통계 추론에서는 panel 안의 이 단위를 먼저 평균한다. 동일 물체의 반복, shape, height, law를 독립 표본으로 세지 않는다.

**방향 추정.** 별도 물체에서 같은 물체의 두 noisy profile이 어느 쌍에 속하는지를 제공한다. 한 panel·shape·height에 32쌍(64 profiles)을 사용한다. 접촉 정답, 실제 변형 진폭·부호, 법칙 이름, deployment label은 추정에 주지 않는다. 관측 noise로 표준화한 차이에서 첫 principal component를 구한다. Oracle은 simulator가 아는 변형 방향을 사용한다. 세 decoder는 original, estimated projection, oracle projection이다. 표본이 다를 때 방향의 부호는 식별할 필요가 없어 |cos θ|에 대응하는 principal angle을 쓴다. 독립 물리·관측 구현 및 4/8/16/32쌍 민감도, 짝을 깨뜨린 경우의 각도를 검산한다.

**주 검정.** Estimated projection 안에서 selector Brier와 (1) fixed, (2) 항상 optimizer Brier의 paired 차이를 구한다. 12개 panel, df=11, 두 대조를 함께 Bonferroni–t로 보정한 양측 95% 구간을 사용한다. 두 상한이 모두 0보다 작아야 '어느 한 정책을 항상 쓸 때보다 선택 이득이 남는다'고 판정한다. 0이 포함되면 미확정, 하한이 양수면 해당 대안보다 악화됐다고 기록한다. 두 감소량의 실용 기준은 기존 0.003이다. 각 individual shape/height/law의 최대 이득을 주 효과로 쓰지 않는다.

**Decoder의 기여.** 동일 fixed 32.5%에서 estimated−original, estimated−oracle을 두 별도 paired 대조로 묶어 동시 95% 구간을 낸다. Estimated−original 상한 <0이면 Brier 개선, estimated−oracle 상한 ≤0.0003이면 사전에 정한 oracle 대비 허용 손실 이내다. 0.0003은 실물의 비용 측정치가 아니라 이전 실용 기준 0.003의 10%로 둔 공학적 여유다. Oracle이 original보다 나으면 oracle Brier 감소 중 추정 방향이 회수한 비율과 panel bootstrap 구간을 보조 보고한다. Original과 oracle 내 selector 대비 항상 적용 정책도 표로 공개한다. 변형별 base probability 변화량, 추정 방향 오차, shape × height별 결과로 projection이 어디서 유효했는지 확인한다.

**비용.** 같은 유휴 GPU 한 장에서 original/oracle/estimated × fixed/optimizer/selector의 동기화된 online GPU p90을 잰다. 1,024개 물체의 같은 workload·observation budget을 유지한다. Selector는 실제 보정 선택까지 잰다. 표면별 direction fitting에 들어간 시간, 64회 passive profile 획득, projection caching을 포함하는 범위와 제외되는 실물 측정·reset 시간을 명시한다. Paired timing은 catalog panel을 함께 재표집해 p90 ratio와 95% 구간을 낸다. Estimated+selector 대비 estimated+fixed, original+fixed 두 기준을 공개한다. Estimated+fixed 대비 ratio의 상한이 1.02 이내면 기존 허용 overhead 이내다. 별도의 law/shape 이름을 요구하는 방법은 실사용 selector로 채점하지 않는다.

**보조 negative control.** 같은 64개 profile에서 두 번째 profile의 물체 대응을 무작위로 깨뜨려 PCA 방향을 구한다. 각 panel의 홀짝으로 사전 선택한 한 block에서 동일 물체·source bank, 32회 중 고정된 두 noise draw(0, 16), fixed 보정을 사용해 estimated·oracle·original과 Brier를 비교한다. 이 대조의 목적은 matched pair가 전달하는 정보의 추가 가치를 임의의 한 방향 제거 효과와 구별하는 것이다. 추가 timing은 주 9개 방법에 집중한다. 모든 주 예측 자료·모델·계획의 SHA-256을 v41 target outcome 생성 전에 서버에 남긴다.
