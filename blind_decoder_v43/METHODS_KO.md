## 무엇을 숨겼고 무엇을 바꿨는가

예측 대상은 현재 센서로 보고 있는 외곽 자체가 아니라, 같은 물체가 이후 +0.425의 속도로 충돌했을 때의 stick 확률이다. 관측 중심은 19 mm 또는 31 mm이고, 실제 충돌점만 그 위아래로 이동한다. 세 decoder 모두 같은 관측을 받는다. Simulation evaluator는 외곽·표면·법칙·위치오차를 알지만 이 정보를 운영 예측기에 제공하지 않는다.

| 실험 단계 | 사용할 수 있는 정보와 비용 |
|---|---|
| Family 설계 | Source의 flare·roll 형상 및 saturating·crossover law에서 벗어나 fold·ripple object family, localized inlay·periodic woven surface field, rational·Gaussian-rate constitutive law를 새로 정의했다. 2×2×2 완전 교차. 세 수준의 contour variant에는 기존 이름 base·hollow·bulge를 재사용하되 새 basis로 변형했다. 같은 rigid point-impact solver와 latent coordinates를 사용한다. |
| Source 학습 | v42 source 12 catalog, 73,728개 row를 공통 neutral prior에서 다시 추론했다. Panel 0–7에서 2방향 shape holdout으로 hidden width를 고른 후 residual을 학습하고, 8–9에서 decoder별 T를 fit하고, 10–11에서 0%·fixed·T 중 단순 대안을 골랐다. T는 비영 offset의 NLL, robust residual은 nominal과 비영 offset의 Brier를 1:1로 가중한다. |
| Blind acquisition | 모든 환경에서 같은 neutral prior와 같은 source bank seed를 썼다. 각 환경에서 factual evidence로 128개 중 query view 115개, physical probe 13개를 구입했다. 채점 시 같은 설계에서 얻을 수 있었던 두 센서 중 미구입 값은 NaN으로 지운다. 환경별 실제 prior, family 이름, 실제 접촉점은 주지 않는다. |
| Original / robust | Original은 기존 16,384개 physics hypothesis와 센서 likelihood로 예측한다. Robust는 같은 base logit에 28개 paid feature에서 학습한 width-8 residual을 더한다. 같은 cutoff의 probability와 profile 기울기·곡률을 쓴다. 네 번째 calibration response는 residual 학습·예측에 들어가지 않는다. |
| Privileged | 같은 bank와 decoder에서 표준화한 query profile의 실제 변형 방향 하나만 제거한다. 그 축은 evaluator가 제공한다. 방향을 알 때의 진단 비교이며 실물에서 무료로 얻을 수 있는 정보로 보지 않는다. Probe branch의 likelihood는 같다. |
| 네 보정 | Zero는 같은 decoder의 p. Temperature는 source 하나의 T로 sigmoid(logit(p)/T)를 계산한다. Fixed와 optimizer는 같은 β와 첫 세 접촉의 signed cue z를 사용한다. 네 번째 response는 source gain table에서 optimizer의 보정 강도 w를 고르는 데 쓰인다. Fixed에서는 w=0.325다. |
| Target 채점 | 독립 12 catalog × 2 object × 2 surface × 2 law × 3 contour × 2 nominal height = 576개 archive. 매번 4개 환경 × 128개 물체, 센서 노이즈 4회. 물체·변형·센서를 대응시키고 9개 실제 높이를 채점한다. 같은 원형 물체를 서로 다른 조건에 적용하므로 반복을 독립 catalog로 세지 않았다. |
| GPU p90 | 동일 GPU에서 같은 512개 물체를 단 한 번의 sensor noise로 처리하는 12개 방법의 완전 crossover. Catalog별 1회 warm-up 후 3회 무작위 순회, 방법별 36개 기록. K256 acquisition, 선택한 sensor의 routing, posterior, 적용한 correction 및 CPU–GPU 전송·동기화를 포함한다. 물체를 움직이고 reset하는 시간과 offline source 준비 시간은 제외했다. |

V42와 달리 neutral prior가 acquisition과 decoding 양쪽에 적용됐다. 이전 GPU p90은 1,024개 물체 및 다른 acquisition 경로에서 측정했으므로 이번 512개 batch와 절대시간을 직접 비교하지 않는다. T는 original 3.296, robust 1.253, privileged 3.614였다. Source에서 선정한 최선의 단순 정책은 original 0%, robust 0%, privileged temperature였다. [학습 분할과 모델](MODEL_KO.md) · [물리 생성식](WORLD_KO.md) · [사전 판정 기준](PLAN_KO.md)
