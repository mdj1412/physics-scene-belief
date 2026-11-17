## Source-trained robust decoder와 네 보정의 정보 경계

V43의 새로운 object·surface·law family를 읽기 전에 v42에서 쓰던 두 shape family와 두 contact law만으로 학습했다. Source와 target에서 동일한 neutral prior와 K256 acquisition, 같은 16,384개 physics hypothesis bank 및 7점 profile/2점 velocity sensor를 쓴다. Source에서 당시 알려주었던 환경별 prior parameter를 decoder에 입력하지 않았다. 기존 v42 archive에서는 label과 source 환경의 보정량만 꺼내고, target과 같은 neutral bank에서 실제 선택된 sensor와 원래 확률을 다시 생성했다. [Source 생성 코드](source_data43.py) · [공통 neutral inference](inference43.py)

12개 source catalog panel × 24개 case × 2개 환경 × 128개 물체에서 73,728개 observation row를 만들었다. 한 row에는 하나의 실제 paid sensor noise realization이 있다. 동일 object draw가 다른 condition에서 다시 등장하므로 row 수는 독립 실험 수가 아니다. 세 friction regime의 명목 위치 결과와 각 regime의 비영 offset 여덟 개 결과를 보존했다. 새 표면으로 바뀌면 local tangent에 따라 label이 달라지는 source counterfactual을 직접 이용한다. Source feature·train/validation에는 target family 이름이나 target label이 없다.

**Robust residual.** 모든 입력은 원래 decoder가 받은 관측의 함수다. Base logit, 선택된 관측 종류, 명목 접촉 높이, factual profile의 상대 위치·충돌 후 속도, 실제 query를 구입한 경우의 7점 상대 profile과 여러 폭의 국소 기울기·곡률 차분, 실제 probe를 구입한 경우의 두 velocity 좌표를 넣는다. 선택하지 않은 sensor는 feature 함수가 접근하지 않는다. 네 번째 calibration response나 signed correction cue도 입력에 쓰지 않는다. 작은 tanh MLP와 linear skip이 원래 logit에 더할 residual을 출력한다. Source feature를 표준화하고 표준화된 값을 ±8로 자른다. 극단적 확률의 수치 처리를 위해 robust feature와 temperature logit의 입력 확률을 [0.00001, 0.99999]로 제한했다. 운영 original과 privileged의 0% 정책은 raw probability를 반환한다. [모델과 입력 함수](model43.py)

**세 단계의 source 분할.** Panel 0–7 안에서 flare로 fit해 roll을 채점하고 roll로 fit해 flare를 채점하는 두 방향의 평균 Brier로 hidden width 8·32 중 하나를 고른다. 정한 폭으로 두 shape를 합쳐 panel 0–7 전체를 다시 학습한다. 같은 물체·contour의 counterfactual은 공유될 수 있으므로 이 source 내부 shape holdout은 독립 물체 일반화의 시험으로 해석하지 않는다. 목표 Brier의 절반은 nominal outcome, 나머지 절반은 nonzero offset의 균등 혼합이다. Binary outcomes의 평균으로 squared error를 학습할 때 예측이 들어가는 gradient는 모든 outcome에 대해 Brier를 각각 평균한 경우와 같다. Brier를 보고할 때는 이 평균을 soft label로 간주한 squared error 대신 각 binary event의 기대 제곱오차를 계산한다. 같은 width·seed·epoch에서 nominal label만 쓴 ERM도 source 진단으로 평가한다.

Panel 8–9에서 각 decoder의 shifted outcome NLL을 최소화하는 양의 scalar temperature 하나씩을 적합한다. Panel 10–11에서는 세 단순 보정(0%, temperature, 고정 32.5%)의 shifted Brier를 비교해 row별로 한 가지를 고른다. 세 split의 catalog prototype은 서로 독립이다. 단순 보정 순위를 정할 때 source original의 0%와 fixed 입력은 저장된 clipped logit으로 재구성했으므로 0%의 확률 오차 상한은 0.00001이다. Target 평가에서는 세 decoder 모두의 raw probability에서 0%, fixed 및 optimizer를 직접 계산한다. [학습 코드](source_fit43.py) · [Source 검증](source_validation43.json) · [동결 모델](frozen_model43.json)

**v38 optimizer.** 첫 세 calibration response의 source log-density posterior에서 방향 cue z를 추정하고, 네 번째 response까지 반영한 source posterior로 원래 gain table을 평균한다. 0, 0.325, 1 및 table knot에서 가능한 weight 중 최대 gain을 고르고 tie에서 0.325에 가장 가까운 값을 선택하는 기존 함수를 그대로 호출한다. Fixed는 0.325를 쓰고 두 방식 모두 원래 β와 signed logit correction을 쓴다. Temperature는 전체 logit에 scalar를 적용한다. 0%, temperature, fixed도 네 번의 접촉 비용을 똑같이 지불하므로 v43은 구입 효용의 비교가 아니다. [v38 원본](../probe_value_v38/continuous38.py) · [Source 독립 검산](model_audit43.json)

### Source에서 동결한 수치

Source 73,728개 observation row는 서로 다른 panel·population·prototype 식별자 기준 11,924개 물체를 포함했다. Feature는 28개, 선택된 hidden width는 8이었다. 같은 모델에서 어느 source label을 학습했는지 바꾸면 결과가 달라졌다. Nominal-only ERM과 비교해 offset을 섞어 학습한 robust decoder는 별도 source 선택 panel에서 shifted Brier를 0.018811 낮췄고, nominal Brier는 0.004200 높였다. 이 차이에는 출현 빈도가 다른 future outcome을 선택한 효과가 포함된다. OOD target을 보기 전에 얻은 수치다.

| Decoder | T (panel 8–9) | Source 0% | Source temperature | Source fixed | Source optimizer | Source-best simple |
|---|---:|---:|---:|---:|---:|---|
| Original | 3.29564 | 0.159185 | 0.160170 | 0.159626 | 0.160272 | 0% |
| Robust | 1.25262 | 0.150106 | 0.150306 | 0.150545 | 0.151038 | 0% |
| Privileged | 3.61367 | 0.171703 | 0.167969 | 0.172182 | 0.172554 | temperature |

표의 source Brier는 panel 10–11의 nonzero offset 결과다. Temperature를 NLL로 고르고 다른 두 panel의 Brier로 단순 정책을 고르므로 같은 환경에서 두 값이 함께 최적화된다고 기대할 수 없다. 선택 검증에 사용한 독립 panel 수가 둘뿐이어서 이 순서를 모집단의 참 순위로 일반화할 수 없다. 0%·fixed의 source original 입력 logit이 clipping 경계에 있는 row는 전체의 28.59%였다. Original의 0%와 fixed 간 source 차이(0.000441)는 작은 clipping 수치 오차 상한보다 컸다. 12개 source score를 독립적으로 다시 계산한 최대 차이는 2.78×10⁻¹⁷이었다. 실제 미구입 sensor를 NaN으로 바꾼 별도 검사에서도 robust 확률은 bitwise 동일했다. [동결 hash](model43.sha256)
