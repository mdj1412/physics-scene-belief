# v45 사전 분석 계획

## 두 질문과 평가 순서

1. 동일한 frozen decoder와 동일한 위치→projection 규칙을 쓰되 접촉 좌표를 사전 영상에서 추정하면, always-original보다 held-out Brier loss가 낮은가? Oracle은 같은 규칙에 정확한 최초 접촉 좌표만 공급한다. Oracle의 결과도 나쁠 수 있다. 두 decoder 중 실제 target loss가 낮은 것을 고르는 oracle은 사용하지 않는다.
2. 최초 접촉 좌표·초기 속도·물체·표면·총 scalar normal impulse를 대응시키고 pressure 폭만 늘렸을 때 fold와 ripple 각각에서 projected−original Brier가 얼마나 변하는가? 두 형상의 효과가 다른지는 paired interaction으로 평가한다. 부호 반전은 두 조건의 D가 서로 다르고 양쪽 구간 모두 0을 제외할 때만 확인으로 쓴다.

기존 v44 label과 예측을 연구 설계에 쓸 수 있다. V45 held-out outcome을 만들기 전에 표본, router, 검정, pressure solver, image estimator를 동결한다. 서버 GPU에서 frozen posterior와 policy latency를 측정하고, 물리 및 통계는 같은 서버의 할당된 CPU에서 실행한다.

## 표본과 정보

- 새 12개 catalog, 각 4환경 × 128개 prototype 추출, fold/ripple × 2 surface × 2 law × 3 contour × 2 nominal height, 정확한 x = 19 또는 31 mm + {−0.5,0,+0.5} mm. Decoder sensor noise 4회. 같은 물체·evidence·영상 좌표를 세 pressure 조건에 공유한다. Camera 반복 수와 imagery 규격은 별도 estimator 명세를 따른다. 조건당 동일한 가중치를 쓴다.
- Pressure arm은 point control, narrow 원형 pressure profile(0.25 mm 반지름), broad profile(2 mm)이다. Primary 배포 평균은 finite narrow/broad를 균등 평균한다. Point는 진단이다. 실제로 matching된 위치·impulse·effective area와 label 전환율을 보고한다. Flat material/geometry negative control과 quadrature sensitivity를 검사한다.
- Primary projection axis는 과거 v44의 paired sensor profile로부터 nominal height별로 하나만 추정하고 동결한다. Target family/변형 방향은 읽지 않는다. Primary router는 과거 v43 point-solver의 조건별 Brier 차이를 family에 걸쳐 평균해 얻은 여섯 좌표만 사용한다. x를 가장 가까운 source 좌표로 양자화해 음수이면 projection한다. 이 표는 기존 privileged axis에서 나온 rule이므로 새 generic axis로의 전이도 포함한다. Secondary는 같은 rule에서 shape별 privileged axis만 허용한 진단이다.
- Always-original, nominal-only(잔차 위치를 항상 0으로 둠), oracle x, visual x는 같은 factual evidence와 선택된 paid sensor를 받는다. Within-height negative control은 동일 nominal height, shape, 물체에서 세 observation 위치의 visual route를 독립적으로 섞었을 때의 평균 손실이다. 세 위치를 열거하여 정확한 permutation expectation을 사용한다. 이 대조와 nominal-only가 영상의 구체적인 위치 정보가 기여했는지 구분한다.

## 추정·판정

Primary endpoint는 같은 catalog에서 Brier(visual)−Brier(always-original), Brier(visual)−Brier(oracle), Brier(visual)−Brier(nominal-only) 세 개다. 12개 panel을 독립 단위로 평균한 paired t interval을 구하고 세 비교에 Bonferroni simultaneous 95% CI를 적용한다. Oracle보다 열등하지 않다는 주장은 사전 정의한 noninferiority margin이 없으므로 하지 않는다. Shuffle과 point control은 보조다. 항상 projection한 경우도 원인 분석을 위해 기록한다. 절대 Brier, contact coordinate MAE/p90, bin·route agreement, projection 빈도와 camera 시간 범위를 보고한다.

D = Brier(projected)−Brier(original). Pressure 비교는 두 shape의 broad−narrow D 차이와 fold−ripple interaction 총 세 개를 한 family의 paired simultaneous 95% CI로 산출한다. 두 면적 × 두 형상의 단순 D는 네 비교로 보정한다. Narrow/broad label disagreement와 stick prevalence도 함께 기록한다. 수많은 충돌·noise 반복을 독립 표본 수로 해석하지 않는다. ±0.35 mm camera 좌표 bias에 대한 민감도는 같은 rule로 보조 채점한다. Online warm p90은 같은 idle GPU에서 512개 batch로, 영상 입출력과 선택/decoder 비용의 포함 범위 및 offline label/render 시간을 분리한다.

## 주장할 수 있는 범위

Camera가 pre-impact frame들을 보고 추정하는 위치 및 미래 query outcome을 같은 held-out 물체에서 연결한 simulation 실험이다. 실제 camera와 실제 pressure 측정은 별도의 외부 검증이다. Pressure 폭은 외부에서 제어하며 자유로운 재료 변형으로 자연 발생하는 patch를 주장하지 않는다. Presence of benefit은 primary CI로, 남는 이득의 크기는 absolute paired loss로 판단한다. 보조 분석을 보고 primary rule을 교체하지 않는다.
