# v43 blind family holdout · 사전 판정 계획

V1–42·직전 temperature replay는 development 자료다. 이 파일, 생성식, frozen source artifact, 최종 runner의 hash를 첫 target 채점 전에 서버에 남긴다. 개발 단계에는 source label과 물리식의 수치 검산만 사용한다. Target score를 보고 generator·방향·보정·prior·가중치를 바꾸지 않는다.

**질문 A.** 새 object·surface·law family가 동시에 나타날 때 source-trained robust decoder의 frozen v38 optimizer가 해당 decoder의 source-fitted temperature보다 낮은 Brier를 내는가? **질문 B.** 보정 0%에서 privileged projection−original의 Brier 차이가 nominal에서는 음수, nonzero placement error에서는 양수인가? A에는 각 nonzero signed offset을 균등 가중한다. B에는 같은 물체·factual prefix·노이즈·확률 아래 두 label 조건을 짝짓는다. Robust·original·privileged 각각에 zero, temperature, fixed 0.325, v38 optimizer를 교차한다. 추가로 source 검증에서 정한 최선의 단순 보정과 optimizer 대조를 보조 보고한다.

**정보 경계.** 실제 세 family 이름, 법칙 계수, true nuisance amplitude, true offset, true contact label, target별 평균 loss 및 환경별 prior-alignment parameter는 운영 decoder·correction 입력에서 제외한다. Source 기준 prior를 acquisition과 decoding 양쪽에 적용한다. 실험 약정상 알려진 명목 높이, factual sensor 및 고른 query/probe, 네 번의 calibration은 사용 가능하다. Privileged row만 실제 변형 방향을 받는다. 해당 방향은 좌표를 지우는 데만 쓰며 실제 접촉점, label 또는 law를 제공하지 않는다. 모든 조합은 같은 네 번의 접촉을 지불한다. 측정 시 correction이 그 중 일부만 읽을 수 있다. 관측 구매의 순효과는 이번 추정량에 포함되지 않는다.

**설계와 단위.** 두 새 object, 두 새 surface, 두 새 law family를 완전 교차하고 나타난 조합을 균등 가중한다. 19·31 mm와 base·hollow·bulge 및 signed offset 0, ±0.125, ±0.25, ±0.5, ±1 mm를 사용한다. Catalog panel이 독립 난수 단위다. 환경/물체·반복 noise·offset 또는 같은 family를 독립 family 표본으로 계산하지 않는다. Panel 안에서 각 설계 요인의 동일 가중 평균 후 paired panel inference를 적용한다. 여덟 family 조합의 부호와 하나의 family 수준을 뺀 결과를 함께 내고 일반화 범위를 이 family 집합에 한정한다.

**판정.** A, B의 nominal 차이 D0, shifted 차이 D1, D1−D0의 네 paired contrast에는 Bonferroni 동시 양측 95% t 구간을 쓴다. A의 상한 <0이면 통계적 우세, <−0.003이고 p90 ratio의 paired 95% 상한 ≤1.02이면 이전 연구에서 정한 비용과 실용 개선까지 충족한다. A의 하한 >0이면 temperature 우세다. Projection 손익 반전은 D0 상한 <0, D1 하한 >0, interaction 하한 >0을 모두 요구한다. 하나만 충족하면 부분 지지 또는 미결정으로 기록한다. 각 decoder의 nominal/nonzero Brier, 9개 signed offset에서의 geometry 차이, robust−original, 각 구성의 GPU p90과 source에서 선택한 단순 정책의 결과는 보조값이다. Geometry 차이를 temperature에서도 재검사한다.

**비용과 검산.** 한 장치 내에서 12개 조합의 실제 online path를 같은 deployment batch에서 순서를 섞어 측정한다. GPU 동기화와 warm-up을 포함하고 GPU p90은 사전 계산 확률에 correction만 적용한 시간과 구별한다. 여러 GPU에서 측정하면 장치별 결과도 낸다. Ratio CI는 같은 순회 또는 같은 panel을 함께 bootstrap한다. 실제 robot 접촉과 reset, source 학습·사전 bank 생성은 온라인 시간과 별도다. 오프라인 분석이 실제 지불한 관측만 읽는지, unchosen sensor와 metadata를 바꿔도 운영 prediction이 같은지, 직접 계산한 Brier와 저장된 score가 같은지 확인한다. GPU 종료 후 사용 상태를 확인한다.

**결론의 범위.** Family 조합을 바꿔도 rigid impact solver는 공유한다. 등록된 direction을 알려주는 방법은 진단 대조군이다. Source에서 learned decoder가 우수해도 실제 마찰 파라미터를 복원했거나 물리적 접촉 비용을 절감했다는 결론은 나오지 않는다.
