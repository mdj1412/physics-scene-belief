# V41 사전 보정 선택식

V38이 산출하는 signed cue z와 보정량 w, fixed 32.5%를 그대로 사용한다. 각 환경에서 calibration 접촉 4회와 128개 별도 deployment 물체에 할당된 센서 결과가 도착한 뒤 **그 환경의 모든 물체에 같은 보정 방법 하나**를 택한다. 따라서 한 물체씩 도착하는 online setting에 대한 평가는 아니다. Sensor noise를 32회 반복하는 평가는 각각 독립적인 선택으로 다시 계산한다. 선택할 때 다른 noise 반복의 관측을 볼 수 없다. 실물의 형상 이름·마찰 법칙 이름·물체별 stick label은 입력에 없다.

두 확률을 q_f, q_o라 하면 Δ=(q_o−q_f)(q_o+q_f−2Y)다. a=0.828976·z·(w−0.325), m=(q_o+q_f)/2, h=(q_o−q_f)/a다. h는 a=0에서 q_f(1−q_f)로 연장하며 수치 오차를 제외하면 음수가 아니다. 환경 내 exact Brier 차이는 −2a·mean(h)·r이고, r은 h로 가중한 실제 잔차 Y−m의 평균이다. V41의 Y를 사용하지 않고 과거 자료에서 r을 ridge regression으로 학습했다. 새 환경에서는 128개 물체의 base probability와 측정된 접촉으로 r을 예측해, 예상 Δ가 음수이면 optimizer, 나머지는 fixed를 택한다.

입력은 전체·query·probe branch의 base probability 평균과 차이, h 가중 중간 확률·불확실성과 probe 비중, z·w·네 번째 측정 및 첫 세 방향별 측정 요약이다. 자료가 드문 입력은 source에서 표준화한 −5~+5 범위로 제한했다. 행 하나의 학습 가중치는 |a|·mean(h)에 비례하므로 손실 변화가 미미한 예제의 영향이 작다. GPU p90에는 각 배치에서 이 통계와 선택을 구하는 시간을 포함한다. 이 선택식 자체는 모든 decoder에 동일하다.

Source는 v39의 두 law에서 얻은 원래 관측과 v40의 두 law에서 얻은 원래 관측·oracle projected 관측이다. V40 projected branch는 v40 archived 물체·noise·calibration·법칙으로 별도 재생했다. 각 환경에서 사전에 정한 4개 독립 평가 noise를 사용했다. V39와 v40 law를 같은 비중으로 놓고 v40 안에서 original/projected를 동일하게 가중했다. 두 feature set(분포·cue 요약 / 여기에 접촉 속도와 교호작용 포함), 세 ridge penalty(0.01, 0.1, 1)를 후보로 두었다. 역사적 v40에서 shape family 전체 또는 law 전체를 가리면서 같은 panel parity도 fitting에서 제외했다. 이 두 검증과 두 decoder의 평균 Brier 차이가 가장 작은 후보를 골라 모든 source에 다시 fit했다. Historical validation은 feature·penalty 선택에 사용됐으므로 독립 target 평가와 구별한다.

기준은 각 decoder의 항상 fixed 및 항상 optimizer이며, 선택 규칙과 source coefficient는 새 target label 생성 전에 봉인한다. Calibration, 추가 geometry 표본 수, 128개씩 결정하는 deployment, 알려진 probe mode를 구현할 수 있어야 같은 운영 의미를 가진다. [구현](selector41.py) · [과거 자료 fitting](fit_selector41.py) · [계수](selector_model41.json) · [Source 검증](selector_source41.json)
