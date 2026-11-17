# v65 · 구매 선택과 계측 오류의 계산 검증

두 조명의 상위 router가 같고, 두 정책이 공유하는 deployment prefix 및 frozen decoder가 같으면 terminal purchase recommendation이 다를 수 없다. 모든 calibration session을 분모에 유지하면서, 이를 이용해 두 추천 간 불일치율을 재현하고 불일치 추정에 필요한 prefix 예산을 계산한다. 처음 feasibility pilot에서는 모든 prefix를 기록한다. 별도 대안은 상위 불일치 전수 및 독립 무작위로 선택한 일치 사례 5%의 prefix를 수집한다. 이는 전환율 전용 절약안이다. 실제 접촉 실행, Brier 또는 marker truth 수집 예산은 이 shortcut으로 자동 절감되지 않는다.

V64의 저장된 raw fixture에서 같은 조명·같은 영상을 복제하고, 한쪽 photodiode ADC에만 ±1, ±4, ±8, ±16 sample 시간 오프셋을 가한다. ADC 16 kHz, 경계는 끝값 연장, cyclic wrap은 허용하지 않는다. ±는 두 방향의 지연 주입이다. 0 offset에서 원본과 분류 code가 정확히 일치해야 한다. 각 오프셋에서 clip 부호/보류·upper policy·shared-prefix terminal 선택의 변화량을 보고한다. 이 fixture 결과를 실제 hardware nuisance 빈도 또는 허용 오차 추정치로 쓰지 않는다. 별도 독립 sham에서 0회 사건일 때 현장 false switch 상한이 p/2, p/5보다 낮아지는 표본 수를 계산한다. 장치가 없어 계산과 software replay만 수행한다.
