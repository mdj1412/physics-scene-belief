# Contact law 이름을 가린 보조 비교

주 실험 시작 후, target Brier를 확인하기 전에 추가한다. 주 forecast는 설계자가 지정한 contour와 law를 둘 다 알고 있다. 법칙 이름이 실제로 성패 예측에 필요한지 보기 위해 같은 tangent와 contour에서 두 law의 사전 Δ를 똑같이 가중해 하나의 예측으로 만든다. 두 경우 모두 그 평균이 음수면 optimizer, 그 외는 fixed를 사용한다. 이것은 contour 이름과 후보 law 두 개의 분포(1/2씩)를 아는 비교다. 정답, v40 관측, 실제 law 이름은 사용하지 않는다. 따라서 source 전체 평균 또는 tangent만 쓰는 기준과 다르다. 주 forecast 대비 12개 조건의 MAE와 같은 catalog에서 이 규칙으로 선택한 Brier를 pointwise 95% interval과 함께 보고한다. 주 실험의 확인 판정 및 실용 기준은 그대로 둔다.
