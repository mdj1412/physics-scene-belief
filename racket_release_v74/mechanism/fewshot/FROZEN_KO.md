# 탐색적 소수 표본 threshold의 고정 규약

이 규칙의 형태는 A-B-S II soft의 전체 fast train과 official test를 사용한 사후 진단에서 영감을 얻었다. 새 independently blinded hypothesis로 해석할 수 없다. 이 문서를 생성하고 SHA-256을 기록한 뒤 아래의 소수 표본 정책을 실행한다. 두 split과 두 예산에 하나의 규칙을 사용하며 held-out 점수로 고르는 hyperparameter는 없다.

- v74의 공식 및 secondary prediction bank의 calibration IDs와 tape를 그대로 쓴다. Source-only 8.079025256745311 m/s cutoff 이상의 fast 입사를 사용하고 paid K=4/8 전체를 지불한다. 그중 target label $|s'|>0.20$ m/s 및 $s_{in}\ne0$인 fast calibration만 threshold를 식별한다. 입력 feature는 $|s_{in}|$ 하나다. 이 분석은 fast deployment에만 적용한다.
- 입·출사 부호가 다르면 reversal로 기록한다. 한 draw의 fast calibration에 reversal과 retain 두 종류가 모두 없으면(0개, reversal만, retain만) 항상 reversal을 적용한다. 두 종류가 있으면 고유 $|s_{in}|$를 오름차순으로 정렬하고 인접 두 **서로 다른 값**의 midpoint 전부와 −∞, +∞를 후보로 둔다. $|s_{in}|\le cut$이면 reversal이다. 유효 fast calibration의 0–1 error 최소, 동률이면 가장 큰 cutoff를 선택한다. 따라서 동률에서는 더 넓은 reversal 영역을 택한다. 같은 입력값을 가진 상충 label도 각각 1표다. Test feature나 label은 tie 결정에 쓰지 않는다.
- Fast test의 모든 연속 오차, $|s'|>0.20$인 test에서만 부호 정확도를 계산한다. 같은 부호인 elastic은 유지하고 다른 경우 v74와 같은 impulse-preserving ±0.20 m/s projection을 쓴다. 항상 reversal도 같은 방식으로 projection한다. 같은 alpha 제약의 범위 밖으로 나갈 수 있는 점은 본문 기준과 같다.
- ‘양쪽 class 관측’과 ‘유한한 경계로 calibration을 완전히 분리’한 draw를 따로 센다. 완전 분리는 cutoff가 수치적으로 정해졌다는 증거가 아니다. 두 class가 완전 분리될 때 실제 cutoff는 가장 큰 reversal 입력과 가장 작은 retain 입력 사이 어디든 낼 수 있으므로 그 폭의 중앙값도 기록한다.
