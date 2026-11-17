# v42 사전 열람 전 geometry-only 보조 예측

Primary outcome 생성은 이미 시작했다. 이 보조 가설은 해당 outcome을 열어 보지 않고 per-family deformation 식과 일곱 측정 좌표만 사용해 만들었다. Primary preregistration과는 시점이 다르다. 예측 단위는 2 shape × 2 nominal height × 아홉 signed offset이다. 각 단위는 두 contact law·방향·세 regime·환경을 합쳐, 변형된 hollow·bulge만 평균한 **0% estimated−original Brier**의 부호로 채점한다. 같은 셀들의 평균 절대 차이 방향과 비영 offset에서의 부호 일치 수도 기록한다.

Bulge−base의 7점 profile 변위에 nominal center 기준 2차 곡선을 least squares로 맞춘다. 대칭 coordinate 때문에 1차 계수 s는 sum(y·d)/sum(y²)다. 이것이 original source quadratic이 볼 수 있는 단순화된 apparent slope다. 실제 contact 위치에서 bulge−base의 tangent 차이를 t라 둔다. 변형 정보를 완전히 없애면 0, 남기면 s로 추정한다고 가정할 때 두 tangent squared error의 차이는 t²−(t−s)²이다. 양수이면 projection이 해롭다고 forecast한다. 이 가정은 bank prior, base shape, measurement noise, offset에 따른 decoder의 전체 posterior, friction, stick 경계의 비선형성을 생략한다. 따라서 물리적 target label이나 Brier를 계산한 예측은 아니다.

Flare의 apparent slope는 19 mm에서 −0.06017, 31 mm에서 +0.04440이다. Roll은 각각 +0.06534와 −0.10246이다. 31 mm flare의 음수 offset에서는 실제 slope 변화가 appearance와 반대 부호다. 이 경우 projection이 유리하리라는 예측은 단순히 offset 크기만 보고 contour를 절대로 제거하지 않겠다는 규칙과 다르다. 반대로 같은 family·height의 양수 offset에는 양수가 나온다. [예측식](slope_forecast42.py) · [36개 방향 예측](slope_forecast42.json)
