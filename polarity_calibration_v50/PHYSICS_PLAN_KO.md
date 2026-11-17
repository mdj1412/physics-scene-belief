# v50 geometry와 contact solver 동결 기록

원래 source의 keel·crown·channel·bead와 v48/49의 ogee·cove·reed·flute를 제외하고 새 네 contour, saddleback·terrace·swale·scallop을 생성한다. 가족별로 서로 다른 C² 비대칭 wing, 미리 지정한 부호별 quadratic/harmonic bending, 새 광학 profile 잔차를 갖는다. 두 query 위치 19/31 mm 인근의 unloaded profile은 family 간 동일하게 맞췄다. 두 surface field인 mosaic(비대칭 두 국소 insert)·nap(방향성이 달라지는 smooth transition)는 material scale과 orientation multiplier에 모두 들어가고, v49의 두 onset-rate 함수를 유지한다. 실제 재료 계수는 아니다.

12개 catalog마다 서로 다른 네 환경의 object prototype을 lane별로 복원 없이 추출한다. Deployment support ID는 900–911과 lane 20, calibration support ID는 920–931과 lane 10이다. 환경 내 object distribution의 상위 변수인 alpha만 공유한다. Trial noise 역시 다른 namespace를 사용한다. 평가의 독립 단위는 catalog이며 factorial cell·sensor repeat는 같은 대상의 조건부 관측이다.

PyBullet external solver에서 full-load signed state별 stick label을 계산한다. v49의 fine 설정(0.0125 ms, 0.025 mm mesh, 80 iterations)을 primary, ultra(0.00625 ms, 0.0125 mm, 160 iterations)를 sensitivity로 고정해 전 catalog를 반복한다. 각 full state의 contact label은 weak/full 유지율이나 정책과 별개로 한 번만 계산한다. Endpoint는 첫 충돌 tangent 방향의 속력이 0.020 m/s 이하인 경우다. 같은 수치의 약하중·강하중 상태를 잇는 방향 전환율은 실험자가 정한 regime이며, 미세 탄성·마모·reset 변화를 독립 solver가 생성하는 구조는 아니다.
