# v51 · 표면 교체와 impact 부호 전이의 관측 계약

각 deployment batch는 먼저 반대 material에서 얻은 calibration 4쌍을 받는다. Mosaic→nap과 nap→mosaic를 같은 비율로 배정한다. 매 catalog에서 contact law 이름이 같고 surface 이름이 다른 유일한 environment가 상대편이다. Session마다 κpre가 0, .25, .5, .75, 1 중 하나다. 교체가 일어나도 확률 1−g이면 κpost=κpre이고, 확률 g이면 κpost=1−κpre다. g는 **물체별 flip율이 아니라 session 전체에 적용하는 확률**이다. 각 active 물체의 post-replacement 2.0 N reference sign에 독립 Bernoulli(ε) flip을 가해 impact sign을 구성한다. κpost는 weak-to-reference 관계이고, ε는 reference-to-impact 관계다. Inert는 계속 inert다. 모든 cell에서 실제 surface index를 바꾸므로 g=0은 표면을 바꾼 뒤 polarity law가 유지되는 대조다.

배포 물체마다 하나의 gauge state, family·height·radius·offset·optical variant·noise와 하나의 중립/± PyBullet label을 쓴다. 한 환경의 128개 물체에는 같은 calibration prefix와 g의 session draw를 적용한다. Primary policy에 환경 이름·κ·g·ε·충돌 label을 제공하지 않는다. 독립 실행 audit에서는 12 catalog × 4 session × 128개 물체의 simulated deployment를 250회 반복한다. 매 반복마다 session당 prefix 하나와 물체당 binary label 하나를 쓴다. 각 환경은 250회 동안 모든 κ를 정확히 50회씩 받는다. 같은 물체와 calibration bank는 여러 조건에 재사용되므로 반복 250회를 새로운 material sample 수로 세지 않는다.

사전에 정한 비교는 ε=.25, g=0 대 ε=0, g=.25다. 두 조건의 단일 물체 평균 전달률은 같지만 후자는 128개에 공통인 drift event가 있다. Session Brier 차이의 variance와 p90, realized session에서 패배할 빈도 및 fleet당 평균의 변동을 비교한다. 실물에서 실제 impact 부호를 측정한 자료는 없으며, 입력으로 지정한 impact deformation geometry에 대해 PyBullet로 미리 계산한 binary outcome을 사용한다.
