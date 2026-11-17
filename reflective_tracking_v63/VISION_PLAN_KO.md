# v63 영상과 독립 판독 사전 명세

24개 새 catalog의 48개 synthetic reflective surface를 새 synthetic board 8개 각각에서 교차하고 표면당 256개의 8-impact pilot을 만든다. LED pulse .80 ms, shutter exposure .25 ms, 1,000 fps의 8개 frame과 v62의 lag schedule을 고정한다. 표면별 난수로 반사 밝기·위치·ghost 비율과 계측기 결합 강도를 새로 뽑는다. 기준 2.0 N 접촉 8회와 impact 8회, mechanical reset 16회, 전자 reset 4회에 추가 취득은 없다.

Impact별 단발성 tracking error와 지속적인 ghost track의 발생률을 각각 0 또는 0.35로 무작위 배정한다. 서로 다른 독립 uniform draw로 두 사건을 부여하며 같은 물체와 noise를 네 조건에서 공유한다. 한 번의 tracking error는 임의 한 frame에서 상부 marker patch의 3 pixel 이동이다. Coherent error는 fixture 상대변위와 반대로 움직이는 specular ghost를 상부 patch에 중첩하여 일관되게 잘못된 track이 강하게 보이게 한다. 동시 발생에서는 coherent layer를 만든 뒤 한 frame만 이동한다. Generator는 이 두 교란을 만들 때 참 부호를 읽지 않는다.

독립 검증은 표면 뒤편에 부착한 별도의 differential strain pickup이 contact 기준의 전후 파형을 측정한다고 가정한다. 영상과 같은 잠재 물리 방향·변형 크기에 반응하지만 board·surface별 coupling, drift, 전자 noise, dropout과 quantization을 별도 난수에서 생성한다. Pre/post 차이를 기준으로 sign을 읽고 불분명하면 결측 처리한다. Simulation truth와 sensor 검증 범위·정확도, sensor와 camera 간 오류, transient·coherent 사건 실제 빈도를 각각 보고한다. 검증 파형·simulator truth·사건 mask·board·surface 이름을 정책 입력에서 제외한다.

같은 영상·waveform·electronics에 기존 v61 reader와 frozen v62 confidence reader를 적용한다. Reference와 impact 양쪽을 읽어야 한 optical mismatch 판독으로 친다. 판독한 수와 optical 오류는 실제 활성인 물체 쌍에서 집계하며, sensor와 camera가 함께 읽은 범위에서도 검산한다. P/C 및 deployment regime으로 mirror한 네 경로는 paired counterfactual로 보존한다. Surface별 optical material은 물리 접촉 law에서 독립적으로 생성한 합성 특성이다. 실물 장비나 확정적인 ground truth sensor를 사용하지 않는다.
