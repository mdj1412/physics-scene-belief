# 실측 정지 표면에서 round-trip 검증

V64에 사용한 DiLiGenT 10개 물체와 정방향·역방향 네 조명의 사전 지정 patch만 사용한다. V64 기준은 첫 영상에 대한 나머지 세 영상의 NCC가 모두 0.8 이상이고, 각 수평 offset이 모두 +2 pixel 이상 또는 모두 −2 pixel 이하일 때 일관된 오등록이다. 이번에는 각 조명의 forward match 지점에서 첫 영상으로 되돌아오는 NCC를 새로 계산한다. 세 backward NCC 모두 0.8 이상이고 세 왕복 위치 오차가 각각 Euclidean 1 pixel 이하인 조건을 추가한다. Forward만 통과한 수와 그중 오등록 수, round-trip까지 통과한 수와 그중 오등록 수를 비교한다. 유지된 물체 좌표를 optical registration 음성 기준으로 사용한다. 이 데이터에는 움직이는 물질점의 실제 coded identity와 strain이 없으므로 실시간 ID 판독 성능·현장 false switch 빈도는 추정하지 않는다.
