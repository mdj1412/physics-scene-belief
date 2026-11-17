# v59 deployment 사전 명세

새 catalog 4000–4023에서 surface 2종과 rate law 2종을 교차하고 각 환경에서 held-out 물체 128개의 stick/slip을 예측한다. 신규 36,864개 physical parameter vector와 기존 v50, v55–58의 6성분 parameter vector의 중복을 검사한다. 같은 frozen physics decoder와 v54·fixed 3R+I terminal에 대해 같은 물체의 두 deformation 방향을 fine/ultra PyBullet로 계산한다. Stick은 residual tangential speed ≤0.020 m/s이다. 독립 deployment replay seed는 59041729, 59084591이다. 생성·score·latency에 실제 target label을 이용한 tuning은 없다. GPU는 node55 job 2744468의 UUID GPU-acff86bb-b0ee-515f-5139-502744220e28가 0 MiB·0%·compute PID 없음일 때만 기동한다.
