# v60 · 별도 fresh deployment, contact solver 및 공통 GPU 모델

Catalog key 5000–5023에서 catalog별 2개 새로운 material, material당 기존 2개 rate law, law당 128개 deployment object를 배정한다. 새 deployment와 교체 전/후 support의 parameter vector는 서로 겹치지 않는지 및 v50/55/56/57/58/59 전체와 중복되는지 검사한다. PyBullet fine/ultra에서 물리 부호를 유지하거나 뒤집은 두 stick label을 계산한다. Stream별 replay seed는 60041729, 60084591이다. Observe/decoder, 네 번째 sensor와 v54 acquisition, joint terminal은 v59와 동일하게 유지한다. Sensor fault 및 trigger는 이 deployment draw에서 독립이며, target surface 이름이나 정답은 policy 입력에 포함하지 않는다.

사용자 GPU allocation 4곳을 조사했다. Node55/2744468 한 장에서 시작 직전 0 MiB·0%·compute PID 없음일 때만 forecast하도록 guard를 둔다. 24개에 대한 paired latency 측정에도 이 GPU를 사용하며 다른 process가 생기면 guard를 따른다. 생성·label·채점은 allocation CPU에서 CUDA를 끈 채 수행한다.
