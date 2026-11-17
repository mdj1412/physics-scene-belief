# v62 · 신규 deployment 및 compute 계약

Catalog 8000~8023에 두 material instance와 기존 두 rate law를 교차하고 각 material × law당 128개 fresh deployment object를 생성한다. 이전 저장 archive와 v61에서 제외된 5100번대 support까지 포함해 6차원 물리 vector의 중복을 확인한 뒤에만 GPU forecast를 시행한다. Label에는 동일 PyBullet solver의 fine 및 ultra resolution, 공통 frozen decoder 및 v54 gate를 사용한다. Binary replay seed는 62041729·62084591이다. Target truth는 평가 전용이다.

4개 Slurm allocation의 memory·utilization·PID를 조사했다. 2744467/2744469에서는 기존 Python이 각각 40 GB 이상 사용 중이었고 나머지 두 GPU는 각 0 MiB·0%·compute PID 없음이었다. Node55/2744468의 한 장을 선택해 forecast와 benchmark 전 각각 다시 검사한다. Guard로 외부 process를 감시하고 CPU solver/통계에서는 CUDA를 비활성화한다.
