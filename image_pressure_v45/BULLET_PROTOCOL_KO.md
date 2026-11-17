# 별도 rigid contact dynamics에서 영상 기반 선택을 검사하는 대조

V45 target outcome 생성 전 추가한 보조 평가다. V44의 PyBullet DIRECT, sphere radius 0.5 mm, 0.10 mm mesh, 20 µs time step, 80 iterations, 0.4 ms horizon, 마지막 하중 normal에 대한 0.020 m/s terminal speed rule을 재사용한다. V45 pressure intervention의 단일모드 마찰식은 사용하지 않는다. 실제 rigid sphere의 면적은 Bullet에서 내생적으로 결정된다. 이 arm은 pressure arm 비교에 포함하지 않는다.

12개 새 catalog에서 환경별 첫 여덟 물체를 고정한다. 48개 shape/surface/law/variant/height cell과 세 signed offset을 모두 평가한다. 중심 최초 접촉을 맞추기 위해 원하는 표면점에 sphere를 접하게 놓았을 때의 center를 normal로 계산한 뒤 접근 궤적을 역산해 0.030 mm gap에서 출발시킨다. 실제 최초 하중점은 mesh·finite timestep의 영향을 받을 수 있으므로 첫 양의 normal force가 관측된 step의 force-weighted contact x를 별도 저장한다. 영상과 selector는 이 사후 측정값을 읽지 않는다. 각 면의 material은 명령한 접촉 위치에서 평가한다. Flat terrain 및 경사 ±0.2, 마찰 0·1에서 실제 접촉 검출과 최초 위치 및 terminal speed를 target 실행 전에 확인한다.

실험이 검증할 범위는 같은 frozen decoder/시각 위치 estimator가 새로운 query dynamics에서도 평균 Brier 개선을 보이는지다. 독립 simulator에서 영상이나 factual evidence를 전부 다시 형성하는 검증은 추가로 필요하다. [구현](bullet45.py)
