# 별도 contact solver 검산

V44의 PyBullet rigid contact에서 새 v45 catalog의 각 환경 처음 8개 물체에 대해 작은 rigid sphere(R=0.5 mm, L=0)를 재실행한다. 초기 위치는 목표 접촉 x에 맞추고 첫 positive-force manifold의 실제 x 편차를 기록한다. Primary pressure 실험과는 별도 평가한다. 같은 frozen 확률과 영상 경로를 쓰되, 지정된 접근 지점과 실제 Bullet 접촉 위치로 oracle이 얼마나 달라지는지 검사한다. Camera는 지정 위치에 맞춘 pin 영상이므로 rigid sphere center와 optical rendering 간 불일치를 명시한다.
