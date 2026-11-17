## 외부 solver 교차 검산 (secondary)

v66의 동일한 12 catalog × 4 환경, 각 환경에서 추출 순서의 처음 8개 object를 사용한다. Wendland와 Dispersive, 세 variant, 두 surface, 두 높이, 두 새 law 및 세 offset 전체를 짝짓는다. v44 PyBullet의 translation-only sphere/mesh rigid collision을 사용하고 반지름은 0.25/2 mm, sole span은 0, 물체의 회전은 고정한다. Primary compliant shoe의 halfwidth와 같은 숫자를 사용하되, rigid tip radius와 실제 contact area는 다른 물리량이다. Basis가 되는 material 식은 v67과 공유한다. Coulomb 계수는 최초 contact 위치와 접근 속도에서 평가한 뒤 해당 충돌 동안 고정한다. 이 검산에서는 네 번째 추천 후의 terminal label만 Bullet로 다시 계산한다. 따라서 처음부터 Bullet로 관측한 deployment를 추정하는 대조는 아니다.

최종 normal에 대한 speed ≤0.020 m/s를 label로 사용한다. 20 μs step, 0.10 mm mesh, 0.80 ms horizon을 사용하고, 12 catalog마다 선정한 첫 환경의 두 물체는 절반 step으로 재검산한다. Impulse가 없는 시도는 misclassification으로 치환하지 않고 별도로 기록한다. 동일 geometry의 frozen probability를 같은 object에 재채점하고 family·laws별 loss와 sign을 공개한다. 주 실험과 별개인 secondary 분석이다. 과거 실험의 v67 target label은 이미 알려져 있고, 새 target은 봉인한 forecast 뒤에 채점한다.
