### 어떤 일반화가 아직 필요한가

1. **방향을 찾는 것과 지워도 되는지 판단하는 것은 다른 문제다.** 두 profile에서 자주 달라지는 방향은 관측만으로 찾을 수 있다. 그 방향을 따라 변형해도 실제 stick event가 같다는 사실은 v41의 물리 실험 설계에서 얻었다. 실제 사용자가 건넨 미지의 물체에서 방향의 운동학적 무관성까지 label 없이 식별한 결과는 아니다. Oracle도 실제 변형 축만 아는 방법으로, Brier에 대한 최적 해법은 아니다.
2. **접촉 높이의 정확도가 중요하다.** 물체를 미는 점이 설계 높이에서 조금만 벗어나도 주변 contour가 실제 접촉에 관여한다. 생성식을 미분하면 ±0.25 mm 오차에서 19 mm의 bulge와 base 사이 local slope 차이는 flare에서 최대 약 0.314, roll에서 약 0.203이다. 명목 높이에서는 0이다. 이것은 미분식의 민감도이며, 높이 오차가 있는 상태의 Brier는 측정하지 않았다. 또한 decoder가 받는 입력은 등록된 7점 profile이다. Raw image, 유한한 contact patch, occlusion은 별도 검증이 필요하다. [세계 생성식](WORLD_KO.md) · [독립 물리 계산](physics_audit41.py)
3. **어느 target에서든 바로 쓸 수 있다는 주장은 범위를 넘는다.** 측정에 쓰는 group과 두 상태의 대응이 주어졌고, 각 group에는 추가 물체 32개가 있다. 세 decoder가 공유하는 source bank에는 환경별 proxy alignment도 주어졌다. 평가한 shape family와 height는 각각 두 개이며 contact law 두 개는 historical selector source에도 있었다. 12개 panel의 confidence interval은 이 조건에서 새 catalog를 뽑을 때의 불확실성을 나타낸다. 실물 material 모집단의 불확실성까지 포함하지 않는다.
4. **GPU latency와 데이터 획득 비용의 단위가 다르다.** Fixed와 optimizer에 같은 네 번의 calibration을 제공했으나 fixed의 보정량은 네 번째 응답에 의존하지 않는다. 64개 passive profile은 새 group에 처음 진입할 때 발생하며, selector는 128개 배치의 관측이 모일 때까지 기다린다. 한 번에 한 물체를 다루는 로봇의 전체 반응 시간 또는 작업 성공률은 GPU p90·Brier만으로 계산할 수 없다.

### 다음 한 실험

다음 버전에는 **어떤 방향을 지우면 실제로 위험해지는지**를 먼저 배치하는 편이 좋다. 같은 물체에서 (a) 접촉부는 고정하고 위·아래 외곽만 교체한 조건, (b) 같은 크기의 변형이 접촉부까지 이동한 조건을 만든다. 관측한 profile 차이의 크기와 sensor noise를 일치시키고, 숨긴 contact height 오차를 0·±0.25·±0.5·±1 mm로 사전 지정한다. 보정 action에 w=0을 포함한 세 가지 선택 규칙을 source에서만 동결한다. 물체별 true event flip rate, uncorrected/fixed/selected Brier, projection과 finite-variance·local-window baseline의 paired 차이를 함께 측정한다. 원래 event가 바뀌는 positive control을 포함해야 너무 많은 정보를 지운 경우를 검출할 수 있다.

실물로 옮길 때는 한 강체의 접촉 strip은 고정한 채 비접촉 insert만 교체한 물체와, 접촉 strip 자체도 바꾸는 물체를 함께 사용한다. Camera 및 contact height 오차, 재정렬 시간, slip 비용을 기록하고, 물체·재질 단위 holdout에서 같은 사전 기준으로 평가한다. 이 실물 실험은 v41의 결과에 포함되지 않는다.
