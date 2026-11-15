## 두 종류의 네 번째 접촉에 같은 gate를 적용하는 대조

Target Brier를 열기 전에 추가한 secondary analysis다. Gate의 source artifact, 처음 세 접촉으로 구한 signed direction z, 보정 계수, threshold를 고정한다. 네 번째 conventional probe의 실제 motion 부호에 맞는 source likelihood를 first-three posterior에 곱해 보정 여부를 결정한다. 같은 절차에서 query 부근의 동작은 해당 높이·속도의 likelihood를 사용한다. 추론 시 필요한 입력은 paid factual contour·velocity, 관측한 probe velocity와 명령 부호다. Target outcome은 두 gate 결정이 끝난 후 채점한다.

세 local regime을 균등 가중한 결과가 primary 분석에 대응하는 이 부록의 주 비교다. Global null과 네 regime 균등 혼합도 공개한다. 각 범위에서 ① 각자의 원래 decoder에 적용했을 때 near gate4 − far gate4, ② (near gate4 − near gate3) − (far gate4 − far gate3), ③ 같은 near decoder로 채점한 near gate4 − far gate4, ④ 같은 far decoder로 채점한 near gate4 − far gate4의 Brier를 정한다. 음수는 near fourth의 이득이다. ③과 ④는 교차된 예측값을 사용해 두 gate를 진단하는 offline 비교다.

각 object × surface와 시작 부호 안에서 paired block bootstrap 12,000회를 시행한다. 두 orientation과 여덟 환경, 여러 sensor draws를 독립 block으로 세지 않는다. 각 범위마다 위 네 비교의 simultaneous 95% interval을 적용한다. v36에서 추가한 conventional gate의 수치는 기존 GPU raw archive를 동결된 source 모델로 CPU 재계산한다. 이 추가 방법의 latency는 따로 측정하지 않는다.
