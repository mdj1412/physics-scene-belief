## 실험과 예측은 무엇을 알고 있었는가

한 환경에서 네 번 calibration probe를 실행한 뒤, 아직 그 결과를 보지 않은 다른 물체가 충돌 후 stick할 확률을 예측했다. 같은 128개 물체 중 13개는 deployment probe를, 나머지 115개는 query view를 받았다. 같은 관측으로 얻은 base probability를 p, 첫 세 접촉에서 얻은 signed cue를 z로 두었다. 최종 확률은 sigmoid(logit(p) + 0.828976·z·w)다. Fixed는 항상 w=0.325를 사용한다. V38 optimizer는 네 번째 접촉의 outgoing velocity까지 보고 source gain table에서 w∈[0,1]을 고른다. 두 방식 모두 같은 fixed 4를 사용하므로 이 비교에서 네 번째 접촉의 구매 효과를 추정할 수는 없다. [동결 정책과 timing](POLICY_KO.md)

기본 shelf와 ridge 곡선은 v39에서 가져오고 각 곡선에 새로운 hollow와 bulge를 추가했다. 실제 접촉 높이 0·23·25 mm의 위치와 미분값, 사실 관측에 쓰는 −10~+10 mm의 profile은 같다. 입력은 렌더링한 RGB image 대신 높이별 위치를 재는 geometry sensor다. 7점 query view에서는 주변 contour가 최대 ±0.90 mm, 센서 표준편차 대비 RMS 1.925배 달라진다. 세 contour 모두에 처음 쓰는 두 friction law를 적용했다. Saturating은 속도에 따라 방향 차이가 커지다 포화한다. Crossover는 tangential speed 0.70을 지나면서 마찰의 방향 차이가 부호를 바꾼다. 두 함수는 stress test를 위해 설계했고, 과거부터 쓴 rigid contact와 stick 판정 방식은 공유한다. [물리 명세와 방정식](PHYSICS_KO.md)

조건별 수치 예측에는 v39에서 이미 관측한 정답만 사용했다. 먼저 v39의 query view에 새 contour 변위를 더해 같은 decoder에서 관측 변화의 효과를 추정했다. 다음으로 두 과거 표면 방향에서 관측한 stick label을 새 law의 nominal 속도에서 예측한 마찰 비율로 섞었다. Δ는 두 Brier의 차이이므로 고정한 두 예측에서 label에 대해 선형이다. 이 성질은 과거 label을 섞는 계산을 정당화하지만, 새 law의 실제 label이나 보정량을 알아낼 수 있게 해 주지는 않는다. [사전 예측표와 가정](PREDICTIONS_KO.md)

네 번째 접촉은 23 mm·+1.10, query 충돌은 25 mm·+0.425로 고정했다. 12개 독립 catalog panel 각각에서 두 시작 방향으로 물체를 뽑고, 8개 환경·128개 held-out 물체·32회 sensor noise를 사용했다. 표면 방향 두 가지, query 부근의 방향성이 유지되거나 약해지거나 사라지는 세 regime, 모든 shape × law를 같은 물체와 noise에 대응시켰다. 288개 block archive는 별도로 생성한 288개 독립 catalog를 뜻하지 않는다. 사전 검증 계획과 forecast를 2026-11-15 18:20:44 KST에 서버에서 동결한 뒤 GPU를 실행했다. [설계와 판정 기준](PLAN_KO.md) · [동결 hash](freeze40.sha256)
