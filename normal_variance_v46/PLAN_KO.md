# v46 · Target label 생성 전 고정한 비교

## 실험 질문

새 contour에서 같은 접촉 전 영상을 볼 때, 추정한 접촉 위치 x에 주변 surface normal의 분산을 추가해 projection 여부를 고르면 원래 v45의 위치 규칙과 always-original보다 미래 stick의 Brier loss가 낮아지는가? 세 번째 비교는 같은 source 물체와 같은 Brier objective로 다시 fit한 position-only다. 이 비교가 성립해야 normal variance의 추가 정보에 대한 근거로 해석한다.

## Source, target, 정보 예산

- Source는 완료된 v45 fold와 ripple의 기존 archive다. v45에서 쓰던 original·source-axis projection, camera, query와 probe 예산을 유지한다. 두 decoder의 forecast 차이가 없는 probe arm도 그대로 포함한다. 미래 event는 stick residual threshold 0.020 m/s다.
- Target은 기존 shape와 다른 terrace·dimple, 12개 새 catalog support 600–611 × 4환경 × 128개 물체다. 두 surface, 두 law, base/hollow/bulge, 명목 높이 19/31 mm, offset −0.5/0/+0.5 mm, 4개 sensor noise를 대응시킨다. Terrace는 두 비주기 smoothstep shelf, dimple은 폭이 다른 Gaussian depression으로 구성한다. 다른 target family를 선택하거나 난이도로 표본을 거르지 않는다.
- 세 grayscale frame에서 동일한 x를 읽고 그 주변 ±2 mm의 local normal angle variance를 계산한다. 2D side view에 나타난 extruded contour의 법선 각도이며 단위는 rad²다. 참 x와 해석적으로 계산한 variance는 별도 평가에서만 사용한다.
- V45 fixed x는 30.5 및 31 mm bin에서만 projection한다. 공정한 refit-x와 joint는 같은 v45 finite pressure의 두 Brier 차이로 학습한다. Measured x bin별 네 variance 분위 strata의 평균 D를 같은 x bin의 평균 D 쪽으로 25% shrink한 후 음수이면 projection한다. 입력 위치를 다시 추정하거나 target label에 맞춰 threshold를 고르지 않는다. Oracle diagnostic은 source에서도 true feature로 같은 규칙을 학습하고 target의 true x·variance에 적용한다.
- Target 결과 생성 전 geometry, source router, camera estimator, 코드와 계획을 파일로 동결한다. Target 입력만 사용하는 예측은 target outcome보다 먼저 계산할 수 있다. 실제 실험은 할당된 서버 compute node에서 한다.

## 분석과 반증 기준

1. Primary는 prescribed pressure의 narrow/broad를 균등 평균한 held-out Brier다. Joint−always-original, joint−v45 fixed x, joint−refit-x 세 paired 차이와 세 비교에 대한 Bonferroni simultaneous 95% CI를 계산한다. 음수일수록 joint가 유리하다. 세 구간 모두 0 아래일 때만 세 비교를 모두 이겼다고 판정한다. CI가 0을 포함하면 동등성을 주장하지 않는다. 독립 단위는 12개 catalog이고, 형상은 고정한 두 family이다.
2. 각 family의 세 차이를 6개 비교의 동시 구간으로 따로 계산한다. Narrow·broad, 높이, variant, 선택 빈도와 항상 projection하는 진단도 기록한다. 결과가 양쪽 family에서 어느 방향인지 구분한다. Same x bin에서 variance route를 바꾸는 물체가 있는지, x-only에서 바뀐 선택이 손실에 기여하는지 기록한다. 같은 panel·environment·추정 x bin에서 joint 선택만 무작위로 바꾼 기대 loss를 계산해 위치별 projection 빈도 효과를 확인한다. 이 shuffled comparator는 배포할 정책이 아니다.
3. True x·true variance로 source fit한 oracle과 true x-only를 대조한다. 같은 bin에 있는 실제 shape variation과 영상 추정치가 일치하는지 살펴서 측정 오차, 상대 risk의 source→target 이동, 공통 projection axis의 부적합 가능성을 구분한다. Oracle을 최적 policy로 해석하지 않는다.
4. 독립 PyBullet query에서 동일 물체와 같은 route로 추가 검증한다. 실제 첫 접촉 x 오차를 확인한다. Surface·law와 pre-impact camera rendering을 공유하므로 서로 독립인 두 배포 현장으로 간주하지 않는다.
5. 512개 batch의 실제 GPU p90은 한 장의 빈 GPU로, warm-up과 randomized paired timing 후 보고한다. Visual arms에서 camera host transfer·pixel→geometry·routing 및 실제 선택된 한 decoder의 physics posterior까지 포함한다. Offline rendering, 실제 video capture와 reset은 제외한다. 비교 정책 간 신뢰구간 및 측정 변동을 함께 기록한다. 적은 수의 family와 합성 영상을 사용하기 때문에 family 전체·실물로의 외삽은 미검증이다.
