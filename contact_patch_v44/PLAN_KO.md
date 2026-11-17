# v44 · target 충돌을 실행하기 전 판정 기준

## 추정량과 검증 단위

v43의 projection은 whitened 7점 query profile에서 실제 contour 변형 방향 하나를 제거한다. 보정 0%에서 D = Brier(privileged projection) − Brier(original)로 둔다. D<0이면 지운 쪽이 이득이다. Target은 새 catalog 물체의 미래 query 충돌이다. 원래 장치의 factual evidence·deployment probe readout과 frozen v43 decoder를 유지하고 **query 결과를 만들어 내는 solver와 접촉 도구**를 바꾼다. 독립 solver의 결과나 tool 설정을 operating decoder에 주지 않는다. 따라서 독립성의 범위는 query dynamics이고, 모든 센서까지 새 engine으로 바꾼 closed-loop system은 별도 실험이다.

## 물체, 도구, 정보

12개 새 catalog × 2 object(fold·ripple) × 2 surface × 2 law × 3 contour × 2 nominal height(19·31 mm). 각 panel의 4개 환경에서 solver 평가에 미리 정한 동일 물체 인덱스를 사용한다. 다른 shape·surface·law·offset·tool에서도 해당 draw와 sensor noise를 대응시킨다. Target tool은 0.5 및 2 mm tip radius × 0 및 2 mm nominal sole span. Sphere와 capsule의 실제 활성 접촉점 분포는 따로 측정한다. 숨은 contact offset은 −0.5, 0, +0.5 mm다. 비영 두 점에는 같은 가중치를 준다. V43의 전체 비영 여덟 점 평균과 동일하게 취급하지 않는다. 새 simulator의 명령값·solver label 정의·실제 활성 patch와 timestep/threshold 검산 범위는 별도의 solver 명세를 동결한다.

## 사전 예측

Forecast에는 v43 과거 예측·과거 outcome, source geometry와 도구의 알려진 설계가 들어갈 수 있다. v44 target outcome 또는 target loss는 금지한다. Shape × nominal height × signed offset × 도구의 각 cell에서 D 부호를 미리 기록하고 folded/ripple의 nominal/shifted 평균에 대해서도 구분한다. 동일 historical point-solver의 손익 부호를 그대로 옮기는 방법, 모든 cell에서 손해라고 답하는 방법을 대조한다. Primary forecast는 base까지 모두 포함하며 deformed-only는 보조다. 이 forecast는 설계자 진단이며 장비·geometry를 모르는 online policy의 예측 정확도가 아니다.

## 판정

1. 주 비교는 네 도구를 동일 가중한 fold와 ripple 각각의 nominal 및 shifted D다. 독립 표본은 12개 catalog이며 panel 내에서 균등 평균 후 paired t를 쓴다. 4개 평균에 대한 양측 simultaneous 95% Bonferroni 구간이 0을 벗어나고 예측 부호와 맞으면 해당 방향이 확인된 것이다. V43에서 관찰한 fold의 nominal 이득/shifted 손해와 ripple의 두 조건 모두 손해라는 패턴은 네 조건을 별도 표시한다. 양쪽 기호가 모두 확인돼야 재현된 sign reversal이라고 한다.
2. 도구별 16개 shape × nominal/shifted D를 한 family의 동시 95% 구간으로 보고한다. 사전 sign forecast의 세부 48개 cell에서는 점추정 부호 일치율과 구간이 0을 벗어난 cell의 coverage를 나눈다. 48개 부호를 서로 독립적인 48회 trial로 해석하지 않는다. 각 hardware에서 original·projection의 절대 Brier와 event prevalence, 다른 도구로 바꿨을 때 같은 물체 event의 전환율을 함께 산출한다.
3. Signed nonzero offset을 평균해 큰 반지름−작은 반지름, 넓은 sole−0폭 sole, 두 효과의 interaction을 D 단위로 계산한다. 세 paired contrast의 동시 95% CI를 낸다. 변화시킨 nominal sole span은 입력 geometry이며 실측 접촉면적은 아니다. New held-out sample의 legacy point labels는 모든 예측을 동결한 뒤 별도 대조로 계산해 sampling과 solver·event 정의 변화의 영향을 구분한다.
4. 0%를 주 비교로 하고 v43 frozen temperature를 보조로 둔다. p90은 동일 GPU·동일한 512개 배포 물체의 warm online 경로로, ordering을 섞어 동기화한 뒤 측정한다. Solver ground truth 생성, 실제 물체 접촉·reset, offline bank 준비 시간을 online GPU latency에 합치지 않는다.
5. Target outcome을 보고 법칙·threshold·예측 부호·표본을 바꾸지 않는다. 접촉 소실·수치 실패·강한 class imbalance가 존재하면 별도로 수치를 남긴다. 실물 stick/slip, tool 회전이 자유로운 조건, 실제 patch pressure에 대한 결론은 이 설계의 범위를 벗어난다.
