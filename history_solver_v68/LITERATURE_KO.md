# v68 · 처음 접촉부터 법칙이 달라진 환경에서 무엇을 검증할 수 있는가

v67에서는 같은 물체의 관측을 유지하고 마지막 충돌만 다시 계산했다. v68에서는 표면과의 첫 상호작용부터 다른 마찰 법칙을 경험한 뒤에도 어느 형상 좌표를 억제하거나 복원하는 것이 유리한지 묻는다. 첫 접촉 정보가 예측기를 통해 결과를 바꾸는 경로와, 같은 예측에서 마지막 접촉의 정답이 바뀌는 경로를 구별해야 한다. 아래는 이 질문과 직접 맞닿은 원 연구 여섯 편에 한정한 검토다. [v66](../invariance_audit_v66/REPORT_KO.md) · [v67 설계](../law_axis_v67/PLAN_KO.md)

## 1. 연구 질문·표본·방법·결과·한계 비교

| 원 연구와 질문 | 표본과 방법 | 주요 관찰과 결론 | v68에서 적용할 수 있는 범위 |
|---|---|---|---|
| Fazeli 등. 어떤 rigid contact model이 실제 impact를 설명하는가? | 평면 낙하 2,000회 중 품질 검사를 통과한 1,718회. 여섯 model에서 ensemble 및 개별 impact의 parameter를 식별하고 측정 impulse가 허용 영역에 속하는지 검사했다. | 개별 충돌에 parameter를 맞춰도 실측 impulse를 표현하기 어려운 경우가 있었다. Ensemble 예측오차와 물리 parameter의 해석 가능성도 따로 평가해야 했다. | 관측을 늘려도 solver가 허용하는 결과 영역이 제한될 수 있다. 두 곡면의 stick 확률이나 Brier 차이는 측정하지 않았다. [원문 §2–4](https://arxiv.org/html/1710.04979) |
| Acosta·Yang·Posa. Contact parameter를 맞춘 solver는 실물의 어느 운동을 재현하는가? | 10 cm acrylic cube 궤적 550개와 Cassie jump 22회. Cube에는 Drake/MuJoCo/Bullet, Cassie에는 Drake/MuJoCo를 비교했다. Parameter를 fit하고 trajectory error를 측정했다. | 비탄성 충돌의 큰 움직임은 대체로 맞았지만 탄성 충돌과 일부 edge/corner에서 어려움이 남았다. 같은 cube 자료에도 최적 friction coefficient는 세 solver에서 0.10, 0.22, 0.39였다. | 여러 solver가 같은 동작을 근사해도 식별한 마찰값이 같다는 보장이 없다. 평균 궤적 오차와 threshold 기반 확률 오차는 측정 대상이 다르다. [원문 §III–VI](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf) |
| Le Lidec 등. Contact formulation의 근사와 반복 계산법은 궤적을 어떻게 바꾸는가? | 저자들의 동일 benchmark 구현에서 plane 위 sliding cube 등 해석해가 있는 단순 장면과 여러 접촉이 있는 장면을 재생했다. Friction cone·complementarity 근사와 수치법을 비교했다. | 단순 장면에서도 근사에 따라 sliding trajectory와 에너지 소산이 달라질 수 있었다. Model formulation과 수렴 정도를 각각 점검하는 근거다. | 수치 실험이다. 두 solver가 다른 답을 내면 어느 쪽이 실물에 가까운지는 별도의 reference가 필요하다. [원문 §IV–V, Figs. 9–10](https://simple-robotics.github.io/publications/contact-models/static/paper/lelidec2024contacts.pdf) |
| Yu 등. 같은 초기 조건에서도 마찰과 최종 위치는 얼마나 변하는가? | Shape 11종과 surface 4종. 접촉 위치, 방향, 속도, 가속도를 조절하고 pusher force와 pose를 기록했다. 한 straight push 설정도 2,000회 반복해서 분석했다. | Polyurethane에서는 방향에 따른 friction 차이가 컸다. 반복한 push의 최종 pose 분포는 단일 Gaussian으로 설명하기 어려웠다. 제목의 백만 단위는 시간 표본이다. | 표면과 이력을 분리할 필요를 뒷받침한다. Planar pushing과 짧은 충돌의 마찰곡선을 같게 둘 실증 근거를 제공하지는 않는다. [원문 §III, V, VII](https://arxiv.org/pdf/1604.04038) |
| Chebotar 등. 소수의 실물 rollout으로 simulator 기반 policy를 개선할 수 있는가? | Swing-peg-in-hole과 drawer opening에서 iteration당 실물 rollout 3회로 simulator parameter distribution을 갱신하고 policy를 다시 훈련했다. 두 task를 각각 실물 20회로 평가했다. | Swing-peg-in-hole은 두 번의 갱신 뒤 18/20회, drawer는 갱신 뒤 20/20회 성공했다. 관측을 실제 추정·행동 경로에 넣어 변화를 검증했다. | 당시의 training·policy 갱신과 v68의 고정된 predictor 및 관측 예산은 다르다. Calibration이 유용할 수 있다는 실증 사례다. [원문 §V.D](https://arxiv.org/pdf/1810.05687) |
| Johansson·Sontag·Ranganath. 일부 정보를 지우는 domain-invariant representation의 transfer risk는 무엇에 달려 있는가? | Transfer bound를 유도하고 digit classification benchmark에서 가정 위반의 영향을 조사했다. 물리 intervention 표본은 없다. | Representation의 non-invertibility와 source support가 target 성능 해석에 필요함을 보였다. | 좌표 삭제·복원 ablation의 이론적 동기다. 유한한 readout의 Brier 차이에는 normalization과 source fit도 관여한다. [원문](https://proceedings.mlr.press/v89/johansson19a.html) |

## 2. 근거의 연결과 겉보기 불일치

**관측 이력 → model update → forecast.** Chebotar 등은 관측 분포로 simulator와 policy를 실제 갱신했다. Acosta 등에서는 같은 실측을 설명하는 parameter가 여러 개였다. 따라서 이력의 예측 기여와 이력에서 물리 parameter를 식별하는 능력은 각각 측정해야 한다. 두 연구는 성공률과 궤적 오차를 측정했으며, binary slip의 확률은 직접 평가하지 않았다. [Chebotar 등](https://arxiv.org/pdf/1810.05687) · [Acosta 등](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf)

**물리 경로와 표현 경로.** 하나는 contact law·표면 → 접촉 운동 → terminal label이고, 다른 하나는 geometry measurement → projection axis → forecast다. Fazeli 등은 첫 경로의 model 한계를 측정했고, Johansson 등은 두 번째 경로에서 정보가 사라질 가능성을 이론화했다. 이 여섯 논문에는 두 경로를 교차해 Wendland와 Dispersive의 Brier 손익을 판정한 데이터가 없다. [Fazeli 등](https://arxiv.org/html/1710.04979) · [Johansson 등](https://proceedings.mlr.press/v89/johansson19a.html)

**두 결과가 달라 보이는 이유.** Acosta 등의 cube에서 평균 궤적이 상당 부분 맞았다는 결과와 Fazeli 등의 개별 충돌이 허용 impulse 밖에 들었다는 결과는 양립한다. 전자는 3D 위치·자세의 연속 오차를 집계했고 후자는 평면 충돌 한 건의 post-impact 운동이 물리적으로 허용되는지 물었다. Yu 등에서는 같은 물체의 저속 push에서도 최종 pose가 여러 방향으로 퍼졌다. 조건과 endpoint가 다른 자료로 어느 solver가 항상 더 정확한지 정할 수 없다. [Acosta 등](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf) · [Fazeli 등](https://arxiv.org/html/1710.04979) · [Yu 등](https://arxiv.org/pdf/1604.04038)

**근거의 강도.** 실측과 수치 실험은 접촉 근사·재료·평가지표에 따라 결과가 달라질 수 있다는 데 대체로 일치한다. 첫 접촉 이력이 도움이 된다는 근거는 과제 의존적이다. 같은 형상 좌표에 대한 손익 부호가 서로 다른 solver 및 history에 걸쳐 유지되는지는 직접 검정해야 한다. [Le Lidec 등](https://simple-robotics.github.io/publications/contact-models/static/paper/lelidec2024contacts.pdf) · [Chebotar 등](https://arxiv.org/pdf/1810.05687) · [v67 검정](../law_axis_v67/EXPERIMENT_KO.md)

## 3. 기존 프로젝트의 설계 심사

- **Calibration의 접근 경로.** v67의 bank.fit(x)와 readout은 11개 factual feature와 선택한 측정 등을 사용했다. 새 이력을 파일로 만드는 일과 그 이력이 predictor에 미치는 영향을 분리해야 한다. 같은 factual·후속 관측·label에서 calibration만 바꿔 forecast의 변화율과 Brier 개선량을 각각 평가할 수 있다. [v67 코드](../law_axis_v67/forecast67.py) · [Chebotar 등](https://arxiv.org/pdf/1810.05687)
- **반전의 검정 단위.** 양수인 family interaction 하나만으로는 두 family의 부호가 반대인지 알 수 없다. 각 law에서 soft − full이 Wendland에서는 음수, Dispersive에서는 양수인지 각각 판정해야 한다. Restored − dropped는 다른 readout 비교다. 동일 catalog의 noise·axis 반복은 독립 배포 표본 수를 늘리지 않는다. [v67 판정 기준](../law_axis_v67/EXPERIMENT_KO.md)
- **Solver 독립성과 수치 오차.** Normal load, tangential update, 접촉 종료 및 time/space discretization에서 무엇이 달랐는지를 명시해야 한다. 동일한 객체 생성기, rate law, renderer를 공유하면 이들 요소에 대한 외삽은 여전히 미확인이다. 특히 cutoff 근처의 residual speed 분포와 fine step에서 label·Brier contrast의 변화량은 효과의 수치적 불확실성을 알려 준다. [Le Lidec 등](https://simple-robotics.github.io/publications/contact-models/static/paper/lelidec2024contacts.pdf) · [Acosta 등](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf)
- **해석할 수 있는 경로.** 새 history가 어떤 observation을 구매할지까지 바꾸면 전체 policy effect에는 sensing 경로의 변화도 들어간다. Matched world에서 history·terminal outcome을 교차하고 같은 action에 고정한 대조와 전체 정책 결과를 구별할 수 있다. Solver state와 law ID는 실험자의 진단용 변수이고, 배포 가능한 입력을 열거해야 정보 누출을 확인할 수 있다. [v67 교차 계산](../law_axis_v67/legacy_cross67.py) · [Chebotar 등](https://arxiv.org/pdf/1810.05687)

## 4. 근거가 남긴 연구 질문 열 가지

| 질문과 연구 공백 | 변수·접근 방법 |
|---|---|
| 1. Query와 속도 영역이 겹치지 않는 calibration도 law를 구별하는가? 마찰의 속도 민감도는 재료에 따라 달랐다. [Yu 등](https://arxiv.org/pdf/1604.04038) | 같은 예산에서 calibration 속도를 law 교차점 아래·위·양측으로 배정하고 Brier와 law 판별률을 분리한다. |
| 2. 같은 이력에 맞는 parameter가 여러 개면 어느 추가 측정이 예측을 개선하는가? 실측 궤적에서 유사한 parameter 영역이 발견됐다. [Acosta 등](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf) | 초기 위치를 고정하고 torque·normal/tangential velocity 중 하나를 추가 공개해 risk와 식별 오차를 비교한다. |
| 3. 같은 terminal solver에서 이력만 바뀌면 어느 물체의 결정이 달라지는가? 기존 primary 비교에서는 마지막 충돌을 바꿨다. [v67](../law_axis_v67/PLAN_KO.md) | World·후속 action·label을 고정하고 history를 교차해 forecast와 Brier·route 변화의 분자를 각각 기록한다. |
| 4. 패치의 공간 해상도를 늘려도 반전이 남는가? Contact cone 근사가 운동 방향에 영향을 주었다. [Le Lidec 등](https://simple-robotics.github.io/publications/contact-models/static/paper/lelidec2024contacts.pdf) | Time step을 고정하고 contact node 수를 바꾼 후, 공간 해상도를 고정해 time step만 바꾼다. |
| 5. 임계값에서 먼 물체에도 반전이 있는가? 궤적 오차가 작아도 bounce·slide 판정은 달라진 사례가 있다. [Acosta 등](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf) | 전체 모집단을 primary로 두고 사전 지정한 speed margin별 표본 수·Brier 기여와 continuous error를 낸다. |
| 6. 회전 coupling이 큰 물체도 좌표 복원 이득을 보이는가? Planar impulse의 모델 허용 영역은 제한됐다. [Fazeli 등](https://arxiv.org/html/1710.04979) | 관성과 moment arm을 교차한 solver 또는 회전 가능한 fixture에서 angular impulse와 stick을 함께 잰다. |
| 7. Source family의 구성은 projection axis의 변동을 얼마나 설명하는가? 좁은 source support는 전이의 제약이다. [Johansson 등](https://proceedings.mlr.press/v89/johansson19a.html) | 같은 관측 수와 head initialization으로 source family holdout을 반복해 axis angle과 target Brier를 측정한다. |
| 8. 보지 못한 damping law에서도 같은 관측이 유용한가? Cube와 Cassie에서 민감한 parameter가 달랐다. [Acosta 등](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf) | Source model bank에 없는 damping을 target에만 넣고 abstention·full·soft의 손익을 비교한다. |
| 9. 표면의 사용 이력을 알려주면 예측 확률이 개선되는가? 같은 표면의 반복 push에서도 분산이 관찰됐다. [Yu 등](https://arxiv.org/pdf/1604.04038) | 실물 surface·session 단위로 분리해 반복 횟수·온도·마모를 기록하고 교체 직후와 사용 후를 비교한다. |
| 10. 적은 실물 초기 관측의 이득이 다른 재료에서도 재현되는가? 기존 rollout 개선 예는 두 manipulation task에 한정된다. [Chebotar 등](https://arxiv.org/pdf/1810.05687) | 표면별 초기 impact로 calibrate하고 독립 물체·날짜의 slip speed와 Brier를 같은 관측 예산에서 측정한다. |

논문에서 가장 강한 판별은 사전에 고정한 부호 검정이 바뀐 물리 가정에서도 성립하는지다. 성립하면 어떤 model 요소를 공유하는지 명시한 cross-solver 재현을, 실패하면 어느 단계에서 부호가 어긋나는지를 보여 준 history/label 대조를 제시할 수 있다. 실제 surface에서의 유용성은 실측 holdout으로 남는다. [v67의 범위](../law_axis_v67/EXPERIMENT_KO.md) · [실물과의 대조 방법](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf)
