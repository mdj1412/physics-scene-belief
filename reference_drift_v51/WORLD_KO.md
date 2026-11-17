# v51 · 실제 surface 교체와 session별 배포 결과

서로 다른 두 원인이 같은 평균 Brier를 만들어도, **오류가 여러 물체에 한꺼번에 나타나는 정도**는 달랐다. 2.0 N reference와 impact 부호의 불일치가 개별 active 물체의 25%에서 일어나는 경우, likelihood soft의 unsigned 대비 session Brier 차이의 표준편차는 **0.00340**이었다. 표면 교체 후 session의 25%에서 polarity law가 뒤집히는 경우 이 값은 **0.00751**이었다. 두 조건의 기대 Brier는 같은 식으로 계산된다. [세션별 결과](rollout51.json) · [독립 rollout과 적분 대조](rollout_replay51.json)

12개 catalog에서 mosaic↔nap 교체를 두 방향 모두 시행했다. 각 deployment environment에 contact law 이름이 같고 surface 이름이 다른 calibration environment를 배정하면 모든 catalog에서 행 번호가 [2,3,0,1]로 대응한다. v50의 독립 calibration/deployment support와 두 surface의 실제 PyBullet friction field를 재사용했다. Environment와 specimen이 바뀌므로 동일 물체의 surface 하나만 바꾼 물리 개입이라고 해석할 수 없다. v50에 등장한 target population에서 교체 전후 transfer를 짝지어 평가한 실험이다. [배정과 개입의 정의](world51.py) · [사전 고정한 설계](WORLD_PLAN_KO.md)

- **Session drift g.** 표면을 교체할 때 Bernoulli(g) 한 번을 추출한다. 참이면 배포 시 weak/reference 방향 유지율 κpost가 1−κpre가 되고, 거짓이면 κpre다. 128개 물체가 이 사건을 공유한다. g=0에서도 표면은 교체된다.
- **Reference error ε.** 새 표면에서 active 물체마다 별도 Bernoulli(ε)를 추출해 그 물체의 post-replacement reference 부호를 impact에서 뒤집는다. Inert는 inert로 남는다. 이 평가에서 impact는 지정된 하중 변형의 부호로 선택한 PyBullet trajectory다. 실제 impact 중 변형을 실측한 자료는 없다.
- **관측 가능성.** 4쌍의 calibration readout은 원래 표면의 별도 물체에서 수집했다. Target 128개에는 각자 한 개의 local weak gauge만 제공했다. 반복 내에서 slot·measurement·prediction을 ε와 g 전체에 동일하게 유지한 후 outcome만 바꿨다. 250개 가상 fleet의 각 catalog·environment에 5개 κ가 각각 50번씩 배정된다. 각 fleet은 환경당 1개 calibration prefix와 물체당 1개 binary outcome을 포함한다. 재사용한 12개 catalog가 추론 단위다. [실현 배포 코드](rollout51.py)

| Likelihood soft − unsigned fit, 128개 물체 한 session | 개별 오류 ε=.25, g=0 | 집단 변화 ε=0, g=.25 |
|---|---:|---:|
| 평균 Brier 차이 | −0.000689 | −0.000744 |
| 동일 catalog·환경·κ 내 반복 표준편차 | 0.003397 | 0.007512 |
| Session Brier 차이의 p90 | +0.003739 | +0.009363 |
| 48개 session fleet 단위에서 unsigned에 진 반복 | 12.0% | 28.0% |

값이 양수이면 soft가 손해다. 평균 차이의 두 조건 간 변화는 −0.000055 [−0.000216, +0.000106]이었다. 반면 session 표준편차는 **+0.004115** 증가했다(12개 catalog의 paired bootstrap pointwise 95% CI +0.003891~+0.004316). Likelihood soft − vote soft의 session 표준편차도 0.000619에서 0.001065로 증가했다. 깨끗한 조건(ε=g=0)에서 첫 fleet 하나만 꺼내면 soft − vote가 +0.000030 [−0.000045, +0.000106]이었다. 균형화한 250개 fleet 중 39.6%에서 vote가 이겼다. 이 모의 반복률은 실시 환경에서 측정된 실패율이 아니다. [원자료](rollout_sessions51.npz)

**기대값이 같아지는 이유.** 한 active 물체의 weak→impact 부호 유지율은 q = ε+(1−2ε)[(1−g)κpre+g(1−κpre)]다. 두 물체가 같은 session에 속하면 weak→impact 부호 일치 indicator 간 covariance는 g(1−g)(1−2ε)²(2κpre−1)²다. 따라서 κpre=0 또는 1에서 위 두 조건은 같은 q를 주지만 inter-object covariance는 각각 0과 0.1875다. 별도의 latent-state 검산에서 8,192개 session × active 물체 64개로 추정한 κpre=0의 covariance는 0.00004와 0.18396이었다. κpre=.5에서는 두 값이 모두 0에 가까워진다. 여러 독립 session에서 물체별 변형을 함께 측정하면 이 가정의 적합성을 시험할 수 있다. [Covariance 실험](covariance51.json)

**조작·채점 검산.** 36개 조건에서 실제 active-object 부호 불일치율과 지정 ε의 절대차는 최대 0.00094, 관측 session drift 빈도와 지정 g의 차이는 최대 0.00517이었다. ε=.25, g=0에서는 전체 물체의 4.56%에서 reference 부호를 그대로 강제한 경우와 실제로 선택한 impact 부호의 stick/slip label이 달랐다. 항상 뒤집히면 이 비율은 18.34%였다. Inert를 포함해 분모를 설정했고, active 부호를 바꿨을 때 원래부터 label이 변하는 비율은 36.75%였다. 독립 채점기와 대조한 soft−vote, soft−unsigned 기대 Brier의 최대 절대 오차는 36개 조건에서 각각 0.0000087, 0.0000830이었다. [조작 감사](rollout51.json) · [적분 대조](rollout_replay51.json)
