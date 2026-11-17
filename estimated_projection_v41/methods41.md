## 무엇을 새 물체에서 검증했는가

가령 로봇이 모서리를 밀 때, 실제로 닿는 곳은 그대로 두고 그 위쪽 외곽만 볼록하게 바꿀 수 있다. 물리적 결과는 같은데 영상만 보고 계산한 미끄럼 확률이 달라지면, decoder가 주변 모양을 물리적으로 너무 강한 증거로 사용한 셈이다. 이때 주변 모양의 변동 방향을 결과 정답 없이 찾아 지울 수 있는지, 그 뒤에도 네 번의 접촉에 따라 보정 세기를 바꿀 이유가 있는지를 분리했다.

| 고정하거나 바꾼 부분 | 실험 절차 |
|---|---|
| 새 물체와 높이 | 과거 데이터와 독립적인 12개 catalog panel에서 flare·roll 형상을 만들고 query contact height를 19·31 mm로 바꿨다. 기존 기준 높이는 25 mm다. 2개 shape × 2개 height × 3개 contour × 2개 contact law로 24개 cell을 평가했다. Saturating과 crossover law는 v40에서 이미 다뤘다. |
| Contour intervention | 각 물체의 base·hollow·bulge는 같은 접촉 위치와 local tangent를 갖도록 만들었다. Factual profile, calibration 충돌, 실제 stick/slip label과 선택된 observation을 대응시켰다. Query view는 접촉 높이의 양옆 6 mm까지 2 mm 간격으로 측정한 7개 좌표다. 좌표별 noise σ는 0.30 mm다. |
| Target label 없는 추정 | Panel·shape·height마다 별도 물체 32개에서 두 번씩, 총 64번 passive readout을 얻었다. 같은 물체 내 signed deformation amplitude는 두 번 독립적으로 달라지지만 그 값·부호를 추정기에 알려주지 않았다. Whitened difference에서 평균을 빼고 첫 principal component를 구했다. 이 과정에 필요한 것은 같은 물체라는 pairing, 좌표 정렬, 같은 deformation을 공유하는 group, 알려진 sensor noise다. |
| Calibration과 배포 | 환경마다 물체를 바꿔 네 번 calibration했다. 세 번은 기존 높이에서 방향을 번갈아 측정하고, 마지막은 23 mm, 속도 +1.10이다. Query 속도는 +0.425다. 한 번의 평가에서 환경당 128개 deployment 물체에 frozen K256 acquisition이 115개 query view와 13개 probe를 배정했다. 모든 방법이 같은 관측을 받았다. |
| Decoder | Original은 7개 좌표 전체를 쓴다. Estimated는 학습한 단위 방향 û를, oracle은 simulator가 아는 u를 써서 whitened query likelihood 공간에서 I−uuᵀ를 적용한다. Probe likelihood는 동일하다. 16,384개 source 가설은 실제 query 높이에 맞춰 관측 template과 가설의 stick event를 새로 계산했다. Acquisition이 사용하는 source 기준 높이 25 mm는 유지했다. 모든 비교에 같은 환경별 두 proxy alignment parameter(geometry와 material의 prior correlation)를 제공해 bank를 생성했다. 이 parameter까지 추정해야 하는 운영 조건은 별도 문제다. |
| Correction | Base stick 확률 p의 log-odds에 βzw를 더한다. β≈0.829와 signed cue z의 추정식은 동결했다. Fixed는 w=0.325, optimizer는 네 번의 calibration을 v38 source gain table에 넣어 w를 얻는다. Selector는 두 값 중 하나를 환경별로 고른다. 보정을 전혀 하지 않으면 w=0이다. |
| Selector의 정보 | 새 contact label·law 이름·shape 이름을 받지 않는다. Source label로 학습한 frozen scorer가 해당 환경에서 이미 획득한 128개 deployment 관측과 calibration을 사용한다. 한 환경의 모든 물체에 한 가지 방법을 적용하고, 각 noise 반복 때 독립적으로 선택한다. 실제 배치가 전부 도착한 뒤 내려야 하는 결정이다. |

**왜 paired difference인가.** 같은 대상의 profile을 xᵢ₁=bᵢ+aᵢ₁d+εᵢ₁, xᵢ₂=bᵢ+aᵢ₂d+εᵢ₂로 쓰면 차분에서 물체마다 다른 bᵢ가 정확히 사라진다. 좌표당 noise를 표준화하면 차분의 covariance는 isotropic noise와 ddᵀ에 비례하는 항으로 구성된다. 독립 Uniform[−1,1] amplitude와 표준화한 방향 v에서는 covariance가 2I+(2/3)vvᵀ다. 네 group의 변형 축 분산은 noise-only 분산의 3.76~6.90배다. 이 조건에서 첫 eigenvector는 변형의 축을 식별한다. 부호는 projection에서 사라진다. 여기서 passive는 충돌 측정 횟수에 들지 않는다는 의미다. 실물에서는 두 외곽 상태를 만들고 물체를 정렬하는 비용이 든다. 같은 강체의 동일 형상만 여러 번 촬영하면 aᵢ₂−aᵢ₁에 신호가 없어 이 방식은 성립하지 않는다. [생성식과 pair 정의](WORLD_KO.md) · [구현](world41.py)

**왜 항상 optimizer와도 비교하는가.** Selector가 거의 모든 환경에서 optimizer 하나만 골라도 평균적으로 fixed를 이길 수 있다. 환경별로 고르는 이득을 인정하려면 같은 projected decoder에서 fixed와 항상 optimizer를 모두 이겨야 한다. 두 정책의 확률 qₒ, q𝒻에서 ΔBrier=(qₒ−q𝒻)(qₒ+q𝒻−2Y)다. 과거 v39/v40 자료에서 이 차이와 연결되는 calibration residual을 예측하도록 ridge regression을 학습했다. v38 보정식을 그대로 사용하면서, 두 정책 중 하나를 고르는 상위 규칙은 v41용으로 학습했다. Source validation에서 선택된 모델의 projected selector−fixed는 leave-law-out에서 −0.000079, leave-shape-out에서 +0.000579였다. 두 결과 모두 hyperparameter 선택에 사용했다. [선택식과 학습 분할](SELECTOR_KO.md) · [Source 검증](selector_source41.json)

**평가 단위.** Panel마다 시작 방향이 다른 두 block, 여덟 환경, 두 표면 방향, strong·weak·null인 세 near-field regime, 32회 observation noise를 대응시켰다. Held-out Brier loss는 stick 사건 Y에 대한 (p−Y)²의 평균이고 작을수록 좋다. 24개 cell을 동일하게 가중했다. 반복 관측이나 재사용된 prototype 수가 confidence interval을 인위적으로 좁히지 않도록 독립 panel 12개를 추론 단위로 삼았다. Estimated 안의 두 selector 대조와 fixed에서의 두 projection 대조를 각각 Bonferroni–t로 보정했다. 각 family 내 양측 동시 95% 구간이다. 생성식·selector·판정 기준의 SHA-256을 target outcome 생성 전에 기록했다. [사전 분석 계획](PLAN_KO.md) · [동결 기록](freeze41.sha256) · [UTC 기록](frozen_utc.txt)
