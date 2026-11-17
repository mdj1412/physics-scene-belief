## 문헌에서 분리한 세 가지 측정 문제

문헌 검토 범위는 외부 camera가 접촉 위치를 관찰하는 연구, 재료와 형상에서 pressure field를 계산하는 연구, 실제 slip을 측정하는 연구의 원저 다섯 편이다. 아래 문헌적 주장은 이 다섯 편에 근거한다. 각 논문의 관측 시점과 정답이 다르므로 보고된 오차를 하나의 표준화한 효과로 합치지 않았다.

| 논문과 연구 질문 | 방법, 데이터와 표본 단위 | 핵심 결과와 저자 결론 | v45에 남기는 한계 |
|---|---|---|---|
| Kim et al., *Im2Contact*. 외부 camera 하나로 로봇이 든 물체의 접촉을 어디까지 찾는가? | Simulation에서 학습한 depth 기반 network에 reference image와 optical flow를 결합했다. 정량 평가는 simulation 500 episodes(75,000 frames), 실물 30 episodes(12,362 frames)였다. | Occlusion·모호한 형태가 단일 depth 추정의 실패 원인이었고 추가 영상 단서로 여러 contact type을 추정했다. | 정답은 영상 내 이미 발생한 contact 위치다. 접촉 이후 생긴 motion cue를 아직 일어나지 않은 impact의 예측에 그대로 사용할 수 없다. [원문 §§3–5](https://proceedings.mlr.press/v229/kim23b/kim23b.pdf) |
| Chaudhury et al., *Using Collocated Vision and Tactile Sensors for Visual Servoing and Localization*. 접촉 전의 영상 위치와 접촉 때의 tactile 정보를 어떻게 결합하는가? | Camera와 GelSight를 같은 축에 둔 실물 장비에서 물체별 접근과 접촉을 측정했다. 임의 위치 검증에서는 물체마다 5개 pose를 사용했다. | 풍부한 특징에서 두 sensor가 상호 보완했다. 반복 무늬와 특징 없는 접촉부에서는 tactile refinement가 잘못된 위치에 수렴하기도 했다. | 접근 경로, 교정과 접촉 부위 가시성이 결과에 영향을 준다. 일정한 접근·보이는 기준점에서 얻은 1D 오차를 일반 장면에 옮기기 어렵다. [원문 §IV](https://arkadeepnc.github.io/projects/collocated_vision_touch/img/ICRA_RAL_FINAL_Formatted-final-compressed.pdf) |
| Elandt et al., *A pressure field model…*. 분포 접촉이 만드는 net force와 moment를 빠르게 근사할 수 있는가? | 두 물체 내부에 pressure field를 부여하고 접촉면의 traction을 적분했다. 여러 기하 예제와 gripper–pencil simulation으로 거동을 보였다. | 한 점의 힘으로 표현하기 어려운 유한 접촉의 moment를 모델링했다. | 표면·재료에서 생기는 patch를 계산하는 방법이다. 중심을 고정하고 외부에서 면적을 지정한 v45에서는 형성 과정과 torsion을 검증할 수 없다. [원문](https://arxiv.org/pdf/1904.11433) |
| Leins et al., *HydroelasticTouch*. 분포 압력으로 채운 synthetic taxel 데이터가 실물 센서의 물체 상태 예측에 쓰이는가? | Hydroelastic surface에서 pressure를 샘플링했다. 5종 primitive로 21,000개 no-pitch 또는 105,000개 pitch 학습 표본을 구성해 실물 물체 방향 오차를 측정했다. | 일부 실물에서 network의 전이를 보였고 soft brick과 ruler에서는 PCA가 더 정확했다. | 이득이 geometry와 baseline에 따라 달랐다. 이 실험의 endpoint는 물체 방향이며 미래 stick의 Brier와 다르다. [원문 §§IV–VI](https://arxiv.org/html/2501.08077v1) |
| Yuan et al., *Measurement of Shear and Slip with a GelSight Tactile Sensor*. 실제 patch가 미끄러지기 시작하는 시점을 볼 수 있는가? | Elastomer의 marker 이동을 영상으로 추적해 gripper형 장치에서 normal·shear·torsional loading과 sliding 과정을 관찰했다. | 가장자리에서 시작해 안쪽으로 전파되는 incipient slip을 관찰했다. | Aggregate tangential speed는 patch 안의 stick/slip 공존을 구분하지 못한다. [원문](https://people.csail.mit.edu/yuan_wz/GelSight1/ICRA15_2740_FI.pdf) |

**지식 지도.** 접근 영상 → 보이는 경계와 경로 → 최초 접촉 위치의 불확실성은 Kim·Chaudhury가 다룬 관측 문제다. 접촉 위치·재료·형상 → pressure 분포 → 힘과 moment는 Elandt·Leins가 다룬 역학 문제다. 국소 변형 → incipient slip이라는 실제 측정 수준은 Yuan이 다룬다. V45의 연결은 최초 위치 추정이 downstream decoder 선택에 주는 이득을 재고, 같은 위치에서 pressure 폭 개입이 그 decoder의 손익을 바꾸는지 묻는 것이다. [Kim](https://proceedings.mlr.press/v229/kim23b/kim23b.pdf) · [Chaudhury](https://arkadeepnc.github.io/projects/collocated_vision_touch/img/ICRA_RAL_FINAL_Formatted-final-compressed.pdf) · [Elandt](https://arxiv.org/pdf/1904.11433) · [Leins](https://arxiv.org/html/2501.08077v1) · [Yuan](https://people.csail.mit.edu/yuan_wz/GelSight1/ICRA15_2740_FI.pdf)

**근거의 강도와 겉으로 상충하는 결과.** 영상에 접촉 위치 단서가 있고 분포 접촉이 실물에서 공간 구조를 가진다는 점에는 직접 관측 근거가 있다. 고정한 접근 장치에서 얻은 예측이 새 장비·표면에서도 유용하다는 근거는 조건부다. 예컨대 Kim의 optical flow는 환경 물체가 접촉 후 움직인 단서를 활용할 수 있고, Chaudhury의 접촉 전 추정은 그런 단서를 쓸 수 없다. Leins의 network와 PCA 순위가 물체에 따라 바뀌는 결과 역시 어느 한 해법이 접촉 추정 전체에서 우월하다고 보기 어렵게 한다. 이 논문들에서 v45의 Brier 개선량이나 실제 pad의 contact-area 효과를 직접 추정할 근거는 부족하다. [Kim](https://proceedings.mlr.press/v229/kim23b/kim23b.pdf) · [Chaudhury](https://arkadeepnc.github.io/projects/collocated_vision_touch/img/ICRA_RAL_FINAL_Formatted-final-compressed.pdf) · [Leins](https://arxiv.org/html/2501.08077v1)

### 남은 연구 질문과 검증 방법

| 질문 | 근거 및 중요성 | 바꿀 변수와 평가 |
|---|---|---|
| 1. 실제 pre-impact 영상에서도 위치 선택의 이득이 남는가? | 문헌의 contact annotation은 이미 발생한 사건을 포함한다. [Kim](https://proceedings.mlr.press/v229/kim23b/kim23b.pdf) | 접촉 직전 frame만 전달하고 뒤의 force trace로 결과를 가린 채 채점한다. |
| 2. 영상에서 하나의 x 대신 위치 분포를 쓰면 임계점 부근의 손실이 줄어드는가? | Occlusion에서 단일 위치가 불확실하다. [Kim](https://proceedings.mlr.press/v229/kim23b/kim23b.pdf) | 가림·시점을 조작해 hard routing과 위치 posterior를 적분한 routing을 paired Brier로 비교한다. |
| 3. 곡선으로 접근하면 경로 외삽이 언제 실패하는가? | 직선 접근·영상 특징은 pre-contact localization의 가정이었다. [Chaudhury](https://arkadeepnc.github.io/projects/collocated_vision_touch/img/ICRA_RAL_FINAL_Formatted-final-compressed.pdf) | 같은 최종 위치에서 경로 곡률과 가속도만 바꾸고 좌표·route 오차를 함께 잰다. |
| 4. Camera를 다시 장착한 날에도 보정이 유지되는가? | 위치 추정에는 optical geometry와 교정이 개입한다. [Chaudhury](https://arkadeepnc.github.io/projects/collocated_vision_touch/img/ICRA_RAL_FINAL_Formatted-final-compressed.pdf) | 재장착 session을 독립 표본으로 삼고 drift 보정 전후 Brier를 비교한다. |
| 5. 같은 최초 접촉 위치에서 자연 형성되는 contact area도 같은 효과를 내는가? | Hydroelastic contact는 물체 geometry와 deformation을 이용한다. [Elandt](https://arxiv.org/pdf/1904.11433) | 같은 tip과 위치에서 pad compliance·normal load를 교차해 pressure film 실측과 대조한다. |
| 6. 같은 effective area에서 pressure profile의 꼬리만 달라도 효과가 달라지는가? | 같은 patch의 분포 모양이 wrench에 관여한다. [Elandt](https://arxiv.org/pdf/1904.11433) | 균일·Hertz형·annular profile의 총하중과 effective area를 맞춰 비교한다. |
| 7. Pressure sensor의 공간 해상도가 area 효과 추정에 필요한 수준은 어느 정도인가? | Sensor의 receptive region으로 압력을 합칠 때 공간 정보가 달라진다. [Leins](https://arxiv.org/html/2501.08077v1) | 같은 충돌을 여러 taxel 해상도로 재구성하고 물체별 area 오차를 잰다. |
| 8. Rotation을 허용하면 중심을 맞춘 면적 효과가 커지는가? | 접촉면 분포가 torque를 전달할 수 있다. [Elandt](https://arxiv.org/pdf/1904.11433) | 동일 질량과 물체에서 angular inertia·회전 구속을 교차하고 angular slip과 Brier를 기록한다. |
| 9. 국소 incipient slip과 전체 gross slip 중 어느 쪽이 영상 위치로 예측되는가? | 둘은 실제 patch에서 다른 시점에 일어난다. [Yuan](https://people.csail.mit.edu/yuan_wz/GelSight1/ICRA15_2740_FI.pdf) | Tactile flow와 중심 속도로 두 label을 만들어 같은 decoder를 각각 평가한다. |
| 10. 재료를 보지 못한 상황에서 어느 정도의 실물 교정이 필요한가? | Synthetic tactile transfer의 성능 순위가 실제 물체마다 달랐다. [Leins](https://arxiv.org/html/2501.08077v1) | Surface session을 통째로 hold out하고 교정 0·4·16회에서 고정 규칙과 재학습을 비교한다. |
