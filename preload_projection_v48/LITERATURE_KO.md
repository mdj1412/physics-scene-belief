## 선행연구 검토와 연구 질문

여섯 편을 세 질문으로 읽었다. 약한 접촉에서 무엇을 실제로 측정할 수 있는가, 얻은 측정값을 예측에 어떻게 사용할 수 있는가, simulator에서의 이득을 어디까지 일반화할 수 있는가이다. 아래 표의 표본 단위와 endpoint는 서로 다르다.

| 연구 | 질문과 방법 | 표본·데이터 | 결과, 한계, v48과의 관계 |
|---|---|---|---|
| Su et al., 2012 | BioTac의 force feedback으로 접촉 방향을 맞추고 indentation으로 compliance를 구별할 수 있는가? | Barrett arm, aluminum plate 1개와 서로 다른 rubber 4종, 목표 normal force 10 N | 상대 compliance를 구별했다. 낮은 force 구간에서는 sensor skin과 joint cable의 변형이 측정값에 섞였다. v48이 가정한 0.4 N에서 signed displacement를 읽을 수 있는지는 독립적으로 확인해야 한다. [원문](https://pmc.ncbi.nlm.nih.gov/articles/PMC3405524/) |
| Van der Merwe et al., RSS 2023 | Partial point cloud와 force/torque로 loaded geometry와 contact patch를 동시에 추정할 수 있는가? Simulated supervision으로 Neural Deforming Contact Fields를 학습했다. | 실물에 46 mm sponge, 두 depth camera, force/torque sensor를 사용했다. Material identification에 flat press 16회를 사용했다. | 실물에서 point-cloud baseline보다 나은 재구성을 보고했다. 여러 sensor와 별도 material fitting을 사용했으므로 one-touch future-slip 예측의 근거로 바로 옮길 수 없다. [원문](https://www.roboticsproceedings.org/rss19/p080.pdf) |
| Pfrommer et al., CoRL | 물리적 contact를 반영한 learned dynamics가 impact와 stiction을 예측하는가? Signed distance와 contact-frame Jacobian을 학습했다. | 실제 cube toss 750회 중 정제한 570개 trajectory | 구조화한 모델에서 물리적으로 타당한 trajectory 예측을 보였다. v48의 두 decoder 중 고르기보다 관측을 outcome에 직접 연결하는 대안의 근거다. Soft surface에서 local preload의 순가치는 별도 문제다. [원문](https://proceedings.mlr.press/v155/pfrommer21a/pfrommer21a.pdf) |
| Mozannar & Sontag, ICML 2020 | 입력별로 예측기와 expert 중 하나를 선택하는 학습 목표를 어떻게 구성하는가? Cost-sensitive surrogate와 일관성을 분석했다. | 이론 및 일반 classification 실험. 물리 접촉 표본 없음 | 상대 오류로 두 predictor 중 하나를 고르는 원리는 이미 연구됐다. 물체를 실제로 한 번 더 만지는 비용이나 측정 위치는 별도 문제다. [원문](https://proceedings.mlr.press/v119/mozannar20b.html) |
| Luo et al., 2026 preprint, PhysiCalWorld | Exploratory interaction clip을 physics descriptor로 압축해 물체별 dynamics에 맞출 수 있는가? | 네 환경에서 action-conditioned rollout과 planning을 평가했다. | 탐색 후 예측이 개선됐다. Clip의 정보량과 수집 비용, latent rollout 단위가 0.4 N scalar/Brier 실험과 다르다. 같은 prefix를 동일 budget으로 제공한 비교가 필요하다. [원문](https://arxiv.org/abs/2610.30125) |
| Acosta et al., RA-L 2022 | Simulator가 실제 impact trajectory를 어느 정도 재현하는가? 시스템별 contact parameter를 fitting하고 Drake·MuJoCo·Bullet을 비교했다. | 실제 tossed cube와 Cassie landing, 두 물리 시스템 | 탄성 regime와 운동에 따라 정확도가 달랐다. v48의 quadrature 수렴은 solver 내부 수치 검증이다. 실제 약한 preload와 이후 impact 사이의 부호 관계는 검증하지 않는다. [원문](https://arxiv.org/abs/2110.00541) |

**지식 지도.** Force command → 실제 material/tool compliance → loaded displacement → 관측으로 추정한 state → 미래 stick 확률이라는 경로에서, Su et al.은 앞 두 단계의 계측을, NDCF는 loaded geometry 추정을, ContactNets는 contact dynamics 예측을 다룬다. PhysiCalWorld는 interaction history에서 예측에 필요한 정보를 학습한다. V48은 한 관측을 두 가지 방식, 즉 decoder 선택과 직접적인 probability update에 넣어 같은 endpoint를 채점한다. Deferral의 일반적 원리와 이 물리 실험의 기여를 구분해야 한다. [Su et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC3405524/) · [NDCF](https://www.roboticsproceedings.org/rss19/p080.pdf) · [ContactNets](https://proceedings.mlr.press/v155/pfrommer21a.html) · [PhysiCalWorld](https://arxiv.org/abs/2610.30125) · [Deferral](https://proceedings.mlr.press/v119/mozannar20b.html)

**근거의 강도와 겉보기 충돌.** 촉각 탐색이 재질이나 변형에 대한 정보를 준다는 점은 실물 연구들이 공통으로 뒷받침한다. 한 번의 매우 약한 preload가 이후 큰 하중에서의 부호를 충분히 알려주는지는 재료·센서·위치에 조건부다. V48의 높은 sign accuracy와 Su et al.의 낮은 force에서 겪은 어려움은 모순되지 않는다. 후자는 실제 fingertip과 actuator deformation이 섞인 displacement를 측정했고, 전자는 두 fiducial의 차이와 명시한 noise model을 사용했다. 고정 위치와 query 위치의 비교, sensor 비용 및 직접 predictor까지 동일 조건으로 맞춘 real-world evidence는 이 문헌 집합에서 부족하다. [Su et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC3405524/) · [실물 impact 검증](https://arxiv.org/abs/2110.00541)

### 다음 연구 질문 10개

| 질문 | 메울 gap, 변수와 검증 방법 |
|---|---|
| 1. 0.4 N에서 실제 표면 변형의 부호가 읽히는가? | Force를 단계별로 바꾸고 unloaded/loaded fiducial, 독립 force transducer, 기구 자체의 displacement를 함께 기록한다. Material session별 sign accuracy와 detection limit를 구한다. [Su et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC3405524/) |
| 2. 작은 preload와 실제 impact 사이에서 부호가 몇 번 뒤집히는가? | Backing stiffness와 표면 마감을 독립 변경한 시험편에서 weak/strong load를 짝짓고 부호 반전율·Brier를 측정한다. [NDCF](https://www.roboticsproceedings.org/rss19/p080.pdf) · [실물 impact](https://arxiv.org/abs/2110.00541) |
| 3. Fixed probe를 얼마나 떨어뜨려도 유효한가? | 0–12 mm 이상의 probe 거리별 spatial covariance와 policy gain을 독립 시험편에서 측정한다. 실제 거리에서 v48 transfer sweep의 가정을 검증할 수 있다. [NDCF](https://www.roboticsproceedings.org/rss19/p080.pdf) |
| 4. 더 저렴한 force channel로 같은 정보를 얻는가? | 같은 preload trajectory에서 differential displacement, force/torque, 두 신호 결합을 비교하고 calibration 시간까지 센다. [Su et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC3405524/) · [NDCF](https://www.roboticsproceedings.org/rss19/p080.pdf) |
| 5. Learned contact decoder에 직접 관측을 주면 routing이 더 필요한가? | Original, projected, smooth direct update와 contact-aware learned dynamics에 같은 source budget과 held-out physical object를 주고 Brier 및 calibration을 비교한다. [ContactNets](https://proceedings.mlr.press/v155/pfrommer21a.html) |
| 6. 어떤 물체는 만지지 않아도 되는가? | No-touch, fixed one-touch, expected-value stopping을 같은 물체에 되풀이하고 Brier 감소와 측정·reset 비용의 실측 교환비를 추정한다. [Deferral](https://proceedings.mlr.press/v119/mozannar20b.html) · [PhysiCalWorld](https://arxiv.org/abs/2610.30125) |
| 7. Preload가 다음 query를 바꾸는가? | No-touch와 preload trial 순서를 무작위화하고 독립 reset·휴지시간을 조작해 hysteresis 및 마모가 event 자체에 미치는 영향을 측정한다. [NDCF](https://www.roboticsproceedings.org/rss19/p080.pdf) · [실물 impact](https://arxiv.org/abs/2110.00541) |
| 8. 동일 측정 예산에서 짧은 interaction clip이 scalar보다 더 가치 있는가? | 동일 trajectory에서 단일 수치와 limited-frame encoder를 학습하고 촬영·전송·훈련 비용까지 맞춘다. [PhysiCalWorld](https://arxiv.org/abs/2610.30125) |
| 9. Binary threshold를 바꿔도 효과가 유지되는가? | Slip speed 연속값, threshold별 Brier, 실제 pose tracking의 해상도를 함께 보고한다. [ContactNets](https://proceedings.mlr.press/v155/pfrommer21a.html) · [실물 impact](https://arxiv.org/abs/2110.00541) |
| 10. 완전히 새 재료와 날짜에서 이득이 남는가? | Probe-site 선택과 preprocessing을 source에서 고정하고, material family·tool·session·날짜를 통째로 제외해 순차적으로 재현한다. [Su et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC3405524/) · [실물 impact](https://arxiv.org/abs/2110.00541) |
