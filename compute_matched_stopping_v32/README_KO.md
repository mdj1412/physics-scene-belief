# V32 · 같은 GPU 학습 시간으로 어떤 접촉 이력에 label을 붙일 것인가?

예측 대상은 하나의 환경에서 이미 관측한 접촉 이력 H다. 현재 과정에서 멈추거나 다음 2회 또는 4회의 접촉을 구매하면, 이후 **다른 128개 물체**에서 어떤 관측을 선택할지 달라진다. 각 물체에서 정확한 boundary profile인 query view 또는 다른 입사 속도의 충돌 뒤 2D outgoing velocity인 probe를 얻고, 별도 충돌에서 stick할 확률을 예측한다. 128개 중 13개에만 probe를 쓸 수 있다. 각 training label은 같은 128개 물체·정답·관측 noise로 구한 **Brier(멈춤) − Brier(계속)**의 평균이다. 양수이면 더 접촉한 덕에 이후 예측이 나아졌다. 최종 policy 비교에는 Brier loss 자체를 쓰므로 작을수록 좋다.

- **여러 이력에 한 번씩 측정하는 1×1**에서는 매 prefix에서 독립적인 추가 접촉 경로 한 개와 독립적인 배포 집단 한 개로 gain을 구한다. **이력당 2×2**에서는 같은 prefix에서 접촉 경로 두 개와 128개씩의 배포 집단 두 개를 새로 뽑아 조합 네 개의 gain을 평균한다. 한 행은 같은 접촉 경로를, 한 열은 같은 배포 집단을 공유한다. 서로 다른 두 arm의 이력과 bank seed는 독립적으로 뽑는다.
- 모든 calibration 물체 1,024개의 noisy factual evidence를 먼저 본다. 그중 이미 지불한 2회 또는 4회의 접촉이 prefix다. 추가 접촉은 같은 pool의 아직 구매하지 않은 물체에서 무복원 추출한다. 각 stage의 continuation은 독립적으로 표집한다. 각 후보는 자신의 prefix에 포함된 정보로 score를 내며, 평가 simulator만 hidden geometry·friction·restitution·target label을 볼 수 있다.
- 두 scorer 모두 Coulomb law에서만 학습한 stage별 15-feature ridge regression이다. 입력은 geometry·velocity proxy coupling의 posterior mean/SD와 그 함수, 이미 지불한 접촉 응답의 통계다. v30 feature 정의를 재사용하고 λ=3, train-only normalization, unpenalized intercept를 공통으로 고정했다. 두 training corpora는 label 생성과 fit에 같은 540 GPU wall-seconds를 배정한다. 약 0.5초를 fit에 남기고 새 환경을 추가할 수 없을 때 멈춘다. 종료 결정은 경과 시간과 최근 처리 속도만 본다. 두 방식의 실제 label 수와 소비 시간을 결과에서 확인한다.
- Held-out에서는 같은 여덟 환경에 대해 전체 stopping policy를 적용한다. Adaptive methods는 모두 두 번씩 접촉한 뒤 score 상위 네 곳에 두 번씩 더 주고, 이 네 곳의 **업데이트된 score** 상위 두 곳에 네 번씩 더 준다. 결과는 환경당 2·2·2·2·4·4·8·8회다. Fixed 4는 모두 네 번씩, uncertainty-only는 각 stage에서 friction coupling의 posterior SD가 큰 순서로 같은 방식으로 배정한다. 모든 policy가 calibration 32회, deployment 104회를 써서 총 136회다. 각 block에서 어느 환경이 8회까지 갈지는 해당 policy가 직접 정한다.
- 세 평가 법칙에 한 번도 학습하지 않았다. Static–kinetic split은 static/kinetic friction을 분리하고, speed weakening은 접선 속력에 따라 마찰계수가 변한다. 두 법칙은 이전 버전의 연구자가 이미 분석했다. 이번에 새로 정한 directional law는 접선 속도 부호에 따라 μ_eff = μ[1+0.23 tanh(vₜ/0.28)]로 바뀐다. 이 설정에서 factual collision은 stick을 유지하되 calibration probe는 slide하고 별도 query collision의 label이 달라질 수 있다. 세 법칙에서 모두 acquisition과 common decoder는 source Coulomb을 가정한다. Common decoder는 환경의 실제 population coupling을 받는 privileged 기준이다. 실제 deployment에서 이 계수를 정확히 안다고 가정할 근거는 따로 필요하다.
- 법칙마다 독립적인 16개 block, block마다 여덟 coupling 환경, 환경마다 128개 deployment worlds를 생성한다. 총 384개 환경, 49,152개 물체다. Noise 32회와 같은 물체의 여러 potential observations는 독립 표본 수에 더하지 않는다. 정책 할당 및 observation choices를 파일에 저장한 다음 ground truth와 sensor observations를 평가한다. 법칙별 16개 block을 함께 재표집하고 세 법칙에 같은 가중치를 준다.
- GPU p90은 여덟 환경 전체(1,024개 future worlds)의 wall latency다. 각 법칙의 처음 6개 block에서 네 정책의 순서를 무작위로 섞고 warm-up 한 회 뒤 세 회씩 측정한다. Full 경로는 이미 도착한 factual 및 지불한 probe response에서 시작해 매 정책마다 posterior, stopping decision, selector bank, acquisition, 선택한 관측 decoding을 실제로 다시 실행한다. Warm 경로는 같은 run 안에서 bank 준비가 끝난 시점부터 확률이 host에 도착할 때까지다. 두 경로 모두 CUDA를 동기화하고, 모든 action과 1,024개 확률을 사전 고정한 결과와 대조한다. 실제 sensor/reset 대기, 평가 데이터 생성, 공통 detector와 privileged common decoder 초기화는 포함하지 않는다.

Directional law는 물리적으로 가능한 shift를 만드는 합성 stress test다. Yu et al.은 네 표면에서 sliding 방향을 바꾸어 마찰을 실측했고 polyurethane에서 가장 큰 마찰과 가장 작은 마찰의 비가 약 1.5였다고 보고했다. 이 관측은 방향을 바꾸어 검증할 동기를 준다. 여기 쓴 signed velocity 함수와 계수를 그 자료에 fit하지 않았고 실제 충돌에서의 Brier를 검증한 것도 아니다. [Yu et al., §V, Fig. 6](https://arxiv.org/pdf/1604.04038)

$$
G_{f,b}(H) = R_b(H) - R_b(H \mathbin{+} C_f),\qquad
\widehat{G}_{2\times2}(H) = \tfrac14\sum_{f=1}^2\sum_{b=1}^2 G_{f,b}(H).
$$

여기서 H는 이미 지불한 접촉 이력, C는 같은 pool에서 구입한 추가 접촉, R은 그 이력에서 결정한 관측으로 얻은 deployment Brier다. 2×2를 평균해도 source simulator, 잘못 지정된 contact law, 공통 decoder 또는 K256 근사가 만드는 체계적 오차는 남을 수 있다.
