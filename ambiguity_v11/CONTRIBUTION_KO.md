## 논문 기여 가능성에 대한 판단

**문제 설정은 v10보다 좋아졌지만, “복잡한 joint-posterior acquisition이 강한 baseline을 이긴다”는 method paper의 핵심 주장은 아직 성립하지 않는다.** Visibility/query shortcut을 제거한 것은 진전이다. 하지만 depth3 tree, LMMSE, mixture plug-in이 경쟁력을 유지했다는 사실을 중심에 놓아야 한다.

### 이번 결과로 말할 수 있는 것

- 같은 visibility/query라도 factual evidence가 contact class를 이미 구분했는지에 따라 필요한 observation이 바뀐다. World000과023이 분명한 예다.
- Primary distribution에서는 추가 observation의 실제 이득이 있고, 하나의 discrete explanation을 고르는 updated hard plug-in 대비 joint predictive mean의 개선도 primary99% CI에서 확인된다.
- Discrete model uncertainty를 유지하는 mixture plug-in은 full posterior predictive mean과 매우 가까웠다. Observation 선택도 동일했다. 현재 설정에서 full predictive integration을 추가하는 이득은 작거나 unresolved다.
- Factorized belief의 평균 성능은 낮았지만 primary99% CI는0을 포함했다. 이 결과를 correlation preservation의 확정적 우위라고 포장하면 안 된다.

### 이번 결과로 말할 수 없는 것

- 새로운 Bayes acquisition objective를 제안했다고 말할 수 없다.
- Strong learned rule이나 LMMSE보다 실제 decision loss를 확실히 줄였다고 말할 수 없다.
- Mixture plug-in의 inference가 저렴하다거나 full joint inference를 완전히 대체했다고 말할 수 없다. 이번 mixture baseline도 joint posterior의 conditional moments를 사용한다.
- Natural RGB, arbitrary unseen shape, unknown physical law 전체를 해결했다고 말할 수 없다.
- Current quadrature/SNIS diagnostic을 기존 tempered SMC bridge의 새로운 우위 증거로 사용할 수 없다. 이번에 bridge algorithm을 비교하거나 개선한 것은 아니다.

### 내가 선택할 방향

지금부터 toy world를 더 복잡하게 만들어 full joint selector가 이기는 조건을 계속 찾는 것보다는, **어떤 uncertainty를 유지해야 decision이 유지되는지**를 연구 질문으로 삼는 편이 낫다. 현재로서는 다음 가설이 더 설득력 있다.

> 하나의 explanation으로 너무 일찍 collapse하는 것이 주요 손실 원인일 수 있다. 반면 full predictive integration과 expensive acquisition은, 소수의 model hypotheses와 적절한 posterior moments를 사용하는 방법으로 대체할 여지가 있다.

이 가설은 아직 compact inference algorithm의 성공을 증명한 것은 아니다. 다음 구현 후보는 **geometry/contact model별 작은 mixture와 component 내부의 joint moments**를 유지하고, observation 이후 갱신하는 방법이다. Quadrature에서 계산한 moments를 그대로 받아오는 방식이 아니라 독립적으로 inference/update해야 하며, 학습한 rule·LMMSE·full quadrature와 cost–loss trade-off를 비교해야 한다.

특히 다음 validation은 세 조건을 분리하는 것이 좋다.

1. **Known-family, matched sensor:** Compact mixture가 full reference의 prediction과 acquisition을 어느 정도 보존하는지 확인한다.
2. **Unseen boundary 또는 unseen contact law를 각각 변경:** Mixture가 confident wrong explanation에 고정되는지, 새 observation이 오히려 bias를 강화하는지 확인한다.
3. **Boundary measurements의 calibration shift:** Sensor precision을 정확히 안다는 가정을 풀어, geometry uncertainty가 physics uncertainty로 전파되는 문제가 실제로 남는지 확인한다.

Strong rule 대비 improvement를 다시 주장하려면, 먼저 minimum useful effect와 sample size를 정한 **새 independent replication**이 필요하다. 이번 CI를 본 뒤 같은 test set에 worlds를 계속 추가해 유의해지는 순간 멈추는 방식은 피해야 한다. 현재 CI는 small effect에 대해 충분히 좁지 않고, 한 번 학습한 rule의 training-seed variation도 포함하지 않는다.

### Stress test가 주는 시사점

Candidate family 밖의128 worlds에서 joint MSE는0.010975069, no-observation은0.014756974였다. 추가 observation의 평균 이득은 남았지만, learned rule(0.010803425)과 LMMSE(0.010660299)의 평균은 joint보다 낮았다. Mixture plug-in(0.011009937)은 여기서도 joint에 가까웠다. 따라서 “더 정교하게 posterior를 적분하면 model mismatch에서도 더 잘 된다”는 근거는 없다.

이것은 별도의 shifted distribution이며 geometry와 contact를 동시에 바꿨다. In-support와 stress test의 absolute loss 차이를 순수한 model mismatch의 causal effect로 해석할 수 없다. 또한 이 평균 순서를 통계적으로 확정된 method ranking으로 해석하지 않는다.

현 단계에서는 **controlled diagnostic과 representation ablation으로서의 가치가 있다. 새로운 acquisition method의 논문 기여로는 아직 부족하다.** 다음 투자는 full SMC/bridge의 규모 확대보다 compact multi-model belief의 독립적 구현, mismatch 원인 분리, 실제 geometry measurement에 대한 검증에 두는 편이 합리적이다.

