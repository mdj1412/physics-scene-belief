## 핵심 판단

**Visibility와 query를 고정해도 factual evidence에 따라 유리한 observation이 달라지는 조건을 만들었다. 하지만 strong rule과 mixture plug-in을 넘어서는 full joint posterior의 추가 이득은 이번 sample에서 확정하지 못했다.**

Main test의 joint selector는 query-view155회, probe101회를 선택했다. 모든 world에서 visibility/query/sensor precision이 같으므로 v10의 visibility + query lookup만으로 이 선택을 설명할 수 없다. 다만 finite outer sampling으로 정한 선택이므로, 실제 conditional risk의 차이가 안정적인지는 아래 independent ranking audit과 함께 봐야 한다.

추가 observation은 유효했다. Joint MSE는 **0.005762328**, no-observation은 **0.011540406**이고 primary99% CI에서도 개선이 확인됐다. Updated hard plug-in은 **0.006350650**으로, joint가 약9.26% 낮았으며 이 차이도 primary99% CI가0을 포함하지 않았다.

반면 **updated mixture plug-in은0.005771137**로 joint에 매우 가까웠다. 두 selector의 선택도256/256 동일했다. Joint−mixture의99% CI는0을 포함하므로 추가 이득을 확정할 수 없다. 이는 equivalence가 증명됐다는 뜻도 아니다.

Strong learned rule의 MSE는 **0.005981087**, LMMSE는 **0.005835918**이었다. Joint가 평균에서는 낮지만, learned rule 대비 primary99% CI와 LMMSE 대비 descriptive95% CI 모두0을 포함했다. **비싼 nested acquisition이 강한 간단한 selector보다 실제로 낫다는 주장은 지지되지 않았다.** Standard와 Cross-fitted SNIS의 actual MSE도 모두0.005741478로, joint selector보다 평균이 약간 낮았다. 이 미세한 차이를 SNIS의 우위로 해석해서도 안 된다.

Factorized factual belief는 **0.006585476**으로 joint보다 평균 loss가 컸지만, 사전에 정한 primary99% CI는0을 포함했다. 따라서 “geometry–contact correlation을 제거하면 반드시 성능이 유의하게 나빠진다”는 결론은 보류한다. Correlation의 존재와 그것을 보존할 때의 statistically resolved decision benefit은 서로 다른 주장이다.

마지막으로, mixture plug-in도 joint inference로 mode weights와 conditional means를 구한다. 이 결과는 **joint inference 전체가 불필요하다는 증거가 아니라**, 현재 조건에서는 각 mode 안의 continuous distribution을 prediction 단계에서 끝까지 적분하는 이득이 작다는 증거다. Compact mixture만 유지하는 별도 inference algorithm의 성능이나 compute 절감은 아직 측정하지 않았다.
