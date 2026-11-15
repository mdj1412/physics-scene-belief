# Test 이전 pilot 변경 기록

첫 설정(a∈[-.5,.5], query incoming velocity=(1,.6))의 8 pilot worlds에서 모든 action이 query-view였다. 해당 결과와 최초 생성 자료는 `pilot_initial_results/`에 보존했다. Test outcomes와 test private truth를 열어 성능을 비교하지 않았다.

32 pilot worlds에서 query incoming velocity=(1,.9)를 고정하고 amplitude bound .10/.18/.28을 비교했다. Query-view/probe 빈도는 각각21/11,25/7,32/0이었다. Action이 가장 균형 있게 경쟁한 .10을 선택했다. 이는 실험 난이도 설정을 위한 pilot selection이며 현실에서 이 조건이 얼마나 자주 나타나는지 주장할 수 없다.

최종 protocol은 a∈[-.10,.10], query v=(1,.9)로 변경한다. 나머지 visibility, sensor precision, contact candidate set과 independent split seeds는 유지한다. 최종 자료는 새로 생성한다. Original pilot 결과와 scan 결과를 숨기지 않는다. Test worlds는 filtering하지 않는다.

Main acquisition quadrature order는 final pilot convergence를 보고 결정한다. Final decoder는 별도 high-order quadrature로 검증한다. Plug-in의 hard model selection은 mode probability가 비슷할 때 numerical discontinuity를 가질 수 있으므로 Bayes prediction의 convergence와 구분한다.

Learned rules에는 raw profile 9 rows를 span(y,y²,y⁴)의 orthonormal basis로 projection한3 features와 factual velocity2 features를 제공한다. 두 candidate family의 image mean이 이 span 안에 있고 noise가 isotropic Gaussian이므로 이는 likelihood 정보를 버리지 않는 sufficient statistics다. 작은 training set에서 pure-noise dimensions 때문에 rule이 불리해지는 것을 방지한다. Joint inference는 원래9 rows를 그대로 사용한다. RBF와 CART의 hyperparameters는 validation expected risk로 선택한다.

Test 이전에 더 강한 mixture plug-in baseline을 추가한다. 각 (geometry family, contact class)의 conditional parameter mean에서 forward response를 구한 뒤 posterior model probability로 평균한다. Continuous uncertainty와 within-mode covariance는 prediction에서 적분하지 않지만 discrete model uncertainty는 보존한다. Acquisition은 이 decoder의 expected risk에 맞춘다. Posterior moments는 동일 quadrature로 계산하므로 inference compute 절감 방법이라고 주장하지 않는다.

Primary contrasts는 joint−none, joint−best learned rule, joint−hard plug-in, joint−mixture plug-in, joint−factorized의5개로 확정한다. Family-wise nominal95%를 위해 각각99% Bonferroni CI를 사용한다. Original protocol의4 contrasts/98.75%는 이 addendum으로 대체한다.

Final pilot의21/31/41/61 orders에서8 worlds의 acquisition choices가 모두 같았다. Main acquisition은 order41, outer H256으로 고정한다. Actual decoder는 order61이며 별도로 높은 order와 비교한다. Pilot의 hard plug-in prediction에는 model choice discontinuity가 있어 order41→61 최대 차이가0.0295 m/s였으므로, 모든 decoder가 완전히 수렴했다고 간주하지 않는다. 이 수치는 Bayes mean과 hard plug-in이 섞인 최대값이다. Main 결과에서 두 항목을 분리하여 다시 확인한다.
