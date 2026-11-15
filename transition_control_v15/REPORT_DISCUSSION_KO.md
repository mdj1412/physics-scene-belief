## 9. 결과 해석과 논문 기여 가능성

### Transition이 존재한다고 adaptive selection이 필요한 것은 아니다

Matched contact event에서 LMMSE는512 worlds 중501개에서 query_view를 선택했고 pair는418개만 선택했다. Pair는34개에서 factual_view,60개에서 probe를 선택했다. 이 task에서는 query 근처 geometry를 직접 보는 fixed rule이 매우 강했다. Always-query의 loss는 LMMSE와 수치상 거의 같았다. Event probability가 `[.2,.8]`인 ambiguous370 worlds에서도 LMMSE와 nested512의 choices로 얻은 loss는 동일했고, pair는 평균적으로 더 나빴다.

따라서 contact boundary가 불연속이라는 사실과 LMMSE의 action ranking이 실패한다는 것은 다른 주장이다. 현재 sensor menu에서는 query_view가 필요한 정보를 직접 제공한다. Ambiguity를 넣었다는 이유만으로 더 복잡한 acquisition의 필요성이 생기지 않았다.

Stick event에서는 더 복잡한 pattern이 나타났다. Low-linear-gain205 worlds에서 nested512−LMMSE mean difference는−0.010651이었지만 secondary95% CI는[−0.022597,+0.001009]였다. Nonlinear acquisition이 도움이 될 가능성은 남아 있으나, 이 subgroup만으로 확정할 수 없다. Pair는 이 subset에서도 개선을 보이지 않았다. Ambiguous134 worlds에서는 pair−LMMSE difference가+0.028715였고 secondary CI도 악화를 지지했다.

### Model mismatch와 finite integration이 설명하는 부분

동일한 shifted data에서 pair의 nominal−correct-law smooth-velocity difference는+0.003468이었다. 그중 decoder term이+0.003124, selection term이+0.000344였다. 반대로 contact-event total difference +0.006279 중 selection term은+0.004886, decoder term은+0.001393이었다. Correct-law 정보가 필요한 위치가 task마다 다르다. 이는 선택한 decomposition path의 결과이며 원인이 완전히 독립적이라는 뜻은 아니다.

QMC16의 actual-observation 이후 ESS도 충분히 큰 것은 아니었다. Matched pair choices에서 contact-event median ESS는26.47, stick-event는20.66이었고 각각 약31.1%,47.9%의 world/noise combinations에서 ESS<20이었다. QMC64를 같은 actions의 decoder로 쓰면 median ESS가102.17/76.69로 올라갔다. 따라서 finite integration error를 무시할 수 없다.

그렇지만 QMC64에서도 pair의 contact/stick event losses는0.057803/0.072897로 native의0.051285/0.066871보다 평균적으로 높았다. Numerical resolution을 높이는 것만으로 현재 selector의 우위가 나타난 것은 아니다. 이 큰-bank 비교는 independent scramble도 바뀌므로 pure sample-count effect와 구분한다.

Ranking audit에서 contact-event nested choices는 first32 worlds에서 안정적이었다. Stick-event는 nested512와 다른-seed nested2048의 action agreement가75%, QMC64 nested512와는68.75%였다. 이 차이를 숨기지 않는다. 다만 같은 audit의 independent-seed reference 내부 평균 regret는 nested512가0.000118, pair가0.005667이었다. Action disagreement 비율만으로 decision impact를 판단해서도 안 된다. Reference 자체가 finite approximation이므로 true Bayes regret라고 부르지 않는다.

### Pair score의 구조적 약점: decision-region mixtures의 overlap

현재 score는 query predictions가 다른 **개별 hypotheses pairs**가 observation에서 멀리 떨어지는지 계산한다. 그러나 개별 pairs가 멀어도, 각 decision region의 전체 observation distributions가 동일할 수 있다. 이 경우 실제로는 query label에 대한 information이 없다.

이를 확인하기 위해 primary 결과를 본 뒤 별도의 **abstract explanatory counterexample**을 구성했다. Physics test를 추가로 튜닝한 것이 아니다. Factual likelihood가 모두 같은 네 hypotheses에 prior1/4씩, binary query labels `[0,0,1,1]`을 둔다. Noise SD는1이다.

- Uninformative action의 observation means: `[-4,+4,-4,+4]`. 두 labels의 conditional mixture가 정확히 동일하므로 observation 이후에도 label probability는1/2이고 Bayes Brier loss는0.25다.
- Informative action의 means: `[-0.5,-0.5,+0.5,+0.5]`. 두 labels의 distributions가 달라 observation이 query prediction을 개선한다.

그런데 구현한 pair score는 첫 action에 약0.249916, 두 번째에 약0.058752를 주어 uninformative action을 더 높게 평가한다. Equal factual likelihood라 cutoffs를 바꾸는 것으로 이 반례가 사라지지 않는다. Brier-scaled target에서 첫 action score는 `¼(1−exp(−8))`, 두 번째는 `½(1−exp(−0.125))`로 직접 계산된다. Expected losses는 서버에서 Gauss–Hermite80 quadrature로 확인했다.

이 반례는 현재 pair objective가 exact Bayes-risk reduction이 아님을 구체적으로 보여 준다. Physics worlds의 모든 오류가 이 mechanism으로 발생했다고 입증한 것은 아니다. 하지만 score 자체가 decision-region overlap을 처리하지 못할 수 있다는 구조적 limitation은 있다. Contact/stick test에서 다른 observations로 자주 전환하면서 loss가 높아진 현상과 함께 검토할 근거가 된다.

Factual-fit band ablation의 개선은 두 matched event tasks에서 secondary CI가0을 포함했다. Query weighting은 contact-event에서 agnostic discrimination보다 분명히 나았지만, 그것만으로 LMMSE/fixed rule 대비 contribution이 성립하지는 않는다. Stick-event pair fallback은137/512였으므로, zero committee score에서는 LMMSE로 되돌아가는 경우도 적지 않았다. Fallback이 모두 true certainty라는 뜻은 아니며 finite committee가 rare outcomes를 놓칠 가능성도 남는다.

### 선행연구와 구분해야 할 주장

EC²는 hypotheses와 equivalence classes를 기반으로 information gathering을 다룬다. “서로 다른 decision을 만드는 hypotheses를 구별한다”는 발상은 기존 연구 맥락 안에 있다. 이번 Gaussian overlap pair score는 EC²의 objective를 그대로 구현한 것이 아니며, 그 논문의 adaptive-submodularity guarantees를 가져올 수 없다. [Golovin, Krause, Ray, 2010 원문](https://proceedings.neurips.cc/paper/2010/file/1e6e0a04d20f50967c64dac2d639a577-Paper.pdf)

GoBOED는 amortized posterior와 differentiable convex decision layer를 결합해 goal-driven design을 수행한다. 현재 discrete sensor menu의 contact-event forecasting과는 다르지만, downstream objective에 맞춘다는 점 자체는 novelty가 아니다. 이 연구의 learned method를 직접 재현하여 비교한 것은 아니다. [Go, Qian, Yoon, 2026 preprint](https://arxiv.org/html/2605.26093v1)

GBOED는 Gibbs inference와 Gibbs EIG를 통해 inference/design 양쪽의 robustness를 다루며, outliers와 noise misspecification을 평가한다. Maximin robust Bayesian experimental design은 adversarial perturbation과 robust belief update를 함께 정식화한다. 이번 contact-law mismatch setting과 동일하지 않으며 두 방법을 benchmark로 재현하지 않았지만, model uncertainty에 대응하는 observation selection이라는 주장에도 선행연구와의 구체적 차이가 필요하다. [Barlas, Sloman, Kaski, 2025 preprint](https://arxiv.org/html/2511.07671v1), [Abdulsamad et al., 2026 preprint](https://arxiv.org/html/2603.14094v1)

### 내가 권하는 연구 판단

현재 score의 cutoffs나 K를 더 탐색하여 positive result를 얻는 방향은 멈추는 편이 좋다. Matched control에서의 악화, optimized compute comparison, explicit mixture-aliasing counterexample를 함께 보면 현재 objective를 주된 novelty로 삼을 근거가 약하다. Negative evidence를 정리한 diagnostic 결과로 보존할 가치는 있지만, 이것만으로 major-conference method contribution이 확보됐다고 볼 수 없다.

계속한다면 **decision-region별 conditional predictive mixture를 보존하는 compact acquisition**이 더 타당한 가설이다. 개별 pair distances 대신 `p(y|decision region, factual evidence)`의 overlap과 actual expected decision loss를 근사해야 한다. 적어도 위 counterexample에서는 uninformative action의 utility를0으로 판정해야 한다. 이는 새로운 구현·성능이 확인된 결과가 아니라 후속 설계 기준이다.

그 다음의 go/no-go 조건도 명확해야 한다. 먼저 independent physics worlds에서 optimized LMMSE와 task-specific fixed rule이 실제로 실패하는 regime를 확보하고, conditional-mixture approximation이 cross-fitted nested reference의 decision quality를 더 적은 compute로 유지하는지 검증해야 한다. 적절한 non-inferiority margin은 application loss tolerance에 따라 실험 전에 정한다. Well-specified exact Bayes-optimal selector를 accuracy로 이겨야만 approximation이 가치 있는 것은 아니다. 그러나 이번 pair는 cheap baselines에도 밀렸으므로 그 일반론으로 현재 결과를 성공으로 바꾸지는 않는다.

이번 실험은 raw RGB, realistic trajectory simulation, real robot intervention을 검증한 것이 아니다. Geometry profiles, finite-reach gate, instantaneous contact, equal action costs를 사용한 analytical simulation이다. 실제 논문으로 확대하려면 mixture mechanism의 physical relevance와 realistic sensing/action costs를 별도로 입증해야 한다.
