## 선행연구와 v38의 기여 범위

v37의 차별성 서베이 중 v38의 해석을 직접 바꾸는 원문을 다시 대조했다. 물성의 식별, 정보를 얻는 실험의 선택, 얻은 정보로 내리는 예측을 각각 무엇으로 채점했는지가 선정 기준이다.

| 연구 질문 | 방법과 평가 대상 | 저자들이 확인한 결과 | v38과의 관련성과 한계 |
|---|---|---|---|
| 물리적 질문에 답하기 위해 어떤 추가 실험이 필요한가? | Saha et al.은 sliding·bouncing·spring의 144개 parameter set에서 6개 VLM을 평가했다. 관측이나 질문만 바꾼 pair에 알려진 정답 실험이 있다. | Direct response는 정답 행동이 다른 image pair의 95.1–100%에서 같은 행동을 반복했다. 측정의 선택과 주어진 측정의 사용을 별도로 평가했다. | 획득과 이용의 분리에는 선례가 있다. 이 benchmark의 답·행동 정확도와 다른 물체의 noisy impact에 대한 Brier gain은 평가 대상이 다르다. [Saha et al., 2026 공개 원고](https://arxiv.org/html/2609.11022v1) |
| 측정의 목적을 미래 행동의 손실로 직접 정의할 수 있는가? | Rossa et al.은 design policy와 action policy의 expected future loss를 함께 학습했다. Source location과 stochastic pendulum 등에서 비교했다. | Joint samples에서 downstream loss를 최적화하는 방법을 유도하고 task별 실험에서 성능을 평가했다. | v38은 한 속도를 사전 고정했다. 새 optimal design algorithm의 검증보다 주어진 probe의 정보량과 frozen policy가 그중 회수한 양을 측정하는 데 해당한다. [ACTION-BED, 2026 공개 원고](https://arxiv.org/pdf/2606.23662) |
| 실제 충돌에서 rigid contact model이 설명하는 범위는 어디까지인가? | Fazeli et al.은 실물 impact data에서 여섯 rigid-body model의 parameter를 추정하고 impulse 예측 범위를 평가했다. | Sticking regime에서 여러 parameter가 같은 결과를 내며, 모델 family의 표현 범위에도 한계가 있었다. | 알려진 finite catalog와 정확한 noise distribution을 전제로 한 v38 Bayes reference는 그 catalog에서만 정확하다. 실물 friction의 정보 상한을 뜻하지 않는다. [Fazeli et al.](https://arxiv.org/pdf/1710.04979) |
| 실제 물체에서 sliding 관측을 물성 추정에 쓸 수 있는가? | Arslan Waltersson과 Karayiannidis는 gripper에서 force와 sliding velocity를 측정해 static·Coulomb friction과 contact radius를 추정했다. 실제 네 물체를 사용했다. | Sliding 중 online estimation을 시연하고 빠른 slip–stick에서 추정이 어려운 경우를 다뤘다. | Force·velocity time series의 높은 관측 밀도와 한 번당 noisy outgoing velocity만 주는 우리 조건의 차이가 성능 차이를 설명할 수 있다. [ISER 2025](https://arxiv.org/pdf/2507.06824) |
| 입력별로 예측 모델을 섞는 게 고정 가중치보다 나은가? | Yao et al.은 data-dependent partially pooled weights를 이론과 응용 예제에서 조사했다. | 모델별 예측 성능의 상대 순위가 입력에 따라 다를 때, 고정 결합을 유연한 결합으로 확장할 수 있음을 보였다. | Predictive distribution의 pooling과 v38의 log-odds 보정은 수식이 다르다. 두 후보 사이에서 가중치를 고른다는 아이디어 자체에는 선례가 있다. [Bayesian Hierarchical Stacking](https://arxiv.org/abs/2101.08954) |
| 다음 사건의 확률 예측을 무엇으로 채점해야 하는가? | Gneiting과 Raftery는 categorical forecast를 포함해 proper scoring rule을 이론적으로 분석하고 예측 사례를 다뤘다. | True distribution을 예보할 때 expected score가 최적이 되는 조건을 정리했다. | Binary squared loss에서는 conditional mean이 최적이다. Brier 차이와 실제 로봇 행동의 이익은 서로 다른 단위다. [JASA 원문](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf) |

**연구 관계와 상반돼 보이는 결과.** 실물 gripper의 friction 식별과 v38의 held-out Brier는 측정 장비, 표본, 목표가 모두 다르다. 한 표면에서 여러 force·velocity를 얻어 물성을 설명하기 쉬워도, 같은 표면의 다른 물체 하나가 stick할지는 이미 확보한 geometry와 deployment observation으로 상당 부분 정해질 수 있다. 이 경우 물성의 posterior가 크게 변해도 추가 Brier gain은 작다. 반대로 source decoder의 체계적 오차를 상쇄한 정책은 자기 자신의 약한 prefix-3 기준에서 크게 개선될 수도 있다. v38의 설계는 이들 설명을 비교하지만 실물 인과효과를 식별하지는 않는다. [물성 식별](https://arxiv.org/pdf/2507.06824) · [Decision-aware design](https://arxiv.org/pdf/2606.23662) · [Proper scores](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf)

**지식 지도.** Contact law·geometry → 관측 가능한 velocity → 가능한 sign·표면 강도의 posterior → 다른 물체의 stick probability → held-out Brier 순으로 연결된다. 물리 모델의 비식별성과 Brier의 최적 예측기는 근거가 확립된 부분이다. 선택 가능한 보정의 폭과 target shift가 손실을 얼마나 바꾸는지는 경험적인 질문이다. 실물에서 한 번의 추가 접촉이 가치 있는지에는 reset 비용과 그 확률로 바꿀 행동이 더 필요하다. [Impact model](https://arxiv.org/pdf/1710.04979) · [실험 선택과 결과 사용](https://arxiv.org/html/2609.11022v1)

### 문헌과 실험에서 이어지는 10개 연구 질문

| 질문 | 왜 필요한가 | 검증할 변수와 방법 |
|---|---|---|
| 1. 새 catalog에서도 순위가 재현되는가? | Finite catalog 내 정확성과 새 population으로의 일반화는 별개다. | Catalog seed를 최상위 독립 단위로 재추출하고 within/between-catalog variance를 분리한다. |
| 2. Continuous population에서도 같은 ideal gain인가? | 알려진 유한 support가 물체 식별을 쉽게 할 수 있다. | Support 크기를 늘려 ESS, posterior predictive gain의 수렴을 측정한다. |
| 3. Regime 비율을 바꾸면 fixed가 어느 정도 유리한가? | Source에서 정한 shrinkage는 구성 비율에 의존한다. | Train prior를 고정하고 연속적인 deployment prevalence에서 Brier를 비교한다. |
| 4. 두 힘의 측정을 더하면 한 속도에서도 충분한가? | 실제 friction 추정은 force와 velocity를 함께 활용한다. | 같은 접촉 수로 normal/tangential force를 별도 sensor channel로 추가한다. |
| 5. 속도에 따라 noise가 바뀌어도 posterior가 유용한가? | 실물의 빠른 slip–stick에서 측정 난도가 달라진다. | Speed-specific noise·clipping 아래에서 calibrated likelihood와 고정 likelihood를 비교한다. |
| 6. 한 표면에서 다른 위치로 갈 때 방향이 반전하면 어떻게 되는가? | 현재 near-field strength 세 값에는 sign reversal이 없다. | 연속 위치·반전 길이를 바꾸고 지역별 gain을 예측한다. |
| 7. 두 개의 deployment sensor 중 어느 쪽이 네 번째 접촉과 중복되는가? | 기존 정보 집합에 따라 추가 관측 가치는 달라진다. | 같은 world의 두 관측을 실험자가 무작위 배정해 interaction을 측정한다. |
| 8. 새로운 source에서 학습한 continuous correction이 더 나은가? | Posterior mean과 expected Brier 최적화는 서로 다른 기준이다. | 환경 단위 cross-fitting 후 동결한 연속 weight와 finite action을 대조한다. |
| 9. 추가 정보를 cohort 내에서 공유하면 얻는 양은 얼마인가? | v38은 각 deployment 물체의 실제 측정만 해당 물체에 사용한다. | Context를 공유하는 순서와 비용을 사전 지정하고 per-object와 sequential pooling을 비교한다. |
| 10. 실제 행동을 바꾸는 데 필요한 최소 Brier gain은 얼마인가? | Proper score만으로 장비 운용 효용을 정할 수 없다. | Slip 실패 비용·접촉 시간을 먼저 정해 predicted probability, action switch, expected cost를 함께 본다. |

1–3과 8은 model combination 및 expected future loss의 적용 범위에서, 4–7은 실제 센서와 contact model의 제한에서, 9–10은 정보를 얻고 사용하는 단위의 차이에서 유도한 후속 가설이다. 표의 항목들은 완료된 검증과 구분한다. [Model combination](https://arxiv.org/abs/2101.08954) · [Decision-aware design](https://arxiv.org/pdf/2606.23662) · [실물 contact 동작](https://arxiv.org/pdf/2507.06824) · [Physical experiment selection](https://arxiv.org/html/2609.11022v1)
