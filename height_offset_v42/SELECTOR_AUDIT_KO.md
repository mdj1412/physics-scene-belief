# v42 보정 선택의 독립 검산

실제 contact offset마다 세 정책 중 가장 작은 Brier를 얻는 선택이 어떻게 바뀌는지 estimated decoder에서 다시 계산했다. 모든 값은 12개 독립 catalog에서 같은 물체·같은 noise를 대응시킨 결과다. 표의 음수는 0% 보정보다 낮은 Brier를 뜻한다.

| 실제 offset (mm) | Fixed−zero | Optimizer−zero | Selector−zero | 같은 빈도의 무작위 정책−zero | Nominal 최선과 실제 offset 최선이 다른 비율 |
|---:|---:|---:|---:|---:|---:|
| -1.000 | +0.000855 | +0.002046 | +0.000349 | +0.000390 | 36.0% |
| -0.500 | +0.000834 | +0.001674 | +0.000065 | +0.000288 | 34.2% |
| -0.250 | +0.000896 | +0.001505 | -0.000120 | +0.000229 | 29.3% |
| -0.125 | +0.000968 | +0.001015 | -0.000432 | +0.000121 | 21.1% |
| +0.000 | +0.001002 | +0.000670 | -0.000637 | +0.000046 | 0.0% |
| +0.125 | +0.000859 | +0.001383 | -0.000043 | +0.000252 | 30.3% |
| +0.250 | +0.000716 | +0.001367 | -0.000268 | +0.000171 | 34.7% |
| +0.500 | +0.000716 | +0.001697 | +0.000218 | +0.000318 | 36.7% |
| +1.000 | +0.000710 | +0.001843 | +0.000192 | +0.000322 | 37.6% |

실제 selector의 0% / fixed / optimizer 비율은 68.8% / 16.5% / 14.6%였다. 같은 구매 관측이 들어와 모든 offset에서 선택은 같았다. Nonzero offset 균등 평균에서 fixed 선택의 전체 Brier 변화 기여는 +0.000027, optimizer 선택의 기여는 -0.000032였다. 각 값은 해당 action을 택한 경우의 loss 변화에 그 사용 빈도를 반영한 것이다.

| Nonzero 8개 offset의 paired 대조 | 평균 차이 [개별 95% CI] |
|---|---:|
| Selector−zero | -0.000005 [-0.000186, +0.000176] |
| Selector−fixed | -0.000824 [-0.000962, -0.000686] |
| Selector−optimizer | -0.001571 [-0.001879, -0.001263] |
| Selector−동일 빈도 무작위 정책 | -0.000266 [-0.000357, -0.000175] |
| Offset별 사후 최선−모든 offset에서 공유하는 사후 최선 | -0.000664 [-0.000740, -0.000588] |
| Offset별 cross-fitted−모든 offset에서 공유하는 cross-fitted | -0.000559 [-0.000632, -0.000486] |
| Selector−모든 offset에서 공유하는 cross-fitted | +0.003848 [+0.003466, +0.004230] |
| Offset별 cross-fitted−zero | -0.004412 [-0.004870, -0.003954] |
| 모든 offset에서 공유하는 cross-fitted−zero | -0.003853 [-0.004256, -0.003449] |
| 모든 offset에서 공유하는 사후 최선−nominal 사후 최선 재사용 | -0.003837 [-0.004213, -0.003461] |

사후 최선은 해당 environment·noise마다 전체 128개 deployment label을 본 진단이다. Cross-fitting은 한쪽 half의 label로 정책을 선택하고 독립 prototype으로 구성된 반대쪽 half에서 평가한 뒤 역할을 바꾼 결과다. 실제 offset을 아는 경우와 모르는 경우에 어느 만큼 다른 선택이 가능한지 평가하며, 실제 selector는 어느 쪽의 deployment label도 쓰지 않는다. 무작위 정책은 각 cell·panel의 선택 빈도를 고정해 환경별 배정을 무작위화한 기대 Brier다.

원 확률에서 재계산한 archive risk 차이 최대 절대 오차는 2.08e-16였다. [원자료](selector_audit42.json) · [별도 분석 계획](SELECTOR_AUDIT_PLAN_KO.md) · [코드](selector_audit42.py)

## Donor 물체당 하나의 event만 쓸 때

위 pooled cross-fitting은 각 donor 물체에서 여덟 개 offset의 counterfactual event를 모두 사용한다. 실제 수집에 더 가까운 사후 보조 대조에서는 각 donor 물체의 offset을 여덟 값에서 균등하게 한 번만 추출하고, 그때의 event label 한 개만 건넸다. 해당 offset 값은 action 선택에 사용하지 않았다. 환경당 약 64개의 donor event로 하나의 공통 action을 고르고 나머지 prototype에서 여덟 offset의 Brier를 평균했다. 네 분할과 난수 추출은 평균 평가를 위한 반복이며, 한 번의 policy 선택에 사용한 label 예산은 donor당 1회다. 같은 조건에서 zero 대비 Brier 차이는 **-0.003784 [-0.004181, -0.003387]**였다. 이는 zero와 동일한 label-free·4-contact 예산을 쓴 운영 방법의 이득으로 해석할 수 없다. Target label을 일부 획득하면 environment 간 보정 순위를 학습할 수 있다는 진단이다. [계산 결과](single_label42.json) · [재현 코드](single_label42.py)
