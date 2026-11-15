# v29 · 추가 calibration probe를 어느 환경에서 계속할 것인가

**평가 때 사용할 probe 수까지 동일하게 고정하면, 추가 calibration probe를 환경별로 재배분해 held-out Brier loss를 조금 낮출 수 있었다.** 24개 독립 block에서 환경 8곳당 calibration probe 32회와 test probe 104회를 두 정책 모두에 배정했다. Adaptive stopping의 common-decoder Brier는 0.10717, 환경마다 4회씩 배정한 fixed 4는 0.10806이었다. 차이는 **−0.00089**, 두 사전 지정 비교에 대한 simultaneous 95% interval은 **[−0.00172, −0.00006]**이었다. 사전에 정한 실질적 개선 폭 0.003에는 미치지 않았다.

관측을 자유롭게 선택하는 streaming에서는 **0.10590 대 0.10629**였고, 차이의 구간은 **[−0.00124, +0.00045]**였다. 24개 block 전체에서 두 정책은 각각 768회의 calibration을 사용했고, adaptive가 실제 고른 test probes는 2,341회로 fixed 4의 2,397회보다 56회 적었다. 같은 단가를 probe당 c라고 하면 접촉 지출 차이는 streaming에서는 −56c, batch quota에서는 0이다. Query view 등 다른 센서 비용은 포함하지 않았다.

첫 2회 이후 점수는 어느 환경에서 **관측 선택이 더 자주 바뀔지**를 잘 구분했다. 환경 두 곳을 비교한 순서 일치율은 78.9%였다. 반면 어느 환경에서 **관측을 바꾸는 것이 Brier를 더 낮출지**에 대한 일치율은 50.3%였다. 이 차이가 이번 연구에서 가장 분명하게 드러난 한계다.

Warm GPU p90은 streaming에서 adaptive 3.497 ms와 fixed 4 3.494 ms, batch 128개 전체에서 424.33 ms와 424.37 ms였다. 해당 환경의 posterior fit·score·bank 생성까지 포함하면 batch p90은 506.30 ms와 481.15 ms로 adaptive가 5.2% 더 걸렸다. 센서와 물체 reset에 걸리는 실제 시간은 측정하지 않았다.


## 1. 같은 예산을 어떻게 보장했나

v28에서 16회의 calibration probe는 friction shift의 방향을 맞히고 최종 stick probability를 개선했다. 그러나 관측을 고르는 정책만 바꾸고 decoder를 고정하면 neutral 대비 Brier 이득을 입증하지 못했다. V29에서는 “다음 probe가 미래 world에서 query view와 probe의 순위를 바꿀 가능성이 큰 환경에 한정된 접촉을 몰아주면 fixed 4보다 좋은 선택을 하는가?”를 검증했다. 정답은 별도 충돌의 stick 여부이고, 평가는 선택한 관측을 한 개 받은 뒤의 Brier loss다. [v28 보고서](../calibration_probe_v28/REPORT_KO.md)

한 world는 곡률·마찰·반발계수가 숨겨진 하나의 충돌 장면이다. Query view는 다음 충돌 위치 부근의 정밀 boundary profile을 관측하고, probe는 별도의 입사속도로 미끄러짐을 일으킨 뒤 2차원 outgoing velocity를 측정한다. 두 관측 중 하나를 받아 다른 충돌이 stick할 확률을 예측한다. Brier loss는 이 확률과 실제 binary label의 차이를 제곱한 값이다. Calibration probe는 같은 물성 분포를 공유하는 과거 world에서 구입하고, test probe는 지금 예측하려는 world에서 구입한다. 환경의 αc는 profile offset과 curvature spread, αf는 velocity proxy와 friction spread 사이의 결합 강도와 방향을 나타낸다. [물리 모델](../within_calibration_v25/physics25.py) · [shift 생성](../proxy_shift_v26/physics26.py)

각 독립 block에서 8개 환경을 동시에 다뤘다. 환경마다 서로 독립인 1,024개 calibration worlds에서 factual boundary profile과 outgoing velocity를 받았다. 서로 다른 물체에 가하는 추가 calibration probe를 무작위 순서로 한 개씩 공개했다. 모든 학습 가능 정책은 한 block에 정확히 **32회**를 쓴다. Fixed 4는 8곳에 4회씩 배정한다. Adaptive는 8곳에 2회씩 사용하고 점수가 높은 4곳을 남겨 각각 2회 추가한다. 네 곳을 다시 평가해 그중 두 곳에 4회씩 더 쓴다. 최종 배분은 2회인 곳 4개, 4회인 곳 2개, 8회인 곳 2개다. 중단한 환경도 똑같은 가중치로 최종 평가에 남는다. 한 환경을 중단하고 예산을 절약하는 단일 환경 정책과는 운영 조건이 다르다.

같은 2/4/8 배분을 무작위로 섞은 정책, friction coupling αf의 posterior 표준편차가 큰 곳을 남기는 정책을 추가했다. Known-prior K256은 환경의 실제 αc와 αf만 아는 privileged comparator이며 calibration을 하지 않는다. 모든 방법이 개별 평가 world의 물성·정답과 선택하지 않은 관측을 볼 수 없도록 했다. 먼저 예산과 단계별 의사결정을 SHA-256으로 고정하고, 그 뒤에만 가능한 모든 prefix(2·4·8회)의 가상 정책을 평가용으로 구성했다. Test action을 다시 hash로 동결한 다음 label과 potential observation을 열었다. [사전 protocol](PROTOCOL.md) · [실험 설정·seeds](design.json) · [GPU runner](runner29.py)

V28의 8개 shift law를 재사용하고 **24개 새 독립 blocks, 192개 환경 cohort, 24,576개 held-out worlds**를 생성했다. 환경마다 128개 새 test worlds와 각 action의 observation noise 32회 반복을 사용했다. V29는 알려진 물리 family 안의 같은 shift panel에 대한 새 표본 검증이다. 동일한 contact law와 sensor calibration을 사용했다. 정책에는 숨은 환경 이름도 주지 않았다. 경쟁하는 환경은 매 block에서 독립적으로 순서를 섞었다.

| 운영 조건 | Calibration probes / block | 평가 관측과 test probes | 해석 |
|---|---|---|---|
| A · Streaming | 32 | 1,024 worlds 각각에서 query view 또는 probe를 자유 선택 | Calibration 예산 일치. 실제 test probe 수 별도 보고 |
| B · Batch quota | 32 | 128개 factual inputs가 모이면 환경당 정확히 13회 probe, 나머지 115회 query | 총 136 probes/block. 두 정책 모두 동일한 batch에 접근 |

조건 B에서는 각 환경 안에서 모든 test world의 risk(query)−risk(probe)를 정렬해 가장 높은 13개에 probe를 배정했다. 양의 gap이 13개보다 적어도 13회에 맞춘다. 한 방법에만 미래 factual inputs를 허용하는 비교를 피했다. 이 조건은 128개 모두의 factual evidence가 도착한 후 처리해야 하므로, 단일 world가 즉시 도착하는 streaming latency와 구별해야 한다.

## 2. 다음 probe의 가치를 어떻게 계산했나

각 환경의 posterior는 v28의 paired likelihood를 사용한다. Factual만 관측한 world에는 p(x|αc), 이미 calibration probe를 산 world에는 p(x,y|αc,αf)를 한 번 적용한다. 추가 probe는 query 위치의 geometry와 contact law에 대한 현재 믿음을 바꾸고, 그로 인해 미래 world에서 선호하는 observation이 달라질 수 있다.

평가 데이터와 별도로 만든 **96개 reference factual worlds**에 대해 5×9개 coupling grid에서 K256으로 두 action의 Brier risk 차이 g=R(query)−R(probe)를 계산했다. 환경을 계속할 때 2회 또는 4회 뒤 가능한 미래를 예측하기 위해 **256개의 독립 simulator probe sequences**를 고정했다. 각 sequence의 측정값에 posterior predictive weight를 주고, 추가 관측 뒤의 coupling 평균에서 grid risk를 bilinear interpolation했다. 각 미래의 점수는 현재와 미래의 g 부호가 바뀐 reference world에서 |g_after|를 더한 값이다. 환경별 factual 분포와 미래 sequence 확률로 가중 평균해 큰 곳에 예산을 줬다. 따라서 단순히 많이 뒤집힐 곳보다는, 뒤집힐 때 틀린 선택의 예상 손실이 큰 곳이 높은 점수를 받는다. 두 stage에서는 각각 앞으로 살 2회와 4회 전체를 예측한다.

계산에는 근사가 있다. 실제 probe 결과를 받은 후의 estimator는 같은 calibration world에서 나온 x,y의 joint likelihood와 두 coupling의 posterior covariance를 보존한다. 미리 점수를 낼 때는 미래의 개별 factual x를 적분한 probe velocity likelihood를 쓰고, risk lookup은 두 축의 평균으로 근사한다. 점수는 K256 내부의 예상 손실로 정의됐고 held-out Brier 감소 자체를 보장하지 않는다. 추가로 산 probe가 두 방법의 실제 우열을 바꾸는지는 §5에서 평가했다. [점수 구현](policy29.py) · [posterior와 미래 가중치 직접 적분 검산](check_policy29.py)

## 3. Held-out Brier와 실제 사용량

| 배분 정책 | Streaming common Brier | Batch quota common Brier | Streaming native Brier | Batch quota native Brier | Streaming test probe 선택률 | Streaming 총 probes / world¹ |
|---|---|---|---|---|---|---|
| Adaptive stopping | 0.10590 | 0.10717 | 0.10749 | 0.10874 | 9.5% | 0.1265 |
| Fixed 4 | 0.10629 | 0.10806 | 0.10796 | 0.10983 | 9.8% | 0.1288 |
| Random 2/4/8 | 0.10557 | 0.10748 | 0.10737 | 0.10929 | 9.7% | 0.1281 |
| Uncertainty stopping | 0.10556 | 0.10722 | 0.10729 | 0.10879 | 9.7% | 0.1281 |
| Known-prior K256 | 0.10284 | 0.10506 | 0.10284 | 0.10506 | 12.2% | 0.1223 |

¹ 총 probe 수는 calibration 횟수를 배포 world 수로 나누고 test probe rate를 더한 값이다. Query view 취득, factual 관측, 물체 reset은 probe 횟수로 세지 않았다. Native는 각 정책이 자기 posterior로 decoding한 결과다. 두 co-primary 비교는 동일한 known-prior decoder에서 같은 world·action에 같은 확률 예측과 Brier를 할당한다. 따라서 common 열에서 나타나는 차이는 observation 선택에서만 나올 수 있다. Known-prior K256도 finite bank를 사용하므로 정확한 Bayes lower bound는 아니다.

| 사전 지정 대비 · Adaptive − Fixed 4 | Δ common Brier | Simultaneous 95% interval | 유리한 block 수 | 한 block씩 제외한 평균 범위 | Fixed 4와 다른 action |
|---|---|---|---|---|---|
| Streaming | -0.00039 | [-0.00124, +0.00045] | 11/24 | [-0.00048, -0.00011] | 3.5% |
| Batch quota | -0.00089 | [-0.00172, -0.00006] | 14/24 | [-0.00101, -0.00070] | 7.5% |

음의 차이는 adaptive에 유리하다. 두 대비에 대해 24개 독립 block 전체를 짝지어 10,000회 재추출하고, 표준오차로 나눈 최대 절댓값으로 simultaneous 95% interval을 만들었다. 경쟁하는 환경들은 예산을 공유하고, batch quota에서는 한 world의 순위가 같은 환경의 다른 world 선택에도 영향을 준다. 따라서 환경과 world를 제각각 독립 표본으로 세지 않았다. Streaming에서 block과 그 안의 world를 함께 재추출한 별도 sensitivity interval은 [-0.00143, +0.00056]다. 사전 기준은 upper bound가 0 미만이면 우월, −0.003 미만이면 연구용 실질 개선이다. Interval이 0을 포함하면 동등성을 주장할 수 없다. [검산된 전체 수치](results/summary.json) · [통계 코드](analysis29.py)

## 4. 어디에 배정했으며 무엇을 얻었나

| 실제 (αc,αf) · 평가용 | 평균 배정 횟수 | 2 / 4 / 8회 배정 block 수 | Streaming adaptive / fixed | Batch quota adaptive / fixed |
|---|---|---|---|---|
| (-0.65, -0.85) | 4.33 | 10 / 7 / 7 | 0.10035 / 0.10193 | 0.10075 / 0.10001 |
| (-0.65, 0.85) | 5.00 | 8 / 6 / 10 | 0.09931 / 0.09815 | 0.10133 / 0.09869 |
| (0.35, -0.65) | 3.75 | 13 / 6 / 5 | 0.11127 / 0.11204 | 0.11226 / 0.11358 |
| (0.35, 0.65) | 3.17 | 16 / 5 / 3 | 0.11497 / 0.11595 | 0.11479 / 0.11987 |
| (-0.2, -0.35) | 2.75 | 19 / 3 / 2 | 0.11174 / 0.11179 | 0.11527 / 0.11800 |
| (-0.2, 0.35) | 3.58 | 15 / 4 / 5 | 0.11249 / 0.11312 | 0.11619 / 0.11491 |
| (0.72, -0.92) | 5.83 | 4 / 7 / 13 | 0.09623 / 0.09638 | 0.09620 / 0.09849 |
| (0.72, 0.92) | 3.58 | 11 / 10 / 3 | 0.10087 / 0.10099 | 0.10054 / 0.10088 |

Fixed 4와의 차이는 네 곳을 일찍 중단하며 생긴 변화와 두 곳에 더 많은 probe를 주며 생긴 변화로 정확히 나눌 수 있다. 두 곳은 예산이 4회로 같아 기여가 0이다. 아래 값은 8개 환경 전체에 대한 가중 기여다. 음수이면 손실이 줄었다.

| 조건 | 조기 중단 (2−4) 기여 [95%] | 계속 관측 (8−4) 기여 [95%] | 균등 무작위 배분 기대 Brier | Adaptive − 무작위 기대 [95%] | Hindsight 배분 Brier |
|---|---|---|---|---|---|
| Streaming | +0.00007 [-0.00028, +0.00044] | -0.00046 [-0.00115, +0.00017] | 0.10588 | +0.00002 [-0.00054, +0.00056] | 0.10302 |
| Batch quota | -0.00054 [-0.00121, +0.00015] | -0.00035 [-0.00075, +0.00005] | 0.10773 | -0.00056 [-0.00103, -0.00008] | 0.10382 |

무작위 기대값은 각 환경에서 2·4·8회로 끝날 확률 1/2·1/4·1/4를 실제 측정한 세 potential losses에 곱했다. 이와 별도로 §3의 random 정책은 예산 순열을 한 번 실제 추출한 결과다. Hindsight는 가능한 420개 배분 중 같은 held-out outcomes에서 가장 낮은 loss를 골랐다. 재배분에 여지가 얼마나 있었는지 보는 낙관적 진단이며 배포 가능한 성능은 아니다. 각 표의 사후 interval은 별도 표기가 없으면 pointwise block bootstrap이다.

## 5. 점수가 높은 환경에서 실제로 더 좋은 선택을 했나

| Stage | 예상 flip 비율 | 실제 flip 비율 | Flip 순서 일치율 [95%] | Brier gain 순서 일치율 [95%] | 계속−중단 실제 common Brier 이득 [95%] |
|---|---|---|---|---|---|
| 2→4, 전체 8곳 | 3.1% | 5.8% | +0.789 [+0.765, +0.812] | +0.503 [+0.440, +0.568] | -0.00079 [-0.00287, +0.00126] |
| 4→8, 남은 4곳 | 3.9% | 7.7% | +0.559 [+0.455, +0.660] | +0.549 [+0.465, +0.639] | +0.00040 [-0.00277, +0.00335] |

같은 prefix에서 후보 환경 모두에 동일한 추가 예산을 가상 적용해 각 held-out world에서 실제 action이 바뀐 비율과 common-decoder Brier 개선을 측정했다. 순서 일치율은 같은 block 안에서 환경 둘을 비교해 순서가 맞으면 1, 동률이면 0.5다. 무작위 기준은 0.5다. 둘째 stage의 결과는 첫 stage를 통과한 환경에 조건부다. 점수가 flip을 예측해도 flip에 따른 이득의 크기와 부호까지 예측해야 재배분으로 평균 Brier를 낮출 수 있다. [Score audit](results/score_audit.json) · [세부 rank correlation과 구간](results/diagnostics.json)

### Batch quota의 경계에 맞춘 보조 실험

주 정책은 probe가 query를 앞서는 지점을 예측한다. Batch quota로 13개를 선발할 때는 13등과 14등의 순위가 바뀌어야 예산 배정에 영향을 준다. 이를 직접 평가하는 보조 정책을 같은 24개 block에서 추가로 실행했다. 각 가상 미래의 factual 분포에 대해 예산 13/128에 해당하는 상위 reference worlds를 다시 찾고, 교체할 world 쌍의 예측 Brier 차이를 가중해 환경의 우선순위를 정했다. 경계에 걸린 마지막 reference world에는 fractional weight를 주어 모든 가상 시나리오에서 정확히 같은 예산을 사용했다. 정책 변경은 탐색적이다. 별도 tuning 없이 24개 block의 보조 배분을 모두 고정하고 test actions까지 동결한 후 평가했다. [보조 정책 소스](quota29.py) · [결과와 hash](results/quota29/summary.json)

| 조건 | Quota-aware adaptive | Fixed 4 | 주 adaptive | 보조 − Fixed 4 [simultaneous 95%] | 보조 − 주 adaptive [simultaneous 95%] |
|---|---|---|---|---|---|
| Batch quota | 0.10740 | 0.10806 | 0.10717 | -0.00065 [-0.00175, +0.00045] | +0.00024 [-0.00057, +0.00104] |
| Streaming | 0.10537 | 0.10629 | 0.10590 | -0.00092 [-0.00204, +0.00019] | -0.00053 [-0.00153, +0.00046] |

계속할 환경의 기준을 바꾸자 원래 정책과 최종 calibration 횟수가 달라진 환경은 62.5%, batch quota에서 다른 관측을 고른 worlds는 5.8%였다. 환경당 보조 scoring 시간은 평균 3.88 ms였다. 이 보조 표의 4개 contrast에만 별도로 simultaneous interval을 계산했다. 동일한 held-out panel을 재사용했으므로 독립적인 confirmatory replication은 남아 있다.

## 6. GPU p90, setup, 접촉 비용

| Common-decoder 실측² | Adaptive p90 | Fixed 4 p90 | p90 ratio [pointwise 95%] | 정책당 시료 수 |
|---|---|---|---|---|
| Streaming, 1 world | 3.497 ms | 3.494 ms | 1.001 [0.998, 1.004] | 480 |
| Batch quota, 128 worlds 전체 | 424.329 ms | 424.375 ms | 1.000 [0.998, 1.002] | 72 |

² 같은 known decoder를 실제로 실행했다. 미리 정한 첫 12개 block에서 seed로 정한 환경 두 곳씩, 총 24개 환경을 사용했다. Streaming은 10개 worlds × 2회, batch는 전체 128개 factual inputs의 risk 계산·상위 13개 정렬·선택한 observation의 decoding을 3회 실행했다. Warm-up 후 방법 순서를 무작위화하고 CUDA synchronization과 host 전송을 포함했다. 모든 action과 첫 noise draw의 예측 확률을 저장 결과와 검산했다. Batch p90은 128개 전체의 wall time이다. 이를 128로 나눈 throughput은 한 world의 응답 지연과 다르다. Interval은 3,000회 paired bootstrap에서 산출했다. [GPU timing 원수치](results/quota_latency29.json) · [측정 코드](quota_latency29.py)

| 환경당 calibration 단계 (ms) | Adaptive | Fixed 4 | 무작위 | Uncertainty |
|---|---|---|---|---|
| Posterior fit | 24.55 | 15.99 | 16.07 | 24.51 |
| 중단 점수 계산·정렬 | 1.82 | 0.00 | 0.00 | 0.01 |
| 최종 K256 bank 생성 | 43.60 | 41.94 | 42.76 | 43.28 |

96개 reference worlds의 risk table 및 미래 probe library를 처음 구성하는 데 13.85초가 걸렸다. 이는 환경마다 되풀이할 필요가 없는 1회 비용이다. Posterior estimator의 공통 초기화는 214.5 ms였다. 위 setup은 실제 배포 경로가 사용하는 stage 계산과 bank 생성에 실측 시간을 배분한 값이다. Prefix가 겹치는 실험 정책들은 같은 계산을 재사용하고 경로별 비용을 별도 산정했다. 실제 장비의 충돌·reset·관측 대기 시간 및 접촉의 금전적 단가는 측정하지 않았다. 두 정책이 같은 수의 calibration probes를 구매해도 adaptive는 응답을 받고 다음 2→4→8 단계를 선택해야 한다. 물리 probes의 단계별 대기 시간과 병렬성 때문에 실물 wall time이 같다고 단정할 수 없다.

| 환경별 setup을 포함한 비용 | Adaptive p90 | Fixed 4 p90 | Ratio [95%] |
|---|---|---|---|
| Streaming · 128개 결정에 균등 배분 | 4.085 ms | 3.938 ms | 1.037 [1.012, 1.050] |
| Batch quota · 환경당 128개 전체 | 506.298 ms | 481.152 ms | 1.052 [1.021, 1.058] |

각 timing 대상 환경의 posterior fit, stopping score, 최종 selector bank 생성 시간을 합산했다. Streaming은 환경당 128개 결정으로 나눈 상각값, batch는 setup 전액을 더한 값이다. 실제 매번 발생하는 단일 요청 대기시간과 상각 cost는 구별해야 한다. 두 방법에 공통인 decoder 초기화, 환경 전체에 공유하는 static table 생성, 실제 센서 대기시간은 이 표에서 제외했다. [상각 계산과 구간](results/cost_with_setup.json)

![Common-decoder Brier and calibration allocation](results/v29.png)

## 7. 논문 기여 가능성과 다음 검증

**논문에 실을 수 있는 주장은 적용 조건이 구체적이다.** 동일한 simulator의 여덟 shift law, 128개 factual world를 동시에 볼 수 있는 batch, 환경마다 test probe 13회라는 조건에서는 calibration 접촉 재배분이 fixed 4 대비 평균 Brier를 낮췄다. 같은 2/4/8회를 무작위로 배정했을 때의 정확한 기대값과 비교해도 차이는 −0.00056 [−0.00103, −0.00008]였다(pointwise 95%). 이 비교는 예산에 차등을 둔 효과와 어떤 환경을 선택했는지의 효과를 구별한다. Batch quota에서 uncertainty-only 대비 차이는 −0.00005 [−0.00068, +0.00054]였으므로 복잡한 rank-flip score를 단순 posterior uncertainty보다 우선할 근거는 약하다. Streaming에서도 정확한 무작위 기대값과의 차이는 +0.00002 [−0.00054, +0.00056]였다.

**왜 두 운영 조건이 달랐는지는 부분적으로 설명할 수 있다.** Fixed 4와 다른 관측을 선택한 world는 streaming의 3.50%, batch quota의 7.50%였다. Quota에서는 한 world가 13위 경계를 넘으면 다른 world도 자리를 내줘야 한다. 따라서 risk gap의 부호가 유지돼도 여러 action이 달라질 수 있다. 그러나 두 효과의 크기가 서로 다르다는 별도 confirmatory interaction test를 하지는 않았다. 첫 stage의 점수는 실제 flip 비율을 절반가량으로 낮게 예측했고(3.09% 대 5.77%), 계속한 환경의 flip은 8.73%, 중단한 환경은 2.82%였다. 관측이 바뀌어 Brier가 개선된다는 인과관계는 이 빈도만으로 뒷받침되지 않는다. 두 번째 stage에 남은 네 곳에서는 flip 순서 일치율도 55.9% [45.5%, 66.0%]로 불확실했다.

보조 실험에서 상위 13/128 경계를 직접 겨냥해도 batch Brier는 0.10740이었다. 원래 adaptive와의 차이는 +0.00024 [−0.00057, +0.00104]였다. 기준을 배포 quota에 맞추는 것만으로 추가 개선이 성립한다는 증거는 얻지 못했다.

**방법 논문으로는 추가 근거가 필요하다.** 현재 독립 단위는 24개 block이고 같은 여덟 물리 분포를 반복했다. 각 world의 32회 noise 반복은 동일한 stick label을 공유한다. Known-prior common decoder는 선택 효과를 분리하기 좋지만 실제 배포 시에는 그 prior를 모른다. 단일 simulator, 동일 contact law·sensor calibration, 공통 reference table에서 얻은 결과이므로 새로운 contact law, sensor, 반복 가능한 실물 접촉에서도 개선되는지는 열려 있다. 평가 기여로서는 같은 잠재 결과를 써서 **calibration budget → 관측 변경 → 실제 decision loss**를 나눠 측정하고, 그중 어느 단계에서 지표가 어긋나는지 보인 점에 가치가 있다.

**다음 검증의 우선순위는 실제 Brier 이득을 예측하는지 확인하는 것이다.** 별도 training environments에서 paid prefix 뒤의 signed common-decoder Brier gain을 학습하고, world·bank·contact law를 모두 분리한 test panel에서 frozen scorer를 평가한다. Fixed 4, uncertainty-only, 같은 budget histogram의 exact random expectation을 유지하고, held-out contact law별로 streaming/batch를 사전 지정한다. 이후 별도 simulator 또는 실제 반복 접촉에서 sensor·reset 시간을 재어 time-matched budget으로 재검증한다. 여전히 uncertainty-only 차이를 좁히지 못하면 논문의 중심을 새로운 acquisition 알고리즘보다는 순위 변화가 실제 decision value의 proxy로 실패하는 조건의 검증에 두는 편이 타당하다.


## 문헌 검토와 논문 기여의 범위

### 1. 원문 다섯 편의 비교

문헌 검토 질문은 “추가 측정이 미래 선택을 바꿀 가능성에 따라 한정된 측정 예산을 어디에 배정해야 하는가?”이다. 아래는 이론, 실험 표본, 최종 평가 단위를 각각 확인한 목적 표집이다. 모집단 전체에 대한 systematic review나 합산 가능한 effect size는 제공하지 않는다.

| 논문 | Research question과 methodology | Sample과 data source | 주요 결과와 결론 | v29와의 차이·한계 |
|---|---|---|---|---|
| Frazier, Powell & Dayanik (2008), *A Knowledge-Gradient Policy for Sequential Information Collection* | 어떤 대안을 한 번 더 측정할 때 마지막에 얻을 expected utility가 가장 많이 증가하는가? 독립 Gaussian prior, 알려진 공통 noise variance에서 one-step knowledge gradient(KG)를 해석적으로 풀었다. | 무작위로 생성한 ranking-and-selection 문제 100개. 문제·정책당 opportunity cost를 10만 회 simulation으로 추정했다. | 한 번의 측정이 남았을 때 optimal이고, 측정 수가 무한대로 갈 때 opportunity cost가 0으로 수렴한다. 시험한 문제 전체의 평균에서 경쟁 정책보다 좋은 성능을 보고했다. | 마지막에 대안 하나를 고른다. v29는 예산을 적게 준 환경까지 포함하여 미래 world 모두에서 Brier를 계산한다. 중간 길이 horizon이나 새로운 likelihood에 대해 동일한 보장을 얻을 수 있는지는 별도 문제다. [원문 §§3–5, 8](https://optimallearning.princeton.edu/Papers/FrazierPowellDayanik_KnowledgeGradientSICON.pdf) |
| Kapoor, Horvitz & Basu (2007), *Selective Supervision* | 어느 label을 구매하면 예상 misclassification cost와 annotation cost의 합이 줄어드는가? Gaussian-process classifier로 각 label의 두 가능한 결과를 계산하고, 최대 net VOI가 음수일 때의 stopping criterion을 제시했다. | 한 사용자가 8개월간 받은 voicemail 207건. 과제별로 class마다 1건으로 시작해 20건을 더 고르는 비교를 20회 반복했다. | Table 2에서 친밀한 발신자 판별 accuracy는 VOI 72.9%, random 67.8%였다. 긴급도 판별은 둘 다 53.2%였다. 같은 비용 기반 정책의 이득이 과제에 따라 달랐다. | 길이에 따른 annotation 단가와 오분류 가격을 가정했고, 미리 확보한 메시지 pool에서 평가했다. 저자들은 단일 label의 net VOI가 낮아도 여러 label의 가치는 남을 수 있다고 지적했다. [원문 §3, §5와 Table 2](https://www.ijcai.org/Proceedings/07/Papers/141.pdf) |
| Chen, Lin & Zhou (2013), *Optimistic Knowledge Gradient Policy for Optimal Budget Allocation in Crowdsourcing* | 총 label 수를 고정하고 여러 item 중 어디에 다음 label을 줄 것인가? 전체 정답 수를 terminal utility로 두는 Bayesian MDP를 세우고 optimistic KG를 비교했다. | Simulated crowd 및 recognizing textual entailment 데이터. 실제 데이터는 sentence pairs 800개, pair당 10개 worker 응답, 서로 다른 workers 164명이다. | 특정 integer Beta prior와 deterministic tie-breaking에서 ordinary KG가 결국 첫 item에 계속 할당하는 반례를 증명했다. 실제 데이터의 homogeneous-worker 비교에서는 optimistic KG와 infinite-horizon Gittins가 좋은 성능을 보였다. | 추가 응답으로 수정하는 것은 해당 item의 label이다. v29에서는 과거 calibration world의 측정값이 독립적으로 샘플링된 미래 world의 관측 선택에 전달되어야 한다. [원문 Proposition 3.1, §7.2](https://proceedings.mlr.press/v28/chen13f.pdf) |
| Karnin, Koren & Somekh (2013), *Almost Optimal Exploration in Multi-Armed Bandits* | 정해진 예산에서 best arm을 틀릴 확률을 낮출 수 있는가? Sequential Halving으로 단계별 생존 arms를 반복 측정하고 절반을 남긴다. Arm gap에 의존하는 error bound를 제시했다. | 여섯 Bernoulli reward 설정. 20·40·80개 arms에 대해 Fig. 1은 조건별 3,000회 독립 실행을 평균했다. | Fixed-budget best-arm identification에서 단계적으로 후보를 제거하는 정책의 이론적 오류 상한과 수치 비교를 제시했다. | 제거의 기준은 arm의 평균 보상이며 마지막 winner 하나를 찾는다. v29에서 중단한 환경에도 관측 선택을 계속 제공하므로 해당 보장을 평균 Brier에 적용할 수 없다. [원문 §4, Theorem 4.1, §5](https://proceedings.mlr.press/v28/karnin13.pdf) |
| Huang, Guo, Acerbi & Kaski (2024), *Amortized Bayesian Experimental Design for Decision-Making* | Design과 최종 decision을 같은 downstream utility로 최적화할 수 있는가? Transformer Neural Decision Process의 query head와 predictive model을 훈련하고, 별도 test problems에서 평가했다. | HPO-B의 네 search spaces에서 top-3 configurations를 골랐다. 초기 5점, 실험의 T=50이며, 논문에는 별도의 retrosynthesis 사례도 있다. | 네 HPO 과제의 초기 10 queries에서 높은 utility gain을 보고했다. Training과 amortization의 계산 비용, 모형 가정이 deployment에서 달라지는 문제를 한계로 제시했다. | 모델과 acquisition을 함께 평가했다. v29는 선택한 관측을 한 decoder에서 교차 채점하여 action만의 효과를 보려는 실험이다. [원문 §§5–7, Appendix G](https://proceedings.neurips.cc/paper_files/paper/2024/file/c59f05d7ab3638b138cc61f32e1a7cd1-Paper-Conference.pdf) |

### 2. 주제별 evidence

**측정의 가치는 어떤 결정을 내릴지에 따라 달라진다.** Gaussian ranking에서는 최종 선택의 expected payoff 증가, binary classification에서는 예상 오분류 비용과 label 가격, decision-aware design에서는 task utility를 기준으로 정보를 평가했다. 셋 모두 측정 이후의 행동을 명시하지만 평가량은 서로 다르다. 따라서 이 문헌은 v29에서 Brier를 기준으로 정책을 비교할 근거를 주며, 특정 Brier 개선량을 예측해 주지는 않는다. [Frazier et al., §4](https://optimallearning.princeton.edu/Papers/FrazierPowellDayanik_KnowledgeGradientSICON.pdf) · [Kapoor et al., §3](https://www.ijcai.org/Proceedings/07/Papers/141.pdf) · [Huang et al., §§3–5](https://proceedings.neurips.cc/paper_files/paper/2024/file/c59f05d7ab3638b138cc61f32e1a7cd1-Paper-Conference.pdf)

**총예산과 각 환경의 중단 기준은 다른 설계 변수다.** Chen 등의 목적은 모든 item의 최종 정확도이고, Karnin 등의 목적은 마지막 best arm의 정확도다. Kapoor 등의 stopping criterion은 추가 label 비용을 정당화할 수 있는지를 판단한다. v29처럼 총예산을 반드시 다 쓰면 한 환경에서 중단한 측정은 다른 환경으로 간다. 이때 실험이 검증하는 것은 동시 배치된 환경 사이의 재배분 효과다. 환경 하나가 단독으로 언제 예산 지출을 종료해야 하는지는 별도 검증이 필요하다. [Chen et al., §2.4](https://proceedings.mlr.press/v28/chen13f.pdf) · [Karnin et al., §4](https://proceedings.mlr.press/v28/karnin13.pdf) · [Kapoor et al., §3](https://www.ijcai.org/Proceedings/07/Papers/141.pdf)

### 3. Knowledge map

| 연구 계열 | 개념·방법 사이의 관계 | 근거 수준과 열린 질문 |
|---|---|---|
| Frazier → Kapoor → Huang | 다음 측정의 가능한 결과 → posterior update → 가능한 downstream decision과 utility → 현재 측정의 가치. 이 화살표는 개념적 관계이며 논문 간 성능 서열을 나타내지 않는다. | 결정 목적을 명시하는 방법은 확립되어 있다. V29의 단순화한 score가 실제 contact observation의 action ranking을 얼마나 정확히 대변하는지는 열려 있다. [Frazier](https://optimallearning.princeton.edu/Papers/FrazierPowellDayanik_KnowledgeGradientSICON.pdf) · [Kapoor](https://www.ijcai.org/Proceedings/07/Papers/141.pdf) · [Huang](https://proceedings.neurips.cc/paper_files/paper/2024/file/c59f05d7ab3638b138cc61f32e1a7cd1-Paper-Conference.pdf) |
| Chen ↔ Karnin | 다수 항목 사이의 자원 경쟁 → 다음 측정 대상 또는 생존 집합 → terminal objective. | Fixed budget을 정확히 지키는 여러 방법이 있다. 모든 환경의 downstream loss에 유리한 중단 깊이, 조기 탈락의 비용, 타이브레이크의 영향은 원래 objective에 달려 있다. [Chen](https://proceedings.mlr.press/v28/chen13f.pdf) · [Karnin](https://proceedings.mlr.press/v28/karnin13.pdf) |

### 4–7. 연구 공백, 상충하는 결과, 방법론, 근거 강도

- **강한 근거가 있는 부분.** KG의 성질은 terminal utility와 관측모형에 의존한다. Frazier 등은 연속 Gaussian 관측과 하나의 최종 winner에 대해 극한 보장을 제시했지만, Chen 등의 discrete labels와 합산 정확도에서는 one-step gain이 동률이 되어 deterministic KG가 한 item에 머물 수 있다. 정리의 가정, tie-breaking, 손실이 달라 결론은 양립한다. V29에서 확인할 공백은 초기 2회 측정에서 점수 오차로 조기 중단한 환경의 손실이다. [Frazier, §§4–5](https://optimallearning.princeton.edu/Papers/FrazierPowellDayanik_KnowledgeGradientSICON.pdf) · [Chen, Proposition 3.1](https://proceedings.mlr.press/v28/chen13f.pdf)
- **중간 수준의 경험적 근거.** Utility-aware sampling은 일부 과제에서 유리했지만 Kapoor의 동일 데이터 안에서도 두 과제의 개선 폭이 달랐다. Huang의 held-out HPO 평가에는 representation과 query policy 학습 효과가 함께 들어간다. 반면 v29의 common decoder는 선택 차이에 초점을 맞춘다. 서로 다른 sample, estimator, metric의 수치를 한 효과 크기로 묶을 수 없다. [Kapoor, Table 2](https://www.ijcai.org/Proceedings/07/Papers/141.pdf) · [Huang, §§5–7](https://proceedings.neurips.cc/paper_files/paper/2024/file/c59f05d7ab3638b138cc61f32e1a7cd1-Paper-Conference.pdf)
- **이 문헌만으로 부족한 근거.** Best-arm error, message classification cost, top-k utility, terminal opportunity cost 중 어느 것도 미래 collision에서 두 observation의 Brier 차이를 직접 측정하지 않는다. 특히 action flip probability에는 틀린 action의 대가가 들어 있지 않다. 동일한 flip 확률이라도 action gap이 작으면 기대 이득은 작다. V29에 적용할 때에는 fixed decoder로 평가한 두 action의 실제 loss gap과 allocation score의 관계가 검증 대상이다. [Karnin, §4](https://proceedings.mlr.press/v28/karnin13.pdf) · [Frazier, §§3–4](https://optimallearning.princeton.edu/Papers/FrazierPowellDayanik_KnowledgeGradientSICON.pdf) · [Huang, §3](https://proceedings.neurips.cc/paper_files/paper/2024/file/c59f05d7ab3638b138cc61f32e1a7cd1-Paper-Conference.pdf)
- **표본·비용의 한계.** Crowd와 voicemail 데이터에는 실제 사용자가 있지만 물체나 접촉면은 없다. Bandit 및 Gaussian ranking 실험은 설계된 reward law에서 실시했다. Huang 등도 well-specified model과 upfront training 비용을 한계로 다뤘다. 따라서 다른 contact law, unequal probe cost, 환경이 유지되는 시간의 차이는 v29로 확장할 때 남는 구체적 공백이다. [Chen, §7.2](https://proceedings.mlr.press/v28/chen13f.pdf) · [Kapoor, §5](https://www.ijcai.org/Proceedings/07/Papers/141.pdf) · [Karnin, §5](https://proceedings.mlr.press/v28/karnin13.pdf) · [Huang, §7.1](https://proceedings.neurips.cc/paper_files/paper/2024/file/c59f05d7ab3638b138cc61f32e1a7cd1-Paper-Conference.pdf)

### 8. 후속 연구 질문 10개

| 질문 | 해결할 공백과 중요성 | 변수·검증 설계 | 근거 |
|---|---|---|---|
| 1. Rank-flip probability에 action gap을 곱하면 probe가 더 가치 있는 환경으로 가는가? | 작은 순위 변화와 큰 손실 회피를 구별한다. | 동일 probe schedule, scorer와 gap weighting만 바꾼 paired common-decoder Brier 비교. | [Frazier, §4](https://optimallearning.princeton.edu/Papers/FrazierPowellDayanik_KnowledgeGradientSICON.pdf) |
| 2. 첫 2회의 noise 때문에 유용한 환경을 얼마나 일찍 버리는가? | 한 번 탈락한 환경이 복구되지 않는 비용이 있다. | 같은 prefix 이후 stop/continue의 potential loss, sensor noise, 탈락률을 독립 blocks에서 측정. | [Karnin, §4](https://proceedings.mlr.press/v28/karnin13.pdf) |
| 3. Score에 동률이 많을 때 tie-break가 budget을 왜곡하는가? | 결정론적 순서가 가짜 선호를 만들 수 있다. | Environment order를 가리고 tie frequency, ID별 추가 배정, randomized tie-break Brier 비교. | [Chen, Proposition 3.1](https://proceedings.mlr.press/v28/chen13f.pdf) |
| 4. Probe 하나의 VOI와 2개·4개 묶음의 VOI는 같은 환경을 우선하는가? | 첫 probe가 단독으로 action을 못 바꿔도 다음 probe와 결합하면 바뀔 수 있다. | One-step 및 multi-step rollout의 선택률·Brier·planner latency 비교. | [Kapoor, §3](https://www.ijcai.org/Proceedings/07/Papers/141.pdf) |
| 5. 환경이 유사해도 score 기반 targeting이 도움이 되는가? | Unequal allocation 자체와 어느 환경에 주었는지의 효과를 분리한다. | Shift heterogeneity를 독립 조작하고 같은 budget histogram을 무작위 배정한 sham과 차이 비교. | [Chen, §§2, 7](https://proceedings.mlr.press/v28/chen13f.pdf) |
| 6. Unit probe cost가 다른 환경에서 count-matched 결론이 유지되는가? | 관측 횟수와 실제 시간이 달라질 수 있다. | Surface별 sensing·reset 시간을 측정하고 count-matched와 time-matched allocation을 대조. | [Kapoor, §5](https://www.ijcai.org/Proceedings/07/Papers/141.pdf) |
| 7. 과거 환경의 physics model이 어긋나면 VOI 순위가 얼마나 흔들리는가? | Score의 calibration과 실제 posterior가 다른 실패를 검출한다. | Unseen contact law에서 score와 realized marginal benefit 순위, paired Brier를 측정. | [Huang, §7.1](https://proceedings.neurips.cc/paper_files/paper/2024/file/c59f05d7ab3638b138cc61f32e1a7cd1-Paper-Conference.pdf) |
| 8. 처리할 future world 수를 알면 우선순위가 달라지는가? | 장기간 쓰이는 환경에서 한 번의 calibration이 더 많은 결정에 영향을 준다. | Deployment horizon을 사전에 달리하고 per-world 및 horizon-weighted utility를 함께 비교. | [Kapoor, §3](https://www.ijcai.org/Proceedings/07/Papers/141.pdf) |
| 9. 평균 이득을 위해 일부 환경에 지나치게 큰 손실을 주는가? | 모든 환경에 결정을 제공하는 목적은 단일 winner를 찾는 목적과 다르다. | Worst-group Brier, upper-tail regret, minimum probe floor별 평균 성능을 함께 평가. | [Chen, §2.4](https://proceedings.mlr.press/v28/chen13f.pdf) · [Karnin, §4](https://proceedings.mlr.press/v28/karnin13.pdf) |
| 10. 더 정확한 scorer의 계산비를 몇 개의 미래 결정에 걸쳐 회수하는가? | Offline scoring과 online latency가 서로 다르다. | Reference bank 크기와 lookahead depth를 바꾸고 score 생성·setup·per-decision latency를 나누어 기록. | [Huang, §7.1](https://proceedings.neurips.cc/paper_files/paper/2024/file/c59f05d7ab3638b138cc61f32e1a7cd1-Paper-Conference.pdf) |

### 9. Research assistant review

**이전 결과가 주는 제한.** V28의 16회−neutral native Brier는 낮아졌지만, common-decoder Brier 차이는 −0.00118이고 simultaneous interval은 [−0.00538, +0.00303]이었다. 따라서 v29의 연구 질문은 “좋은 환경을 골라 관측을 사면 선택 이득이 *새 환경에서도* 재현되는가?”이다. 독립 환경을 새로 구성하고 사전에 고정한 comparator와 측정해야 한다. V28의 예산별 평균을 보고 유리한 환경을 고르는 것은 deployment policy의 근거가 되지 않는다. [V28 실험 §4](../calibration_probe_v28/REPORT_KO.md)

**기여 주장에 필요한 대조.** Fixed 4와 budget이 같은지 allocation block별로 확인해야 한다. 같은 2·4·8회 histogram의 random allocation과 비교하면 unequal allocation 효과에서 score targeting 효과를 분리할 수 있다. 두 정책이 같은 action을 고른 world의 common-decoder loss는 같아야 한다. Rank flip 빈도, posterior의 좁아짐, 높은 점수와 실제 환경 난도의 상관만으로 평균 loss 개선이 입증되지는 않는다. 해당 환경을 더 측정했을 때의 *추가 이득*이 클 때 targeting이 정당화된다. 한 환경이 받는 예산이 다른 환경의 점수에 의존하므로 추론 및 재표집 단위에는 공동 allocation block이 포함되어야 한다. [Chen, §2.4](https://proceedings.mlr.press/v28/chen13f.pdf) · [Frazier, §4](https://optimallearning.princeton.edu/Papers/FrazierPowellDayanik_KnowledgeGradientSICON.pdf)

**논문 범위.** 고정 예산, 단계적 탈락, VOI, decision-aware acquisition에는 각각 직접 선례가 있다. 이 연구가 보탤 수 있는 것은 calibration을 받는 과거 물체와 미래 물체가 다른 상황에서 **shared environment를 거치는 간접적인 decision value**를 동일 비용·동일 decoder로 측정하는 것이다. Fixed 4와의 차이가 불확실하면 방법의 평균 우위는 열린 문제로 남는다. 성공했다면 다음 근거는 독립적인 contact simulator나 실제 다른 재질에서의 외적 타당성이다. 다섯 편의 문헌만으로 분야 전체의 최초성을 주장할 수 없다. [Karnin, §4](https://proceedings.mlr.press/v28/karnin13.pdf) · [Chen, §2](https://proceedings.mlr.press/v28/chen13f.pdf) · [Huang, §§3–7](https://proceedings.neurips.cc/paper_files/paper/2024/file/c59f05d7ab3638b138cc61f32e1a7cd1-Paper-Conference.pdf)


## 재현 자료

서버 작업 폴더는 `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/adaptive_stopping_v29`이다. 세 allocation 중 다른 작업이 없던 node55의 RTX PRO 6000 Blackwell Server Edition 한 장에서 GPU guard 아래 순차 실행했다. 각 block에서 allocation과 선택의 hash, paired outcomes, 2·4·8회 각각의 risk/action, GPU timing을 보존했다. 수치 분석은 할당된 compute node의 CPU에서 수행했다. [사전 protocol](PROTOCOL.md) · [실험 설정](design.json) · [대표 block의 예산 동결](results/block_000/allocation_lock.json) · [주 실험 종료 기록](gpu_worker/results/worker_exit.json) · [timing 종료 기록](timing_worker/results/worker_exit.json) · [보조 실험 종료 기록](quota_aware_worker/results/worker_exit.json)
