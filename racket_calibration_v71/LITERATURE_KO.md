# v71 문헌 검토와 독립 평가 감사

## 1. 연구별 질문·방법·표본·결과·한계

문헌의 공통 질문은 새 환경에서 적은 관측으로 **어느 부분을 바꿔야 하는지**이다. 아래에서는 원 연구 결과와 v71에서 확인할 질문을 다른 열에 두었다. Jia 등의 논문은 발행사 공개 abstract 범위로 한정했다.

| 원 연구와 질문 | 방법과 표본 | 관찰과 결론 | v71에 적용되는 범위 |
|---|---|---|---|
| Gossard (2026, preprint). Racket마다 다른 bounce dynamics를 예측하고 온라인에서 보정할 수 있는가? | 10개 구성, 6개 rubber 제품의 planar impacts. 공식 배포 원본 8,184행. 실측 velocity와 spin에서 state-dependent restitution 및 impulse response를 fit했다. §VI는 inverted 구성 1·2·3·4·9의 train data로 prior를 구성하고 test observations를 순차 반영했다. | Online 계산에 (1, 표면 상대속도, normal velocity) 특징의 linear prior를 썼다. 저자는 약 10회 burn-in, 약 30회 이후 velocity 오차 개선, 평균 약 0.06 m/s·최대 약 0.17 m/s 감소를 보고했다. | 같은 rubber type의 prior, velocity 중심 online curve다. 제품 전체를 미리 제외한 Grass의 release 부호 감지, expert-only·gate-only 대조, spin 개선 여부를 이 수치로 정할 수 없다. [원문 §III·V·VI, Fig. 8–9](https://arxiv.org/html/2604.11349v1) · [공식 데이터](https://github.com/gossardt/tt_rackets/tree/159c6f040dd7832570b537b3b5fd00a97fc28c15) |
| Jia 등. 다른 rubber의 속도·각도·회전 반응은 어떤 실측으로 비교되는가? | 네 종류 표면, 무회전의 다섯 입사속도와 topspin/backspin의 다섯 회전속도, 1,500 fps 측면 촬영 후 SIMI-Motion 분석. | 무회전 조건에서 입사속도와 restitution 사이 linear fit의 R²가 0.84를 넘었다고 보고했다. | 다른 실험진에서 표면과 입사조건을 바꾼 사례다. 조건평균·회귀 적합도로 held-out spin error나 few-shot gate 효과를 추정하기는 어렵다. [발행사 원문 abstract](https://doi.org/10.1177/17543371241310338) |
| Rinaldi 등 (2019). Surface 층과 입사각이 rebound를 어떻게 바꾸는가? | Foam과 foam+compact를 비교. 14 m/s, 3개 각도·3개 spin 설정, high-speed 실측과 companion FE model. | 표면 층에 따른 spin ratio의 상대적 효과가 normal과 tilted incidence에서 달라졌다. 동일한 재료 구성에서도 입사 조건이 관측 관계를 바꾼다. | 측정된 spin 자체의 부호와 v71의 $v_x'-Rω_y'$ 부호는 서로 다른 양이다. v71의 Grass 실패 원인을 FE상의 pimple 변형 하나로 귀속하려면 event별 추가 관측이 필요하다. [원문 §2–4, Fig. 5·6·9](https://www.mdpi.com/2076-3417/9/1/158) |
| Harrison·Sharma·Pavone (2018). 이전 task에서 학습한 prior를 새 task의 소수 관측으로 어떻게 갱신하는가? | ALPaCA에서 task family의 feature와 last-layer prior를 offline 학습하고 새 context로 Bayesian linear regression. 사례에 mass/length가 다른 simulated pendulum 및 friction/torso가 다른 hopper가 있다. | 동역학 실험에서 초반 몇 관측만으로 오차와 predictive likelihood가 개선됐다. | Source response를 보존하면서 소수 관측으로 expert를 정규화해 갱신하는 선례다. 학습 prior와 target task의 관계, 센서와 목적함수가 다르므로 racket에서의 표본수·정확도를 예고하는 수치로 쓰지 않는다. [원문 §3–6 및 Fig. 4–5](https://arxiv.org/abs/1807.08912) |
| Gossard·Tebbe 등 (2023). SpinDOE가 회전을 얼마나 정확히 읽는가? | Dot 검출, geometric hashing, orientation sequence의 regression. Motorized ball spinner의 측정값으로 검증하고 10개 영상, 약 26 ms를 사용한 시험을 보고했다. | 원문은 각속도가 매우 낮거나 높은 영역의 relative error 증가, 검출점 수와 ball logo에 따른 orientation failure를 구별한다. | 별도 계측 검증이다. Racket dataset의 각 impact에 같은 오차를 부여할 근거는 없다. $v_x'$와 $Rω_y'$가 비슷한 행에서 release sign이 민감한 이유와 margin sensitivity의 필요성을 보여준다. [원문 §IV.B–C](https://arxiv.org/abs/2303.03879) |

## 2. 문헌을 주제별로 읽은 결과

**표면을 바꾸면 달라지는 항.** Gossard는 normal rebound와 tangential impulse의 상태 의존성을 분리한다. Rinaldi 등은 표면 층을 바꿨을 때 같은 변형의 효과가 입사각에 따라 달라질 수 있음을 보였다. Jia 등은 무회전에서 간단한 restitution fit이 설명할 수 있는 범위를 보였다. 따라서 부호 하나가 잘 전이되는지와, 그 부호를 맞혔을 때 실수값 velocity·spin까지 맞는지는 따로 평가해야 한다. [Gossard](https://arxiv.org/html/2604.11349v1) · [Rinaldi 등](https://www.mdpi.com/2076-3417/9/1/158) · [Jia 등](https://doi.org/10.1177/17543371241310338)

**적은 관측으로 조정할 항.** Gossard는 같은 rubber type의 prior를 현재 상호작용으로 갱신했다. ALPaCA는 많은 source tasks에서 학습한 feature와 weight prior가 작은 target context를 활용하게 한다. v71에서는 같은 calibration impacts에서 routing만 변경했을 때와 새 표면의 연속 response까지 갱신했을 때의 차이가 핵심 대조다. [Gossard §VI](https://arxiv.org/html/2604.11349v1) · [ALPaCA](https://arxiv.org/abs/1807.08912)

## 3. Knowledge map

Rubber·sponge·blade 및 입사 상태 → 접촉 동안의 변형·impulse → 이탈 translation과 rotation → 영상으로 추정한 $s'=v_x'-Rω_y'$ → calibration의 disagreement → gate 방향 결정 → expert의 연속 예측 → 실제 held-out velocity·spin 오차. ALPaCA의 prior는 마지막 예측과 갱신 사이에 대응한다. SpinDOE는 rotation 추정 단계에 오차를 도입할 수 있다. **직접 비교 가능한 연결**은 공개된 입·출력으로부터 계산한 disagreement, routing, endpoint loss다. **남는 연결**은 pimple 변형, 압력 분포, 국소 stick/slip이 해당 부호를 발생시킨 경로다. [Gossard 데이터](https://github.com/gossardt/tt_rackets) · [SpinDOE](https://arxiv.org/abs/2303.03879) · [Rinaldi 등](https://www.mdpi.com/2076-3417/9/1/158)

## 4. 근거에서 드러난 공백

- **추가 표면의 독립 반복.** 공개 데이터에서 Grass D.TecS는 같은 제품을 sponge 두께 1.2 mm와 0 mm로 바꾼 구성 두 개다. 이 둘에서 방향이 같아도 독립 제조사 long-pips 두 제품으로 계산할 수 없다. v70에서 이미 발견한 사례라는 점도 결과 해석에 남는다. [구성 표](https://github.com/gossardt/tt_rackets/tree/159c6f040dd7832570b537b3b5fd00a97fc28c15)
- **감지 대상과 최적 행동의 불일치.** $s'$ 부호를 틀렸는지에는 1표씩 주지만, 틀린 방향의 expert가 만드는 m/s·rad/s 오차는 shot마다 다르다. Gossard의 목표 지표는 실제 velocity·spin이다. 부호 정확도를 최종 목표로 두면 작은 오차를 여러 번 낸 경우와 큰 오차를 한 번 낸 경우를 구별하지 못한다. [Gossard §V.E](https://arxiv.org/html/2604.11349v1)
- **교정량의 식별.** 소수 calibration에서 비슷한 입사 조건만 관측하면 expert slope와 offset을 동시에 안정적으로 정하기 어렵다. Source를 따르는 regularization과 새로운 입사각·spin을 포함하는 calibration 설계를 분리해 볼 이유다. [ALPaCA](https://arxiv.org/abs/1807.08912) · [Rinaldi 등](https://www.mdpi.com/2076-3417/9/1/158)
- **Measurement uncertainty.** SpinDOE의 별도 성능은 ground truth가 알려진 장치에서 나온다. 각 impact의 joint velocity–spin uncertainty와 annotation repeatability는 공개 14열에서 확인할 수 없다. [SpinDOE](https://arxiv.org/abs/2303.03879) · [Gossard 자료](https://github.com/gossardt/tt_rackets)

## 5. 상충하는 듯한 연구의 비교

Gossard의 10–30회와 v71의 몇 회는 같은 sample-complexity 추정량이 아니다. 원 논문은 inverted 계열 prior에서 연속적인 restitution과 response를 새 관측으로 수정하고, velocity 오차의 평균 변화를 그렸다. 이미 학습된 gate의 이진 방향 교환 여부는 훨씬 제한된 가설이다. 이진 방향을 적은 샘플로 알아도 연속 예측이 충분히 정확해졌다는 결론은 별도 검증을 요구한다. Jia 등의 무회전에서 강한 선형 상관과 Rinaldi 등의 oblique 조건에서 관찰한 spin 효과도 서로 다른 조건·지표를 설명한다. [Gossard §VI](https://arxiv.org/html/2604.11349v1) · [Jia 등](https://doi.org/10.1177/17543371241310338) · [Rinaldi 등](https://www.mdpi.com/2076-3417/9/1/158)

## 6. 방법 비교와 독립 평가 감사

| 확인할 항목 | 공정한 해석 |
|---|---|
| Rubber holdout | Source gate와 source expert에서는 동일 제품을 모든 blade·sponge 조합에서 제외한다. Target calibration은 공식 train에서 구입한 prefix만, 최종 평가는 공식 test로 제한한다. [구성 목록](https://github.com/gossardt/tt_rackets/tree/159c6f040dd7832570b537b3b5fd00a97fc28c15) |
| Budget | 동일한 draw와 prefix를 gate-only, gate+expert, inclined rigid, single affine 모두에 준다. Frozen comparator와 달리 각 baseline도 같은 수의 target observations로 parameter를 갱신해야 한다. [ALPaCA](https://arxiv.org/abs/1807.08912) · [Gossard](https://arxiv.org/html/2604.11349v1) |
| 계측 시점 | Calibration shot의 $s'$는 그 shot 종료 후 알려진다. 이후 test shot은 incoming state로 예측한다. 이번 실험의 비용 단위는 이미 기록된 impact 수다. 실제 sensing latency·reset 비용은 별도 검증 대상이다. [Gossard §III](https://arxiv.org/html/2604.11349v1) |
| 반복 단위 | 같은 pool에서 뽑은 무작위 prefix 수는 calibration draw의 변동을 나타낸다. 예측한 test shots와 물리적 표면은 반복마다 같다. 새 표면의 유효 표본 수를 draw 횟수로 늘릴 수 없다. [Gossard 공개 자료](https://github.com/gossardt/tt_rackets) |
| 역사적 재사용 | v70에서 같은 test set으로 Grass를 실패 사례로 발견했다. v71 test labels를 개별 fit에서 숨기는 것과, 미지의 대상에서 처음 가설을 확인하는 것은 다른 검증 단계다. 결과는 알려진 개발 사례를 이용한 protocol replay이며, 최종 confirmatory test는 새 acquisition이 필요하다. [데이터 release](https://github.com/gossardt/tt_rackets/tree/159c6f040dd7832570b537b3b5fd00a97fc28c15) |

또 하나의 구별은 **source prediction에 대한 global disagreement**와 **같은 shot에서 incoming sign이 outgoing sign으로 바뀌는 물리적 reversal**이다. v71의 detector가 직접 검정하는 것은 전자다. Global swap이 평균적으로 맞는 제품에서도 특정 입사 조건의 모든 shot이 뒤집힌다는 뜻은 아니다. 한 제품 안에서 pooled calibration 자료의 빈도가 달라지면 관찰되는 전환율도 바뀔 수 있다. [Gossard §II–V](https://arxiv.org/html/2604.11349v1)

## 7. Evidence synthesis

**강하게 뒷받침되는 부분**은 장비가 측정한 rebound response가 rubber와 입사 조건에 따라 달라지고, prior에 새 표면의 관측을 넣어 오차를 줄일 여지가 있다는 점이다. **제한적 근거**는 gate disagreement의 공간적 범위와 지속성이다. 이 실험은 동일한 여섯 제품과 기록된 조건에서 비교할 수 있다. **남는 불확실성**은 새 제조사의 long pips, 완전히 다른 spin 구간, 다른 공·날짜·습도에서의 검출과 계수 전이다. 공개 자료에서 그 방향을 물리적으로 일으킨 구조나 실제 경기 성능은 확인되지 않는다. [Gossard](https://arxiv.org/html/2604.11349v1) · [Rinaldi 등](https://www.mdpi.com/2076-3417/9/1/158) · [Jia 등](https://doi.org/10.1177/17543371241310338)

## 8. 자료와 공백에 근거한 후속 질문 10개

| 질문 | 필요한 변수·이유 | 검증 방식과 근거 |
|---|---|---|
| 1. 다른 제조사의 long pips에도 같은 global flip이 필요한가? | 제품·batch, disagreement 빈도, velocity·spin 오차 | 미관측 제품과 acquisition session에 v71 rule을 고정하고 평가. [Gossard](https://arxiv.org/html/2604.11349v1) |
| 2. 같은 rubber의 OX에서 배운 방향이 sponge 장착 시에도 통하는가? | sponge 두께, 입사 상태, 방향·magnitude | 한 구성 calibration으로 다른 구성 test를 평가한 뒤 순서를 바꾼다. [Gossard 구성 표](https://github.com/gossardt/tt_rackets) |
| 3. 부호보다 예상 velocity·spin 손해로 투표하면 나아지는가? | 같은 disagreement에도 다른 endpoint 손실 | Source-only로 손실 차이를 예측하고 동일 prefix의 단순 투표와 대조한다. [Gossard §V.E](https://arxiv.org/html/2604.11349v1) |
| 4. $|s'|$가 작은 calibration label을 보류할 이득이 있는가? | spin 측정 오차·margin·구입한 총 impact 수 | 사전에 margin 규칙을 정하고 동일 sensing budget의 무작위 보류와 비교한다. [SpinDOE §IV](https://arxiv.org/abs/2303.03879) |
| 5. 실제 표면별로 global flip 하나면 충분한가? | 입사각·normal speed별 disagreement 빈도 | 입사 조건을 층화해 고정 gate swap과 source에서 지정한 조건별 swap을 비교한다. [Rinaldi 등](https://www.mdpi.com/2076-3417/9/1/158) |
| 6. 두 expert에 공유하는 보정만으로 충분한가? | 공유 offset, slope, branch별 residual | 같은 budget에서 shared-only, branch-specific, shrinkage update를 nested holdout한다. [ALPaCA](https://arxiv.org/abs/1807.08912) |
| 7. 새 표면에서 입사 범위를 넓게 고르면 필요한 횟수가 줄어드는가? | calibration state coverage·held error | Random sampling과 outcome-free space-filling sampling을 같은 pool에서 paired 비교한다. [Jia 등](https://doi.org/10.1177/17543371241310338) · [Gossard](https://arxiv.org/html/2604.11349v1) |
| 8. 같은 예산에서 physical parameter prior와 affine prior 중 어느 쪽이 유리한가? | contact type, restitution, tangential gain, shrinkage strength | 같은 source와 target calibration으로 update하고 제품별 두 metric을 비교한다. [Gossard §VI](https://arxiv.org/html/2604.11349v1) · [ALPaCA](https://arxiv.org/abs/1807.08912) |
| 9. 부호와 오차의 순위가 장비·spin 추정 절차를 바꿔도 유지되는가? | camera·marker, repeated annotation, confidence | 같은 impact에서 독립 spin 추정을 생성해 detector·loss의 불일치를 측정한다. [SpinDOE](https://arxiv.org/abs/2303.03879) |
| 10. 감지 증거가 충분해지면 일찍 멈춰도 되는가? | stopping time, false flip, session loss | Sequential rule을 source에서 고정하고 동등 평균 비용의 fixed budget과 새 acquisition에서 비교한다. [Gossard §VI](https://arxiv.org/html/2604.11349v1) |

## 9. 논문 기여에 대한 검토

Gossard가 이미 few-observation online adaptation을 제안했고, ALPaCA는 source prior와 새 관측으로 연속 출력을 갱신하는 일반 원리를 제시했다. v71의 기여 후보는 **같은 calibration evidence가 잘못된 routing을 얼마나 고치며, 고친 뒤에도 연속 response에 얼마의 오차가 남는지 실제 rubber 단위로 분해하는 것**이다. Downstream 성능에서는 단순 single affine 및 physical model이 동일 budget으로 적응했을 때의 결과를 함께 제시해야 한다. 불리한 제품이나 budget도 그대로 내야 새로운 방법의 적용 조건을 평가할 수 있다. [Gossard](https://arxiv.org/html/2604.11349v1) · [ALPaCA](https://arxiv.org/abs/1807.08912)

독립 검산에서는 v70의 source-only 예측식을 재구성하고, 공개 calibration pool을 전수 집계해 무복원 표본의 gate flip 확률과 gate-only endpoint loss의 기대값을 계산했다. 이 집계는 평가자만 접근한다. 실제 detector는 받은 prefix만 사용해야 한다. [독립 계산 원리와 결과](review/REFERENCE_KO.md)
