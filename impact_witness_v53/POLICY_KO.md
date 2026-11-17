# v53 · 하나의 impact 방향으로 무엇을 추정할 수 있는가

교체 전 네 물체와 교체 후 첫 세 물체에서는 같은 위치에서 0.4 N weak displacement와 2.0 N reference displacement를 측정한다. 교체 후 네 번째 물체에서는 0.4 N 측정을 유지하고 두 번째 접촉을 배포 조건의 simulated impact로 바꾼다. 매 접촉 뒤 한 번 reset한다. 네 번째 물체가 앞선 세 물체와 다르며 deployment support가 calibration support와 분리되어 있는지 검산했다. Reference와 impact에서 방향 판독에 필요한 관측 시간과 reset 시간은 장비에 따라 달라진다. 같은 8 contacts·8 resets가 실물에서 같은 시간을 보장하지는 않는다. [사전 설계](PLAN_KO.md)

**실제로 입력되는 측정.** v50의 동일 물체에 대한 두 endpoint intervention은 같은 sensor noise와 bias를 공유한다. 개별 2.0 N counterfactual 측정값의 차이에서 latent reference 방향을 복원한 뒤, v52의 reference–impact error 확률로 독립 반전시켜 calibration 물체의 full-load 상태를 지정한다. 해당 물체 family와 높이에서 impact 순간의 loaded contour와 relaxed contour를 각각 두 위치(접촉 중심 ±1.5 mm)에서 읽어 differential displacement의 부호 −1·0·+1을 얻는다. 방향을 복원하는 counterfactual 계산은 관측 generator 안에서만 한다. 정책은 네 번째 weak 측정과 impact 방향만 읽는다. 네 번째 reference 값을 NaN으로 바꾸어 모든 mixed arm 출력이 정확히 보존되는지 검사했다. 이 모델에서 contour state는 contact solver에 공급되는 외생 변형이다. 실험 장비나 deformable-body solver에서 측정한 시간별 변형률로 일반화하려면 독립 검증이 필요하다. [관측 generator와 정책](policy53.py)

**Posterior.** κ는 post-swap active 물체의 weak→reference 방향 유지율이고 e는 reference→impact 방향 불일치율이다. 교체 후 weak→impact 유지율 q는 e+(1−2e)κ이다. Pre/post κ는 v52와 같이 유지되거나 반전되며 drift prior는 0.25다. e의 prior는 0.5×Beta(1,4)로 가정해 nominal mean 0.10을 v52와 맞춘다. κ에는 16-point Gauss–Legendre quadrature를 사용했다. Impact 방향이 활성 ±1이면 weak displacement의 Gaussian likelihood를 정·역 방향의 mixture로 평가한다. 정확한 0은 inactive를 뜻하고 방향에 대한 likelihood ratio는 1이다. 한 impact observation의 likelihood가 q에 대해 선형이므로 predictive posterior mean을 구하는 데 e의 두 번째 moment까지만 필요하다. 24-point 독립 numerical integration과 결과가 일치했다. 네 쌍 모두 reference로 관측한 경우에는 e prior의 평균만 작용해 v52 continuous joint의 예측을 모든 panel에서 재현했다. [Policy 검산](calibration53.py)

**비교.** Continuous joint는 p(+d)·p(−d)의 혼합 가중치로 posterior mean q를 쓴다. v52 세 행동을 현재 mixed 측정에 맞춘 selector는 같은 q에서 v47 source가 학습한 Brier risk를 계산한다. Keep은 pre-swap standalone κ, refit은 세 reference와 한 impact로 갱신한 post-swap standalone κ, unsigned는 기존 unsigned decoder를 쓴다. 같은 세 행동의 vote selector는 관측 크기를 버리고 readable sign agreement를 센다. Post-only vote는 네 번째 weak–impact agreement도 한 표로 넣는 Laplace-smoothed count다. Continuous vote는 sign count를 쓰면서 continuous blending을 유지해 magnitude의 효과를 검사한다. [동일 행동·동일 관측 정책](policy53.py) · [Continuous vote](vote_joint53.py)

| 숨긴 reference–impact error e | 0 | 0.25 | 0.50 |
|---|---:|---:|---:|
| Posterior mean e | 0.09937 | 0.10022 | 0.10104 |
| 한 impact 방향이 바꾼 q의 평균 절댓값 | 0.02998 | 0.03267 | 0.03531 |
| 활성 방향이 관측된 경우 같은 변화량 | 0.06090 | 0.06637 | 0.07174 |

Calibration-only audit은 12개 catalog, 5개 pre-swap κ, 4개 drift rate를 균등 가중했다. 네 번째 물체의 49.23%에서 활성 방향이 관측됐다. e=0과 0.50을 구분하는 posterior mean e의 AUROC는 전체 0.591, 활성 물체에서 0.647이었다. 한 관측으로 환경들의 순서는 다소 구별해도 e의 크기는 prior에 가깝게 남는다. 측정을 얻어 q가 바뀌는 것과 e를 정확히 식별하는 것은 별개의 결과다. κ=.5에서는 e가 달라져도 q=.5이고, 유지·반전 두 조건의 policy 출력이 수치적으로 같았다. 일반적인 κ에서도 한 weak–impact 쌍은 q에 대한 증거를 주며 e와 κ를 독립적으로 측정하지 않는다. 같은 네 번째 물체의 2.0 N 값을 관측하려면 별도 접촉이 필요하다. [Posterior diagnostics](calibration53.json) · [식별성 검산](identifiability53.json)

Primary는 exact ternary direction readout을 가정한다. 별도 sensitivity에서 실제 활성 부호만 10% 확률로 교란했고 10.10%가 뒤집혔다. Likelihood에도 동일한 10%를 반영했다. 이는 실제 sensor의 오류율·false activation을 추정한 값이 아니다. Reader의 uncertainty, prior를 uniform(0,0.5)로 변경한 대응 실험, 같은 네 번째 reference 또는 masked impact의 lower-information 대조는 본 보고서에 함께 비교한다. [실행 전 코드 checksum](policy_freeze53.sha256) · [Policy cache checksum](calibration_results53.sha256)

**같은 posterior를 다른 행동으로 바꿨을 때.** 사전 지정한 비교를 마친 다음, v52에서 이미 고정한 v47 source Brier polynomial의 최소점 α*=clip(−c₁/(2c₂),0,1)을 같은 q에서 계산했다. α*만 쓰면 평균 0.158532, session p90 0.182411로 q를 그대로 혼합하는 정책의 0.158519·0.182310보다 둘 다 높았다. Source가 unsigned risk를 더 낮게 평가하는 35.07%의 prefix에서 unsigned로 전환하면 0.158470·0.182391이 됐다. 원래 continuous joint 대비 평균 차이는 −0.00004890, session p90 차이는 +0.00008012였다. 같은 posterior에서도 어떤 action을 허용하느냐가 mean과 tail에 서로 다른 영향을 준다. 동일 source-opt/unsigned를 네 reference에 적용하면 0.158506·0.182488로, 해당 decision rule에서는 impact 대체가 두 지표 모두 개선했다. 이 진단은 target label로 refit하지 않았고 source-risk coefficient도 같지만 primary 결과를 안 뒤에 설계했다. [효과 크기와 nominal CI](action_risk53.json)
