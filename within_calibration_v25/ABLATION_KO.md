### 생성한 proxy에 얼마나 의존했는가

추가 분석에서는 전체 모델에서 두 가지 제한 모델을 만들었다. 하나는 profile mean과 수평 velocity만 쓰고, 다른 하나는 두 값을 제거한 나머지 네 features를 쓴다. 각 제한 모델을 원래 training/validation worlds에서만 24개 후보로 다시 학습한 뒤, 고정된 test K1024 fallback과 같은 decoder로 평가했다. 별도 prior width rule은 observation의 두 proxy를 generator의 offset 및 restitution 분포에서 유도한 근사 log-odds로 변환해 Q/P를 고른다. 이 rule은 실제 cell이나 label을 보지 않으나 generator 관계식을 사용한다. 이 표는 primary 계획 이후의 CPU replay로, 추가 p90은 측정하지 않았다.

| 모델 | Held-out Brier | 직접 선택률 | 모델−전체 gate [95% CI] | 모델−K256 [95% CI] | 전체 gate와 달랐던 고유 worlds |
|---|---|---|---|---|---|
| 전체 factual features | 0.076195 | 81.0% | +0.000000 [+0.000000, +0.000000] | -0.000113 [-0.002066, +0.002067] | 0 |
| Profile mean 및 수평 velocity | 0.075975 | 60.4% | -0.000221 [-0.001296, +0.000829] | -0.000333 [-0.002168, +0.001741] | 12 |
| 위 두 features 제거 | 0.080588 | 19.8% | +0.004392 [+0.001386, +0.007361] | +0.004280 [+0.000989, +0.007730] | 66 |
| 훈련 없는 prior width rule | 0.093122 | 100% | +0.016927 [+0.010569, +0.023504] | +0.016814 [+0.010682, +0.023268] | 306 |

| 전체 모델의 score 진단 | 고유 worlds | Spearman ρ · score와 실현 P−Q | Score 부호와 실현 winner 일치 | 평균 score / 실현 P−Q |
|---|---|---|---|---|
| train | 3072 | 0.582 | 80.0% | +0.0694 / +0.0694 |
| validation | 1024 | 0.549 | 79.5% | +0.0665 / +0.0585 |
| test | 1280 | 0.525 | 79.7% | +0.0699 / +0.0648 |

| Test score quintile | 평균 예측 advantage | 평균 실현 advantage | Probe 우세 worlds |
|---|---|---|---|
| 1 | -0.1311 | -0.1351 | 76.6% |
| 2 | +0.0099 | +0.0344 | 41.0% |
| 3 | +0.0912 | +0.0909 | 23.4% |
| 4 | +0.1522 | +0.1427 | 12.1% |
| 5 | +0.2272 | +0.1912 | 2.7% |

두 noise halves에서 0.02 이상 안정적으로 Q가 유리했던 worlds 중 직접 선택한 667개에서 Q를 놓친 경우는 39개였다. P가 안정적으로 유리했던 worlds 중 직접 선택한 278개에서는 P를 놓친 경우가 99개였다. Held-out score는 선택 전의 예측이고, 여기서 대조한 winner는 실제 hidden state와 반복 observation에 조건화된 realized loss로 정해진다.
