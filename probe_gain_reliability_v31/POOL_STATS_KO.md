### 원래 1,024개 물체 안에서 추가 접촉 대상을 고른 보조 실험

별도의 18개 law-block 계산이 주 결과의 factual pool, paid prefix, bank, deployment worlds와 decoder Brier를 그대로 공유한다. 추가 대상은 처음 2개 또는 4개 paid IDs를 제외한 원래 pool에서 뽑는다. 한 future 안에서 중복은 없고 서로 다른 future에서는 같은 물체가 다시 선택될 수 있다. 각 draw는 별도의 probe noise를 사용한다. 따라서 이 조건에서는 추가 접촉과 동시에 새로운 factual 입력을 구입하지 않는다. 순위와 분산은 같은 고정 표본·paid prefix에 조건부다.

| Stage | Future variance ×10⁶ (비중) | Batch variance ×10⁶ (비중) | Interaction ×10⁶ (비중) | Batch−future variance ×10⁶ [95%] | 1×1 독립 concordance [95%] |
|---|---|---|---|---|---|
| 2→4 | 2.49 (3.8%) | 31.08 (47.7%) | 31.53 (48.4%) | 28.59 [22.41, 34.77] | 49.1 [47.8, 50.3]% |
| 4→8 | 1.64 (2.5%) | 32.82 (50.7%) | 30.24 (46.7%) | 31.18 [24.85, 37.51] | 50.3 [49.0, 51.6]% |

보조 panel 내부의 두 stage·두 질문에 separate simultaneous 95% interval을 사용했다. 새 물체와 finite pool의 paired 차이는 아래처럼 같은 law-block을 두 panel에서 함께 재표집해 계산했다.

| Stage | 대상 | Fresh | Pool | Pool − fresh [8-contrast simultaneous 95%] | 단일 block 제외 paired 차이 범위 |
|---|---|---|---|---|---|
| 2→4 | 평균 Brier gain | -0.00008 | -0.00025 | -0.00017 [-0.00062, 0.00028] | [-0.00031, -0.00011] |
| 2→4 | Batch−future variance ×10⁶ | 27.29 | 28.59 | 1.30 [-4.22, 6.82] | [0.07, 2.47] |
| 2→4 | 1×1 concordance (%), 차이는 %p | 49.46 | 49.05 | -0.41 [-2.16, 1.35] | [-0.73, -0.00] |
| 2→4 | 2×2 concordance (%), 차이는 %p | 49.40 | 48.54 | -0.86 [-6.19, 4.47] | [-1.80, 0.21] |
| 4→8 | 평균 Brier gain | -0.00077 | -0.00057 | 0.00020 [-0.00027, 0.00067] | [0.00006, 0.00028] |
| 4→8 | Batch−future variance ×10⁶ | 33.95 | 31.18 | -2.77 [-10.46, 4.93] | [-4.61, -1.39] |
| 4→8 | 1×1 concordance (%), 차이는 %p | 49.82 | 50.32 | 0.50 [-0.76, 1.76] | [0.27, 0.79] |
| 4→8 | 2×2 concordance (%), 차이는 %p | 49.83 | 49.96 | 0.13 [-3.61, 3.87] | [-0.35, 1.16] |

두 방법 모두 물체를 추출하지만 fresh는 새 물체와 그 factual 입력·probe 응답을 함께 더하고, pool은 이미 본 factual 기록을 선택해 추가 probe만 관측한다. 두 실험의 estimand는 다르다. 차이의 구간에 0이 포함되어도 두 방법의 동등성이 입증되는 것은 아니다. [사전 paired 분석 설계](POOL_STATS_PLAN_KO.md) · [Pool 추정량](results_pool/crossed_stats31.json) · [Paired 차이와 law/block sensitivity](results_pool/paired_pool31.json) · [CPU 검산 코드](paired_pool31.py)
