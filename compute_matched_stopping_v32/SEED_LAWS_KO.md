# Training seed별 contact law와 stopping stage 효과

Loss 차이는 repeated(2×2)에서 broad(1×1)를 뺀 값이다. 양수는 repeated의 Brier가 더 높음을 뜻한다. Primary seed 1의 directional 결과를 확인한 후 이 분석을 정했고, 나머지 두 seed를 새로 학습했다. 세 seed 모두 같은 48개 target blocks에서 평가했다.

| Training seed | split_static | speed_weak | directional | directional − historical 평균 |
|---|---:|---:|---:|---:|
| 1 | -0.000144 [-0.000942, +0.000633] | +0.000129 [-0.000730, +0.001056] | +0.000716 [+0.000165, +0.001276] | +0.000724 [-0.000090, +0.001559] |
| 2 | -0.000515 [-0.001613, +0.000709] | -0.001299 [-0.002619, +0.000044] | +0.001120 [+0.000045, +0.002335] | +0.002027 [+0.000607, +0.003522] |
| 3 | +0.000236 [-0.000537, +0.000963] | -0.000033 [-0.001138, +0.001092] | +0.000802 [-0.000054, +0.001595] | +0.000701 [-0.000362, +0.001754] |
| 3 seed 평균 | -0.000141 [-0.000910, +0.000605] | -0.000401 [-0.001605, +0.000666] | +0.000880 [+0.000288, +0.001628] | +0.001150 [+0.000203, +0.002355] |

각 seed의 괄호는 같은 law 안에서 16개 target blocks를 복원추출한 pointwise 95% interval이다. 세 seed 평균에는 독립 training seed pair도 같이 재표집했다. 각 draw에서 target block index는 세 seed에 공통으로 적용했다. Historical 평균에는 split_static과 speed_weak의 가중치를 절반씩 주었다. Independent training seed가 3개이고, 개별 law·stage 비교의 multiplicity를 조정하지 않았으므로 crossed intervals는 탐색적으로 해석한다.

## Stage별 ΔBrier 기여

각 policy에 대해 4회까지 진행한 환경에서 R4−R2, 8회까지 진행한 환경에서 R8−R4를 계산하고, 각 합을 전체 환경 8개로 나눈다. 아래에서는 두 policy의 stage별 손실 변화를 뺀다. 4→8 기여에는 앞 stage에서 살아남은 환경의 구성이 반영된다.

| Training seed / law | 2→4 기여 | 4→8 기여 | 전체 ΔBrier |
|---|---:|---:|---:|
| 1 / split_static | -0.000509 | +0.000365 | -0.000144 |
| 1 / speed_weak | +0.000107 | +0.000022 | +0.000129 |
| 1 / directional | +0.000547 | +0.000169 | +0.000716 |
| 1 / 세 law 평균 | +0.000048 | +0.000186 | +0.000234 |
| 2 / split_static | -0.000363 | -0.000152 | -0.000515 |
| 2 / speed_weak | -0.001217 | -0.000082 | -0.001299 |
| 2 / directional | +0.000438 | +0.000682 | +0.001120 |
| 2 / 세 law 평균 | -0.000381 | +0.000150 | -0.000231 |
| 3 / split_static | -0.000245 | +0.000480 | +0.000236 |
| 3 / speed_weak | -0.000057 | +0.000024 | -0.000033 |
| 3 / directional | +0.000355 | +0.000448 | +0.000802 |
| 3 / 세 law 평균 | +0.000018 | +0.000317 | +0.000335 |
| 세 seed·세 law 평균 | -0.000105 [-0.000608, +0.000287] | +0.000217 [-0.000145, +0.000585] | +0.000113 [-0.000426, +0.000588] |

각 block의 실제 ΔBrier와 두 stage의 합 사이 최대 잔차는 4.16e-17였다. 전체 48개 block의 probability로 각 observation의 Brier를 다시 계산하고 stable top-13과 순차적 stopping을 재생했다. 독립 코드가 full-budget 재학습 분석의 policy별 Brier, 다섯 contrasts, 세 seed pooled target-only 및 crossed intervals를 절대 오차 2×10⁻¹⁴ 이내로 재현했다.

[분석 계획](SEED_LAWS_PLAN_KO.md) · [전체 추정치와 stage별 interval](results_eval/seed_laws32.json) · [독립 검증 코드](seed_laws32.py)
