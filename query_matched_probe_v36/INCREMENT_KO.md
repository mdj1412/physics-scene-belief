# 네 번째 factual feature를 공통으로 제공한 대조

| 조건 | Gate4 − gate3+x4 Brier [paired 95%] | Gate3+x4 − gate3 Brier [paired 95%] | Gate3+x4 적용률 | Paid velocity를 보고 결정이 바뀐 비율 |
|---|---:|---:|---:|---:|
| local_equal | -0.001253 [-0.001699, -0.000790] | -0.000039 [-0.000082, -0.000001] | 7.0% | 33.9% |
| strong | -0.003292 [-0.003889, -0.002713] | -0.000075 [-0.000133, -0.000021] | 7.0% | 36.1% |
| weak | -0.000816 [-0.001304, -0.000316] | -0.000035 [-0.000078, +0.000003] | 7.0% | 33.8% |
| null | +0.000351 [-0.000136, +0.000873] | -0.000008 [-0.000048, +0.000035] | 7.0% | 31.8% |
| global_null | -0.000156 [-0.000594, +0.000328] | +0.000042 [-0.000012, +0.000105] | 23.8% | 42.6% |

같은 첫 세 접촉과 네 번째 factual feature에서 strong과 weak/null을 구별하는 posterior strength의 AUROC는 0.500, 네 번째 paid velocity까지 주면 0.524 [paired 95% 0.501, 0.546]였다. 세 local regime의 기존 관측들이 완전히 짝지어 있으므로 paid velocity 이전의 strong 및 weak/null score 분포는 같다. AUROC는 “보정 여부”가 아니라 source posterior strength의 순위를 평가한다. 강도 차이를 일부 읽어도 Brier 개선과 적용 여부의 순위는 달라질 수 있다.

세 local regime에서 같은 물체와 noise를 짝지은 상태로 gate decisions만 regime들 사이에서 무작위 permutation했을 때의 평균 loss와 실제 배정을 비교했다. 각 조건의 원래 deployment action, base probability와 실제 query label을 사용했다. 실제 − 무작위 배정 Brier는 -0.000179 [-0.000270, -0.000086]였다. 세 regime에서 gate decision이 모두 같은 경우는 85.7%였다. 이 대조는 환경별 평균 적용률과 네 번째 접촉에서 드러나는 공통 물체 성질을 고정한다. 여기에서 음수이면 강도 차이 때문에 바뀐 결정 자체가 유익했다는 근거가 된다.

각 CI는 조건별 95% paired cluster bootstrap이다. Object–surface와 첫 motion을 층화한 32개 block 안에서 orientation, regime, 환경과 32개 observation noise를 묶었다. Global-null은 4개 cell × 8개 독립 block에서 각각 실행했다. 보조 score는 동결한 source Gaussian을 marginalize해 계산했고 별도의 모델 학습이나 GPU timing은 수행하지 않았다. [수치](increment36.json) · [계산](increment36.py)
