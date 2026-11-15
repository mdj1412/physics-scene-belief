# 학습한 contact law에서도 추가 probe의 가치를 예측했는가?

**독립적으로 재생한 Coulomb 이력 48개에서 두 scorer 모두 추가 접촉의 평균 이득과 양의 상관을 보이지 않았다.** 이 비교는 source law를 그대로 둔 채 훈련에 등장하지 않은 prefix로 바꾼 것이다. 따라서 target law의 변화만으로 약한 예측력을 모두 설명하기는 어렵다. 이 표본에서 예측할 수 있었던 신호의 크기는 여전히 불확실하다. [사전 분석 계획](SOURCE_REFERENCE_PLAN_KO.md) · [수치와 interval](results_reference/source_reference32.json)

v31의 finite-pool panel에서 서로 독립인 Coulomb blocks 6개를 골랐다. 각 block의 8개 환경은 모두 포함했다. 환경마다 이미 지불한 2개 또는 4개 접촉에서 나온 15개 features로 v32의 두 frozen scorer를 각각 실행했다. 실제 label은 같은 prefix에서 추가 접촉 경로 4개와 배포 집단 4개를 교차해 얻은 16개 Brier gain의 평균이다. 한 cell의 gain은 같은 배포 집단·decoder로 측정한 추가 접촉 전 Brier에서 접촉 후 Brier를 뺀 값이다. v31의 모든 historical labels는 v32 fit에서 배제했다. [v31 측정 설계](../probe_gain_reliability_v31/REPORT_KO.md) · [CPU scoring 코드](source_reference32.py)

| Stage | Scorer | MSE ×10⁶ Brier² | 해당 훈련 평균만 쓴 MSE ×10⁶ | MSE 차이 ×10⁶ [95%] | Pearson r [95%] |
|---|---|---:|---:|---|---|
| 2→4 | Broad 1×1 | 9.013 | 8.649 | +0.365 [−0.379, +1.092] | −0.054 [−0.292, +0.180] |
| 2→4 | Repeated 2×2 | 10.002 | 9.787 | +0.216 [−0.482, +0.795] | −0.118 [−0.373, +0.095] |
| 4→8 | Broad 1×1 | 9.604 | 9.619 | −0.015 [−0.285, +0.242] | −0.033 [−0.235, +0.170] |
| 4→8 | Repeated 2×2 | 10.292 | 10.244 | +0.048 [−0.397, +0.440] | −0.023 [−0.239, +0.197] |

MSE 차이는 각 scorer에서 자기 arm의 intercept-only MSE를 뺀 값이다. 어느 stage에서도 feature를 이용해 훈련 평균 한 개를 쓰는 것보다 MSE가 낮아졌다고 판단할 수 없었다. Repeated−Broad MSE 차이는 2→4에서 +0.989 [−0.361, +2.338], 4→8에서 +0.688 [+0.096, +1.217]이었다(모두 ×10⁻⁶ Brier²). 4→8에서 두 arm의 intercept-only MSE 차이만으로도 +0.625×10⁻⁶이었다. 같은 stage에서 실제 held-out 평균 gain은 −0.000912였고 훈련 평균은 Broad +0.000605, Repeated +0.000799였다. 이 stage의 MSE 우열에는 평균값을 맞힌 정도가 크게 기여한다. 환경 사이 순서를 더 잘 예측했다고 볼 근거는 약하다. Pearson r의 Repeated−Broad 차이도 두 stage에서 각각 −0.064 [−0.248, +0.082], +0.009 [−0.180, +0.277]이었다. [전체 지표와 pointwise intervals](results_reference/source_reference32.json)

**어디까지 해석할 수 있나.** V31의 4×4 평균에도 finite-label noise가 남으며 두 축을 공유하는 16개 cell을 독립 표본으로 셀 수 없다. Pearson correlation이 약하면 예측 가능한 신호가 작을 수도 있고 label이 부정확할 수도 있다. 표의 interval은 6개 blocks를 10,000회 paired bootstrap한 pointwise 구간이다. Scorer를 다시 훈련했을 때의 변동과 임의의 다른 물리 법칙은 포함하지 않는다. 4→8에는 모든 환경의 이미 관측한 4회 prefix가 들어간다. 실제 두 단계 정책에서 첫 선택을 통과한 환경만을 평가한 결과와는 모집단이 다르다. 이 표로 최종 deployment Brier나 전체 stopping trajectory의 기준선을 대체할 수 없다. [v31 반복 실험](../probe_gain_reliability_v31/REPORT_KO.md) · [v32 통계 계획](STATISTICAL_PLAN_KO.md)

**재현 기록.** 서버 node56의 allocation 2744469에서 CUDA를 비활성화한 CPU float64로 재구성했다. 당시 저장한 v30 예측값의 최대 절대 오차는 5.64×10⁻¹⁷, posterior friction SD의 최대 절대 오차는 2.89×10⁻¹⁵였다. 원본 prefix·metric hashes와 훈련 기록의 prefix seed 비중복을 확인했다. 새 모델을 학습하거나 historical labels로 parameter를 조정하지 않았다. [Feature 봉인 및 검산](results_reference/prefix_seal.json) · [원본 지문과 model lock 지문](results_reference/source_reference32.json)
