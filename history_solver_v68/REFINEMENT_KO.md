# 사후 수치 민감도 · v68

사전 지정한 17-site×half-step 수치 대조에서 16개 family×law×solver×contrast 가운데 2개 부호가 달라졌다. 결과를 본 뒤 추가한 확인이다. 신규 모델 fit 또는 cutoff 조정 없이 compliant target의 총 12 catalogs와 전체 128 prototype/environment를 17 sites, time step 1/(48×9,000) s로 재계산한다. 기존 fine subset의 32개 값과 bitwise agreement를 요구한다. 33 sites와 1/(96×9,000) s의 추가 대조는 그 12 catalog·4 environment 각각 처음 8개 prototype으로 정해 결과에 따라 선택하지 않는다. 모든 surface, 높이, law, shape variant, offset, radius가 같다.

같은 사전 봉인 standard history·forecast와 동일한 source seed 네 개를 이용한다. 먼저 각 catalog에서 네 seed를 평균하고 12개의 paired catalog를 단위로 각 해상도별 두 law×(두 family의 soft−full과 두 family pooled restored−dropped) 6개 Bonferroni interval을 낸다. 33-site subset에서는 동일한 8개에 대한 17-site 결과도 함께 제시한다. 개별 family의 좌표 복원은 pointwise 보조 결과다. 모든 미확정 행의 binary 선택을 허용한 점추정 최악범위도 계산한다. 이는 whole-history fine forecast 평가와 구별되는 terminal-only numerical sensitivity다.
