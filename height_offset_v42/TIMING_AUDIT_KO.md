# V42 GPU timing의 분산

한 배치의 8개 환경·1,024개 물체에 대해 같은 GPU에서 각 방법을 무작위 순서로 실행했다. Catalog panel 12개 × 반복 4회의 실측값 48개가 방법별 표본이다. 아래 값은 사후 민감도 분석이다. 모든 측정값을 포함한다.

| Policy | Warm median | Warm p90 | Warm 최솟값–최댓값 | Cold p90 | 개별 panel 제외 시 warm p90 ratio 범위* |
|---|---:|---:|---:|---:|---:|
| original_zero | 1412.78 | 1426.94 | 1396.42–1435.70 | 1432.84 | 1.0003–1.0030 |
| estimated_zero | 1409.38 | 1423.51 | 1390.70–1430.08 | 1430.30 | 1.0000–1.0000 |
| estimated_fixed | 1420.32 | 1432.44 | 1403.67–1450.36 | 1439.28 | 1.0056–1.0074 |
| estimated_optimizer | 1422.24 | 1436.84 | 1402.33–1575.07 | 1443.68 | 1.0087–1.0106 |
| estimated_selector | 1423.06 | 1436.13 | 1405.98–1439.25 | 1442.71 | 1.0073–1.0101 |

시간 단위는 ms다. *Ratio의 분모는 동일 재표집에서 estimated + zero의 p90이다. 이 범위는 12회 leave-one-panel-out 결과로, 사전 지정 bootstrap confidence interval은 전체 보고서에 제시했다. 매 반복 내 실행 위치별 median과 각 panel의 최댓값을 [timing 원수치](timing_audit42.json)에 공개했다. Online elapsed time은 실제 GPU 계산 외에 CPU calibration·selection 및 동기화도 포함한다. Sensor 획득과 source bank 최초 구축은 제외했다.
