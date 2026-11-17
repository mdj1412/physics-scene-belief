# v41 방향 추정 검산

모든 값은 독립 passive calibration 물체에서 얻었다. Angle은 추정 방향과 oracle 방향의 부호를 무시한 각도이며 0°면 완전히 일치한다. 남는 변형 길이 비율은 sin(angle)이다. 같은 물체 pairing을 깨뜨린 control은 각 prefix에서 두 번째 profile만 한 행씩 옮겨 계산했다.

| Shape / height | 4 pairs | 8 pairs | 16 pairs | 32 pairs [panel 평균 95%] | 32 pairs 제거 후 남는 길이 | Unpaired 32 평균 각도 |
|---|---:|---:|---:|---:|---:|---:|
| flare / 19 mm | 34.79° | 22.14° | 19.67° | 12.43° [10.35, 14.51] | 21.5% | 79.82° |
| flare / 31 mm | 41.17° | 23.97° | 18.28° | 13.74° [10.72, 16.76] | 23.7% | 30.50° |
| roll / 19 mm | 45.96° | 34.88° | 28.38° | 17.01° [13.80, 20.23] | 29.1% | 75.45° |
| roll / 31 mm | 32.15° | 23.76° | 15.85° | 10.63° [7.84, 13.42] | 18.4% | 88.47° |

매 group의 첫 32개 pair를 사용하는 기준에서 48개 독립 group calibration에 1,536개의 추가 물체, 3,072회 profile readout이 필요했다. Calibrated group은 두 law 및 모든 variant·시작 block에 재사용했다. Angle을 서로 다른 eigensolver로 재검산했고 oracle 방향도 별도의 compact support 식으로 검산했다. [원자료](direction_audit41.json)
