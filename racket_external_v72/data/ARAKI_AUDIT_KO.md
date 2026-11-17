# Feint OX의 Figure digitization 기록

Araki, Sato, Yamazaki (1996)의 *Collisional Properties of Ball-Racket Interactions in Terms of Normal and Tangential Coefficients of Restitution*에서 Feint OX의 Figure 11과 12를 사용했다. 원본 PDF의 SHA-256은 `3a8ffa283f0a3f26647d191e0d171d44d73352da86f51047b83892ac002ea86c`이다. [원본](https://sasportssience.blob.core.windows.net/ijtts/IJTTS_3_pdf%20files/IJTTS_3_17_49_Araki_Collisional.pdf)

1. **Normal restitution.** PDF 28쪽(인쇄 44쪽)을 110 dpi로 렌더했다. Figure 11의 hollow square 17개 중심을 Figure 12와 독립적으로 읽었다. `araki_fig11.py`에 raster 크기와 crop, marker 좌표, 축 끝점을 기록했다. 스캔된 수평축이 기울어져 있어 각 x에서 위·아래 경계를 선형 보간했다. `normal_v_out_nominal_mps`는 이 비율에 명목 입사 속력 30 km/h를 곱한 값이다. 원본 영상으로 읽은 개별 trial의 절대 normal velocity를 확보하려면 매 trial의 실제 입사 속력도 필요하다. 이 축에서 원본 raster 1 pixel의 y 차이는 normal restitution 약 0.003, 환산 속력 약 0.025 m/s이다. 이 값은 판독 해상도의 척도이며 전체 측정 오차의 상한을 제공하지 않는다.
2. **Tangential velocity와 spin.** PDF 29쪽(인쇄 45쪽) Figure 12(a)의 u′와 (b)의 n′는 서로 다른 패널에 표시되어 있다. `feint_fig12_centers.csv`에 marker 좌표가 있다. 약 15 rps 구간에서 두 출력 marker가 모두 0 부근에 보인다. 본문에 묘사된 평균적 경향을 이용한 보간은 적용하지 않았다. n 단위는 회전수/s로, rad/s 변환에는 2π를 곱한다.
3. **Panel matching.** 실험 일자 표식과 incident-spin 축을 공유한다. 약 21 rps, 161 rps 부근에 두 점씩 있어 두 패널 사이의 marker 대응을 단정하기 어렵다. 총 네 점을 제외하고 남은 13개는 세 패널 x 좌표의 전체 폭이 2 rps 이내인 유일한 조합으로 대응했다. `join_feint.py`는 중복 사용과 복수 후보를 검사한다. 각 패널의 x 값과 최대 차이는 `feint_ox_digitized.csv`에 남겼다. 공개된 trial ID가 없어 정확한 대응은 x 좌표에 근거한 추론이다.
4. **Release direction.** 구형 ball의 radius 0.019 m를 사용하면 v72 좌표에서 접선 상대 속도는 u′ − 2π·0.019·n′이다. 거의 0인 값은 raster의 작은 위치 변화에도 부호가 뒤바뀐다. `near_zero_release`는 절대값 0.18 m/s 이내를 표시한다. 이 구간의 방향 판정과 제품 전체에 대한 단일 부호 판정은 구분해야 한다.

참고로 별도 Niigata/JTTA 자료의 SHA-256은 `1593e6d334ba28d31b016e14e3154c7587acaf4331294ef3996ea2dfbd5fa6ff`이다. 이 발표의 제품 목록과 마찰 통계는 실험 범위 확인에 사용했고, Feint OX의 measurement로 합치지 않았다. [JTTA 발표](https://jtta-coach.s3.ap-northeast-1.amazonaws.com/pdf/investigation_a/23_15.pdf)
